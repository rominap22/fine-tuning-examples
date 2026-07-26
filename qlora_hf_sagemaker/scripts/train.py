import argparse
import gc
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    return p.parse_args()


def format_chat(example, tokenizer):
    return tokenizer.apply_chat_template(example["messages"], tokenize=False)


def main():
    args = parse_args()
    hf_token = os.environ["HF_TOKEN"]

    train_dir = os.environ["SM_CHANNEL_TRAIN"]
    val_dir = os.environ["SM_CHANNEL_VAL"]
    model_dir = os.environ["SM_MODEL_DIR"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    # QLoRA: 4-bit is a loading flag applied at runtime, not a stored file format.
    # The weights download as fp16 and get compressed to 4-bit as they enter GPU memory.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    dataset = load_dataset("json", data_files={
        "train": os.path.join(train_dir, "train.jsonl"),
        "validation": os.path.join(val_dir, "val.jsonl"),
    })

    # Masks loss to -100 for every token before the assistant turn begins, so
    # gradient updates focus on the answer, not on re-learning the question.
    # This marker matches Llama 3.1's chat template header for the assistant role.
    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    sft_config = SFTConfig(
        output_dir="/opt/ml/checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        max_seq_length=1024,
        lr_scheduler_type="cosine",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora_config,
        processing_class=tokenizer,
        formatting_func=lambda example: format_chat(example, tokenizer),
        data_collator=collator,
    )
    trainer.train()

    # Two-step merge: W_original + B*A requires matching precision, and the
    # base model is still loaded in 4-bit here, so merging now would produce
    # garbage. Save the adapter (always fp16), free the 4-bit model from GPU
    # memory, then reload the base in full fp16 before merging.
    adapter_dir = "/opt/ml/adapter"
    trainer.model.save_pretrained(adapter_dir)
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    base = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float16, device_map="auto", token=hf_token,
    )
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(model_dir, safe_serialization=True)
    tokenizer.save_pretrained(model_dir)


if __name__ == "__main__":
    main()
