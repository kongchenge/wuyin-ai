#!/usr/bin/env python3
"""
QLoRA fine-tuning script for Qwen2.5-1.5B on custom JSONL data.
Hardware requirement: >= 6GB VRAM (4-bit quantized base model + LoRA r=8).

Install dependencies before running:
    pip install torch transformers datasets accelerate peft bitsandbytes trl
    pip install ninja packaging
    pip install flash-attn --no-build-isolation   # optional, for speed

Usage:
    python train_wuyin.py

Data format (wuyin_train.jsonl):
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer

# ── Config ────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-1.5B"           # or "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_PATH = "wuyin_train.jsonl"        # local JSONL file
OUTPUT_DIR = "wuyin-lora-adapter"
NUM_EPOCHS = 3
BATCH_SIZE = 2                            # per-device; adjust for VRAM
GRAD_ACCUM_STEPS = 4                      # effective batch = 2 * 4 = 8
MAX_SEQ_LENGTH = 2048
LEARNING_RATE = 2e-4
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# ── 4-bit quantization config ────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ── Load base model ──────────────────────────────────────────────────────
print(f"[1/5] Loading base model: {MODEL_ID}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2",  # remove if flash-attn not installed
)
model.config.use_cache = False                     # required for gradient checkpointing
model = prepare_model_for_kbit_training(model)     # prep for LoRA

# ── Load tokenizer ───────────────────────────────────────────────────────
print(f"[2/5] Loading tokenizer: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token           # Qwen has no pad token by default
tokenizer.padding_side = "right"

# ── LoRA config ──────────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected: ~2-3M trainable params (<0.2% of 1.5B)

# ── Load dataset ─────────────────────────────────────────────────────────
print(f"[3/5] Loading dataset: {DATASET_PATH}")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

# ── Formatting function for chat template ────────────────────────────────
def format_chat(example):
    """Convert {'messages': [...]} into tokenizer chat template string."""
    messages = example["messages"]
    # Apply Qwen2.5 chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


dataset = dataset.map(format_chat, remove_columns=dataset.column_names)

# ── Training arguments ───────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    bf16=True,                                  # set to False + use fp16 if GPU doesn't support bf16
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="paged_adamw_8bit",                   # memory-efficient optimizer
    dataloader_num_workers=2,
    report_to="none",                           # disable wandb; set to "wandb" to enable
    remove_unused_columns=False,
    seed=42,
)

# ── SFT Trainer ──────────────────────────────────────────────────────────
print(f"[4/5] Starting training ({NUM_EPOCHS} epochs)...")
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
)

trainer.train()

# ── Save adapter ─────────────────────────────────────────────────────────
print(f"[5/5] Saving LoRA adapter to: {OUTPUT_DIR}")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Done! Adapter saved. Merge with base model using:")
print(f"  python -m peft_utils.merge {MODEL_ID} {OUTPUT_DIR} merged-model/")
