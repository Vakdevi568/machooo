
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate
import torch
import os
# 🧠 Meta LLaMA Guard 2 - 8B with proper auth
model_name = "meta-llama/Llama-3.2-1B-Instruct"
# Load tokenized datasets
train_dataset = load_from_disk("data/tokenized_dataset/train")
eval_dataset  = load_from_disk("data/tokenized_dataset/test")
print(f"✅ Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")
# Load tokenizer and model with auth token
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_auth_token=hf_token,
    trust_remote_code=True,
    revision="main"
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_auth_token=hf_token,
    torch_dtype=torch.float32,
    trust_remote_code=True,
    revision="main"
)
# Add pad token if needed and resize embeddings
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))
# Apply LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(model, peft_config)
# Compute metrics
evaluator = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=-1)
    # flatten
    preds = preds.view(-1)
    labels = torch.tensor(labels).view(-1)
    return evaluator.compute(predictions=preds, references=labels)
# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
# Training arguments
training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=30,
    logging_dir="./logs",
    load_best_model_at_end=True,
    save_total_limit=2,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    push_to_hub=False
)
# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)
# ✅ Train the model
trainer.train()
# ✅ Save best model
model.save_pretrained("./best_model")
tokenizer.save_pretrained("./best_model")
print("✅ Training complete. Best model saved.")
