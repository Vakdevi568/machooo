from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer

# ✅ Correct model repo ID (no spaces)
model_name = "meta-llama/Llama-3.2-1B-Instruct" # Replace with your actual token

# ✅ Load datasets from disk
dataset = DatasetDict({
    "train": load_from_disk("data/train_dataset"),
    "test":  load_from_disk("data/test_dataset"),
})

print(f"✅ Loaded train: {len(dataset['train'])}, test: {len(dataset['test'])}")

# ✅ Load tokenizer with HF token
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=hf_token,
    trust_remote_code=True,
    revision="main"
)

# ✅ Add [PAD] token if missing
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

# ✅ Tokenization function
def tokenize(batch):
    return tokenizer(
        batch["clean_text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )

# ✅ Columns to remove (except the text column)
keep_col = "clean_text"
columns_to_remove = [col for col in dataset["train"].column_names if col != keep_col]

# ✅ Tokenize both splits
dataset = dataset.map(
    tokenize,
    batched=True,
    remove_columns=columns_to_remove
)

# ✅ Save tokenized data
dataset["train"].save_to_disk("data/tokenized_dataset/train")
dataset["test"].save_to_disk("data/tokenized_dataset/test")

print("✅ Tokenized dataset saved successfully.")
