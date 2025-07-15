import pandas as pd
from datasets import Dataset

# ✅ Step 1: Load your cleaned CSV
df = pd.read_csv("spam_dataset_cleaned.csv")

# ✅ (Optional) Preview the first 5 rows
print("📊 Preview of Dataset:")
print(df.head())

# ✅ Step 2: Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df)
print("✅ Loaded as Hugging Face Dataset:")
print(dataset)

# ✅ Step 3: Split into train and test sets (90% / 10%)
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)

# ✅ Convert each to HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

print(f"\n✅ Train size: {len(train_dataset)} | Test size: {len(test_dataset)}")

# ✅ Save datasets to disk for reuse
train_dataset.save_to_disk("data/train_dataset")
test_dataset.save_to_disk("data/test_dataset")

print("💾 Saved to: data/train_dataset and data/test_dataset")
