# models/fine_tuned_model/prepare_dataset.py

import pandas as pd
from datasets import Dataset

def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)
    
    # Make sure required columns exist
    assert {"text_in_msa", "author", "text_in_author_style"}.issubset(df.columns)
    
    df["input_text"] = df["author"] + " [SEP] " + df["text_in_msa"]
    df["target_text"] = df["text_in_author_style"]
    
    return Dataset.from_pandas(df[["input_text", "target_text"]])


def tokenize_data(example, tokenizer):
    model_inputs = tokenizer(
        example["input_text"], max_length=256, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        example["target_text"], max_length=256, truncation=True, padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

