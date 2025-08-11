import os
import sys
import json
import time
# Add project root to Python path before any other imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from config import Config

# Load model
model_path = "models/fine_tuned_model/model_weights/"
model = MBartForConditionalGeneration.from_pretrained(model_path)
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer.src_lang = "ar_AR"

# Load validation set
val_df = pd.read_csv(Config.VAL_FILE)
val_df["input_text"] = val_df["author"] + " [SEP] " + val_df["text_in_msa"]

# Inference
outputs = []
for _, row in val_df.iterrows():
    input_ids = tokenizer(row["input_text"], return_tensors="pt", truncation=True, max_length=256).input_ids
    output_ids = model.generate(
        input_ids=input_ids,
        forced_bos_token_id=tokenizer.lang_code_to_id["ar_AR"],
        max_new_tokens=256
    )
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    outputs.append(generated)

# Save for evaluation
val_df["output"] = outputs
val_df["ground_truth"] = val_df["text_in_author_style"]
val_df[["id", "author", "output", "ground_truth"]].to_csv("evaluation/metrics_log.csv", index=False)
