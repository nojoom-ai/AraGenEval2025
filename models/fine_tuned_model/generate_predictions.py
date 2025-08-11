# generate_predictions.py

import pandas as pd
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Load model and tokenizer
model_path = "./models/arabic_ast_mbart"
model = MBartForConditionalGeneration.from_pretrained(model_path)
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer.src_lang = "ar_AR"

# Load validation data
df = pd.read_csv("data/validation.csv")  # Or wherever your validation set is

# Generate predictions
outputs = []
for _, row in df.iterrows():
    input_text = f"{row['author']} [SEP] {row['text_in_msa']}"
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256).input_ids

    output_ids = model.generate(
        input_ids=input_ids,
        forced_bos_token_id=tokenizer.lang_code_to_id["ar_AR"],
        max_new_tokens=256
    )
    prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    outputs.append(prediction)

# Save outputs
df["output"] = outputs
df["ground_truth"] = df["text_in_author_style"]  # required for evaluate.py
df[["id", "author", "output", "ground_truth"]].to_csv("evaluation/metrics_log.csv", index=False)
