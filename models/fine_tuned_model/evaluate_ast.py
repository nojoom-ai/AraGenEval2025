# models/fine_tuned_model/evaluate_ast.py

import pandas as pd
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from datasets import Dataset
import evaluate

model = MBartForConditionalGeneration.from_pretrained("./models/arabic_ast_mbart")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer.src_lang = "ar_AR"

df = pd.read_csv("validation.csv")
df["input_text"] = df["author"] + " [SEP] " + df["text_in_msa"]

# Generate
preds = []
for i, row in df.iterrows():
    input_ids = tokenizer(row["input_text"], return_tensors="pt", truncation=True, max_length=256).input_ids
    output_ids = model.generate(input_ids=input_ids, forced_bos_token_id=tokenizer.lang_code_to_id["ar_AR"])
    pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    preds.append(pred)

# Save to Excel (for submission)
df["style"] = preds
df[["id", "style"]].to_excel("predictions.xlsx", index=False)

# Evaluation
bleu = evaluate.load("bleu")
chrf = evaluate.load("chrf")
results_bleu = bleu.compute(predictions=preds, references=df["text_in_author_style"].tolist())
results_chrf = chrf.compute(predictions=preds, references=df["text_in_author_style"].tolist())

print("BLEU:", results_bleu)
print("chrF:", results_chrf)
