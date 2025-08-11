import torch
from datasets import Dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Config
model_name = "aubmindlab/bert-base-arabertv2"  # You can use CAMeLBERT, etc.
num_labels = 21  # Adjust to your number of authors
batch_size = 16
epochs = 5

# 2. Load Dataset (Assuming list of dicts or convert your .txt properly)
# Example:
data = [
    {"text": "Sample sentence 1", "label": 0},
    {"text": "Another example", "label": 1},
    # ...
]

dataset = Dataset.from_list(data)
dataset = dataset.train_test_split(test_size=0.1)

# 3. Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(examples):
    return tokenizer(examples['text'], padding=True, truncation=True)

encoded_dataset = dataset.map(preprocess, batched=True)

# 4. Model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)

# 5. Metrics
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    acc = accuracy_score(p.label_ids, preds)
    f1 = f1_score(p.label_ids, preds, average="macro")
    return {"accuracy": acc, "f1": f1}

# 6. Training
args = TrainingArguments(
    output_dir="./style_classifier",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
