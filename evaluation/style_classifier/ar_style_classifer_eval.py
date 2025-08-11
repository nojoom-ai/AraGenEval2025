# Load Trained Classifier
model_path = "./style_classifier/checkpoint-best"  # Or latest saved checkpoint
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Evaluate Generated Outputs
generated_texts = ["your generated text 1", "text 2", ...]
target_authors = [0, 1, ...]

correct = 0
total = len(generated_texts)

for text, target_author in zip(generated_texts, target_authors):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_author = torch.argmax(probs, dim=-1).item()
    
    if pred_author == target_author:
        correct += 1

style_match_accuracy = correct / total
print(f"Style Match Accuracy: {style_match_accuracy:.4f}")
