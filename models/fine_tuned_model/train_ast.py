# models/fine_tuned_model/train_ast.py

import os
import sys
import json
import time
# Add project root to Python path before any other imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from transformers import MBartForConditionalGeneration, MBart50Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset
import pandas as pd
from models.fine_tuned_model.prepare_dataset import tokenize_data
from config import Config

# 1. Load data
try:
    # Try reading with openpyxl engine first
    train_df = pd.read_excel(Config.TRAIN_FILE, engine='openpyxl')
    print("Successfully loaded the training data")
    print(f"Number of training examples: {len(train_df)}")
    
    # Create input and target text
    train_df["input_text"] = train_df["author"].astype(str) + " [SEP] " + train_df["text_in_msa"].astype(str)
    train_df["target_text"] = train_df["text_in_author_style"].astype(str)
    
    # Convert to HuggingFace dataset
    train_dataset = Dataset.from_pandas(train_df[["input_text", "target_text"]])
    print("Successfully created training dataset")
    
except Exception as e:
    print(f"Error loading training data: {str(e)}")
    print("Please ensure the file exists at 'data/AuthorshipStyleTransferTrain.xlsx' and is a valid Excel file")
    sys.exit(1)

# 2. Load tokenizer & model
tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer.src_lang = "ar_AR"
tokenizer.tgt_lang = "ar_AR"

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# 3. Tokenize with optimizations
print("Tokenizing dataset (this may take a while)...")
train_dataset = train_dataset.map(
    lambda x: tokenize_data(x, tokenizer),
    batched=True,
    batch_size=1000,  # Process in larger batches
    num_proc=4,  # Use multiple processes
    remove_columns=train_dataset.column_names,  # Remove unused columns to save memory
    load_from_cache_file=True  # Cache the tokenized dataset
)
print("Tokenization complete")

# 4. Training args - Optimized for GPU/CPU training
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# Common training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="models/fine_tuned_model/model_weights/",
    num_train_epochs=3,
    per_device_train_batch_size=8 if device == 'cuda' else 4,  # Larger batch size for GPU
    gradient_accumulation_steps=2 if device == 'cuda' else 4,  # Fewer accumulation steps for GPU
    learning_rate=5e-5,
    save_total_limit=2,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),  # Enable FP16 only if CUDA is available
    bf16=torch.cuda.is_bf16_supported(),  # Use BF16 if supported
    logging_steps=50,
    no_cuda=not torch.cuda.is_available(),  # Auto-detect CUDA
    save_steps=1000,
    dataloader_num_workers=4 if device == 'cuda' else 2,  # More workers for GPU
    dataloader_pin_memory=torch.cuda.is_available(),  # Pin memory for GPU
    gradient_checkpointing=True,  # Reduce memory usage
    optim="adamw_torch",
    report_to="none",
    max_grad_norm=1.0,
    eval_strategy="steps",
    eval_steps=500,  # Evaluate every 500 steps
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
)

# Add memory tracking
try:
    import psutil
    process = psutil.Process()
    print(f"Memory usage before training: {process.memory_info().rss / 1024 / 1024:.2f} MB")
except ImportError:
    print("psutil not available, skipping memory tracking")

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

# Print model size
print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# 5. Train with progress tracking
import time
from tqdm import tqdm
from datetime import datetime

print(f"Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
start_time = time.time()

try:
    trainer.train()
    training_time = time.time() - start_time
    print(f"Training completed in {training_time/3600:.2f} hours")
except KeyboardInterrupt:
    print("\nTraining interrupted. Saving current progress...")
    trainer.save_model("models/fine_tuned_model/model_weights/checkpoint-interrupted")
    print("Progress saved. Exiting...")
    sys.exit(0)

# 6. Save final model
trainer.save_model("models/fine_tuned_model/model_weights/")
