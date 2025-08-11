# AraGenEval2025/models/baseline_gemini/generate_baseline.py

import os
import sys
import time
import argparse

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import json
import random
import google.generativeai as genai
from config import Config
from prompts.prompt_templates import build_fewshot_prompt

def call_gemini_api(prompt: str) -> str:
    max_retries = 3
    retry_delay = 10  # seconds
    
    for attempt in range(max_retries):
        try:
            print(f"[DEBUG] Attempting to call Gemini API (attempt {attempt + 1}/{max_retries})...")
            genai.configure(api_key=Config.GEMINI_API_KEY)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            # Add timeout and request options
            response = model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": 2048,
                    "temperature": 0.7,
                },
                request_options={"timeout": 60}  # 60 seconds timeout
            )
            
            if not response.text:
                print("[WARNING] Received empty response from Gemini API")
                continue
                
            return response.text.strip()
            
        except Exception as e:
            error_msg = str(e)
            print(f"[Gemini ERROR] Attempt {attempt + 1}/{max_retries} failed: {error_msg}")
            if attempt < max_retries - 1:  # Don't sleep on the last attempt
                print(f"[DEBUG] Retrying in {retry_delay} seconds...")
                import time
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print("[ERROR] All retry attempts failed")
                return f"[ERROR: Generation failed after {max_retries} attempts: {error_msg}]"

def load_fewshot_examples(csv_path, author_name=None):
    """
    Load few-shot examples from a CSV file.
    
    Args:
        csv_path: Path to the CSV file containing few-shot examples
        author_name: Optional author name to filter examples. If None, returns all examples.
        
    Returns:
        List of tuples containing (input_text, output_text) pairs
    """
    try:
        print(f"Loading few-shot examples from {csv_path}...")
        fewshot_df = pd.read_csv(csv_path)
        
        # Filter by author if specified
        if author_name is not None:
            fewshot_df = fewshot_df[fewshot_df['author'] == author_name]
            
        if fewshot_df.empty:
            print(f"[WARNING] No few-shot examples found for author: {author_name}")
            return []
            
        # Return list of (input, output) pairs
        return list(zip(fewshot_df['text_in_msa'], fewshot_df['text_in_author_style']))
        
    except Exception as e:
        print(f"[ERROR] Failed to load few-shot examples: {str(e)}")
        return []

def sample_examples(train_df, author_name, k=3):
    """
    Sample examples from the training dataset.
    
    Args:
        train_df: DataFrame containing training data
        author_name: Author name to sample examples for
        k: Number of examples to sample
        
    Returns:
        List of tuples containing (input_text, output_text) pairs
    """
    author_data = train_df[train_df['author'] == author_name]
    if author_data.empty:
        print(f"[WARNING] No training data found for author: {author_name}")
        return []
        
    sampled = author_data.sample(n=min(k, len(author_data)), random_state=42)
    return list(zip(sampled['text_in_msa'], sampled['text_in_author_style']))

def sample_validation_data(val_df, sample_size):
    """
    Sample a specified number of examples per author from the validation dataset.
    
    Args:
        val_df: DataFrame containing validation data
        sample_size: Number of samples to take per author
        
    Returns:
        DataFrame containing the sampled data
    """
    if sample_size is None:
        return val_df
        
    print(f"Sampling {sample_size} examples per author...")
    sampled_dfs = []
    for author in val_df['author'].unique():
        author_data = val_df[val_df['author'] == author]
        n_samples = min(sample_size, len(author_data))
        author_samples = author_data.sample(n=n_samples, random_state=42)
        sampled_dfs.append(author_samples)
    
    return pd.concat(sampled_dfs, ignore_index=True)

def process_text(author, text, train_df, num_fewshot_examples):
    """
    Process a single text through the style transfer pipeline.
    
    Args:
        author: Author name for style transfer
        text: Input text to process
        train_df: DataFrame containing training examples (used if few-shot file doesn't exist)
        num_fewshot_examples: Number of few-shot examples to use
        
    Returns:
        Tuple of (styled_output, prompt) containing the processed text and prompt used
    """
    # Try to load examples from the configured few-shot file, fall back to training data
    if os.path.exists(Config.FEWSHOT_EXAMPLES_FILE):
        examples = load_fewshot_examples(Config.FEWSHOT_EXAMPLES_FILE, author)[:num_fewshot_examples]
    else:
        examples = sample_examples(train_df, author, num_fewshot_examples)
        
    if not examples:
        print(f"[WARNING] No examples found for author: {author}")
        return "[ERROR: No examples found for author]", ""
    prompt = build_fewshot_prompt(author, examples, text)
    styled_output = call_gemini_api(prompt)
    return styled_output, prompt

def process_validation_data(val_df, train_df):
    """
    Process the validation dataset through the style transfer pipeline.
    
    Args:
        val_df: DataFrame containing validation data
        train_df: DataFrame containing training data for few-shot examples
        
    Returns:
        Tuple of (results, generations) containing the processing results
    """
    results = []
    generations = []
    total_rows = len(val_df)
    
    print(f"Starting processing of {total_rows} rows...")
    start_time = time.time()
    
    for idx, row in val_df.iterrows():
        try:
            # Progress reporting
            if idx > 0 and idx % 10 == 0:
                elapsed = time.time() - start_time
                rows_per_second = idx / elapsed if elapsed > 0 else 0
                remaining_time = (total_rows - idx) / rows_per_second if rows_per_second > 0 else 0
                print(f"Processed {idx}/{total_rows} rows | "
                      f"Speed: {rows_per_second:.2f} rows/sec | "
                      f"Elapsed: {elapsed/60:.1f} min | "
                      f"ETA: {remaining_time/60:.1f} min")
            
            author = row['author']
            text = row['text_in_msa']
            
            print(f"\n[{idx+1}/{total_rows}] Processing author: {author}")
            print(f"Input text length: {len(text)} characters")
            
            # Process the text
            print("Calling Gemini API...")
            styled_output, prompt = process_text(
                author=row['author'], 
                text=row['text_in_msa'],
                train_df=train_df,
                num_fewshot_examples=Config.NUM_FEWSHOT_EXAMPLES
            )
            print(f"Received response of length: {len(styled_output) if styled_output else 0} characters")
            
            # Store results
            results.append({"id": row['id'], "style": styled_output})
            generations.append({
                "id": row['id'],
                "author": author,
                "input_text": text,
                "ground_truth": row['text_in_author_style'],
                "prompt": prompt,
                "output": styled_output
            })
            
            # Save progress every 5 samples
            if (idx + 1) % 5 == 0:
                save_results(results, generations)
                print(f"Progress saved after {idx + 1} samples")
            
        except Exception as e:
            print(f"[ERROR] Failed to process row {idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            # Continue with next row even if one fails
    
    return results, generations

def save_results(results, generations):
    """Save the processing results to output files."""
    pd.DataFrame(results).to_excel(Config.OUTPUT_FILE, index=False)
    with open(Config.LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(generations, f, ensure_ascii=False, indent=2)
    pd.DataFrame(generations).to_csv(Config.LABELED_LOG, index=False)
    print(f"Results saved to {Config.OUTPUT_FILE}, {Config.LOG_FILE}, and {Config.LABELED_LOG}")

def main():
    parser = argparse.ArgumentParser(description='Generate baseline style transfer using Gemini API')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Number of samples to take per author (default: None, process all)')
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    train_df = pd.read_excel(Config.TRAIN_FILE, engine='openpyxl')
    val_df = pd.read_excel(Config.VAL_FILE, engine='openpyxl')
    
    # Sample data if requested
    sampled_val_df = sample_validation_data(val_df, args.sample_size)
    
    # Process validation data
    results, generations = process_validation_data(sampled_val_df, train_df)
    
    # Save results
    save_results(results, generations)

if __name__ == "__main__":
    main()