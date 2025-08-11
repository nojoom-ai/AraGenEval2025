# AraGenEval2025/evaluation/evaluate.py

import os
import sys
import json
import time

# Add project root to Python path before any other imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from sacrebleu import corpus_bleu, corpus_chrf
from config import Config
from datetime import datetime
from bert_score import score as bert_score
import torch
import warnings
import os

import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# Toggle BERTScore computation
CALCULATE_BERTSCORE = True

def safe_print(*args, **kwargs):
    """Safely print Unicode text in Windows console with immediate output."""
    try:
        print(*args, **kwargs, flush=True)  
    except UnicodeEncodeError:
        # If we can't print the text, print a placeholder instead
        text = ' '.join(str(arg) for arg in args)
        encoded = text.encode('ascii', errors='replace').decode('ascii')
        print(encoded, **kwargs, flush=True)  

def evaluate_bleu_chrf(preds, refs):
    bleu = corpus_bleu(preds, [refs])
    chrf = corpus_chrf(preds, [refs])
    return bleu.score, chrf.score

def evaluate_bertscore(preds, refs):
    P, R, F1 = bert_score(refs, preds, model_type='roberta-base', lang='en', verbose=False)
    return torch.mean(F1).item()


def evaluate_per_author(df):
    """
    Compute BLEU and chrF per author and save results.
    """
    records = []
    total_authors = len(df['author'].unique())
    
    # Calculate metrics per author
    for i, (author, group) in enumerate(df.groupby('author'), 1):
        safe_print(f"\n[PROGRESS] Processing author {i}/{total_authors}")
        safe_print(f"[AUTHOR] {author}")
        safe_print(f"[SAMPLES] {len(group)} samples")
        
        preds = group['output'].astype(str).tolist()
        refs = group['ground_truth'].astype(str).tolist()
        
        # Calculate BLEU and chrF scores
        bleu, chrf = evaluate_bleu_chrf(preds, refs)
        safe_print(f"[SCORES] BLEU: {bleu:.2f}, chrF: {chrf:.2f}")
        
        bertscore = None
        if CALCULATE_BERTSCORE:
            safe_print(f"[BERTSCORE] Calculating...")
            bertscore = evaluate_bertscore(preds, refs)
            safe_print(f"[BERTSCORE] Done - Score: {bertscore:.4f}")
        
        records.append({
            'author': author,
            'count': len(group),
            'BLEU': bleu,
            'chrF': chrf,
            'BERTScore': bertscore
        })
        
        safe_print(f"[PROGRESS] Completed {i}/{total_authors} authors")
    
    # Create DataFrame from author metrics
    author_df = pd.DataFrame(records).sort_values(by='BLEU', ascending=False)
    
    # Calculate overall metrics
    all_preds = df['output'].astype(str).tolist()
    all_refs = df['ground_truth'].astype(str).tolist()
    overall_bleu, overall_chrf = evaluate_bleu_chrf(all_preds, all_refs)
    overall_bertscore = None
    if CALCULATE_BERTSCORE:
        safe_print("[BERTSCORE] Calculating overall BERTScore...")
        overall_bertscore = evaluate_bertscore(all_preds, all_refs)
        safe_print("[DONE] Overall BERTScore complete.")
    
    # Add overall metrics as a summary row
    summary_row = pd.DataFrame([{
        'author': 'OVERALL',
        'count': len(df),
        'BLEU': overall_bleu,
        'chrF': overall_chrf,
        'BERTScore': overall_bertscore if overall_bertscore else None
    }])
    
    # Combine author metrics with summary row
    result_df = pd.concat([author_df, summary_row], ignore_index=True)
    
    # Save to CSV
    result_df.to_csv(Config.PER_AUTHOR_METRICS_LOG, index=False)
    return result_df


def main():
    # Load logs
    df = pd.read_csv(Config.LABELED_LOG)
    df = df.dropna(subset=['output', 'ground_truth'])

    preds = df['output'].astype(str).tolist()
    refs = df['ground_truth'].astype(str).tolist()

    # Compute per-author metrics (includes overall metrics as a row)
    safe_print("\n[REPORT] Per-Author BLEU & chrF:")
    author_df = evaluate_per_author(df)
    
    # Get overall metrics from the last row of the DataFrame
    overall_row = author_df[author_df['author'] == 'OVERALL'].iloc[0]
    
    # Print summary
    safe_print("\n[SUMMARY] Overall Evaluation:")
    safe_print(f"[DATE] Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    safe_print(f"[SAMPLES] Samples Evaluated: {len(df)}")
    safe_print(f"[BLEU] BLEU: {overall_row['BLEU']:.2f}")
    safe_print(f"[CHR-F] chrF: {overall_row['chrF']:.2f}")
    if overall_row['BERTScore'] is not None:
        safe_print(f"[BERTSCORE] BERTScore: {overall_row['BERTScore']:.4f}")
    else:
        safe_print(f"[BERTSCORE] BERTScore: N/A")
    
    # Print per-author metrics (excluding the OVERALL row)
    safe_print("\n[REPORT] Per-Author Metrics:")
    safe_print(author_df.to_markdown(index=False))


if __name__ == "__main__":
    # Suppress HuggingFace symlink warning
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    warnings.filterwarnings("ignore", message=".*symlinks.*", category=UserWarning)
    main()
