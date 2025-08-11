import os
import sys
import json
import time

# Add project root to Python path before any other imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import pandas as pd
import os
import argparse
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description='Compare metrics between a baseline and multiple model runs')
    parser.add_argument('fewshot_file', help='Path to the baseline (fewshot) metrics CSV file')
    parser.add_argument('comparison_files', nargs='+', help='Paths to one or more model metrics CSV files for comparison')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Read the baseline file
    df_fewshot = pd.read_csv(args.fewshot_file)
    print("Original fewshot columns:", df_fewshot.columns.tolist())
    
    # Convert all column names to lowercase for case-insensitive matching
    df_fewshot.columns = df_fewshot.columns.str.lower()
    print("Lowercase fewshot columns:", df_fewshot.columns.tolist())
    
    # Rename columns for clarity in baseline
    df_fewshot = df_fewshot.rename(columns={
        'bleu': 'bleu_fewshot',
        'chrf': 'chrf_fewshot',
        'bertscore': 'bertscore_fewshot'
    })
    print("Final fewshot columns:", df_fewshot.columns.tolist())
    
    # Merge all comparison files into the baseline
    merged = df_fewshot.copy()
    model_names = []
    for comp_file in args.comparison_files:
        df_finetuned = pd.read_csv(comp_file)
        print(f"Original finetuned columns for {comp_file}:", df_finetuned.columns.tolist())
        # Extract model name from the finetuned file path (get the parent directory name before 'run_...')
        model_path = Path(comp_file).parent
        model_name = model_path.parent.name  # This gets 'google_mt5-small' from the path
        model_names.append(model_name)
        # Convert all column names to lowercase for case-insensitive matching
        df_finetuned.columns = df_finetuned.columns.str.lower()
        print(f"Lowercase finetuned columns for {comp_file}:", df_finetuned.columns.tolist())
        # Rename columns for clarity
        df_finetuned = df_finetuned.rename(columns={
            'bleu': f'bleu_{model_name}',
            'chrf': f'chrf_{model_name}',
            'bertscore': f'bertscore_{model_name}'
        })
        print(f"Final finetuned columns for {comp_file}:", df_finetuned.columns.tolist())
        # Merge with the accumulating DataFrame
        merged = pd.merge(
            merged,
            df_finetuned[['author', 'count', f'bleu_{model_name}', f'chrf_{model_name}', f'bertscore_{model_name}']],
            on=['author', 'count'],
            how='inner'
        )
    # Calculate change columns for each model
    for model_name in model_names:
        merged[f'bleu_change_{model_name}'] = (merged[f'bleu_{model_name}'] - merged['bleu_fewshot']).apply(lambda x: f"{x:+.4f}")
        merged[f'chrf_change_{model_name}'] = (merged[f'chrf_{model_name}'] - merged['chrf_fewshot']).apply(lambda x: f"{x:+.4f}")
        merged[f'bertscore_change_{model_name}'] = (merged[f'bertscore_{model_name}'] - merged['bertscore_fewshot']).apply(lambda x: f"{x:+.4f}")
        # Round metric columns
        merged[[f'bleu_{model_name}', f'chrf_{model_name}', f'bertscore_{model_name}']] = merged[[f'bleu_{model_name}', f'chrf_{model_name}', f'bertscore_{model_name}']].round(4)
    merged[['bleu_fewshot', 'chrf_fewshot', 'bertscore_fewshot']] = merged[['bleu_fewshot', 'chrf_fewshot', 'bertscore_fewshot']].round(4)
    # Build output columns
    result_columns = ['author', 'count', 'bleu_fewshot'] + [f'bleu_{m}' for m in model_names] + [f'bleu_change_{m}' for m in model_names]
    result_columns += ['chrf_fewshot'] + [f'chrf_{m}' for m in model_names] + [f'chrf_change_{m}' for m in model_names]
    result_columns += ['bertscore_fewshot'] + [f'bertscore_{m}' for m in model_names] + [f'bertscore_change_{m}' for m in model_names]

    # Build output columns for the merged DataFrame
    result_columns = ['author', 'count', 'bleu_fewshot'] + [f'bleu_{m}' for m in model_names] + [f'bleu_change_{m}' for m in model_names]
    result_columns += ['chrf_fewshot'] + [f'chrf_{m}' for m in model_names] + [f'chrf_change_{m}' for m in model_names]
    result_columns += ['bertscore_fewshot'] + [f'bertscore_{m}' for m in model_names] + [f'bertscore_change_{m}' for m in model_names]
    result_df = merged[result_columns]
    # Output filename in baseline directory
    baseline_dir = Path(args.fewshot_file).parent
    output_filename = 'comparison_fewshot_multi_model.csv'
    output_path = baseline_dir / output_filename
    result_df.to_csv(output_path, index=False)
    print(f"Comparison saved to {output_path}")

if __name__ == "__main__":
    main()