"""
Run the baseline model pipeline and archive all results with timestamps.
"""

import os
import sys
import subprocess
import shutil
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config


def run_command(command, log_file):
    """Run a shell command and log its output to a file and console."""
    print(f"\n{'='*80}")
    print(f"RUNNING: {command}")
    print(f"{'='*80}")
    
    try:
        # Open log file with explicit UTF-8 encoding and error handling
        with open(log_file, 'a', encoding='utf-8', errors='replace') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"COMMAND: {command}\n")
            f.write(f"{'='*80}\n\n")
            f.flush()
            
            # Start the process with explicit UTF-8 encoding for the output
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
                encoding='utf-8',
                errors='replace'
            )
            
            # Stream output to both console and log file
            for line in process.stdout:
                try:
                    print(line, end='', flush=True)
                    f.write(line)
                    f.flush()
                except UnicodeEncodeError:
                    # Fallback for console if it can't handle the encoding
                    print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
                    f.write(line.encode('utf-8', 'replace').decode('utf-8'))
                    f.flush()
            
            process.wait()
            return process.returncode
            
    except Exception as e:
        print(f"\nERROR in run_command: {str(e)}")
        return 1


def create_timestamped_dir(base_dir):
    """Create a timestamped directory for storing results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def archive_files(output_dir, files_to_archive):
    """Archive important files from the baseline pipeline."""
    
    print("\nArchiving files:")
    for src_file in files_to_archive:
        if not os.path.exists(src_file):
            print(f"  - Skipping (not found): {src_file}")
            continue
            
        dest_file = os.path.join(output_dir, os.path.basename(src_file))
        print(f"  - Copying: {src_file} -> {dest_file}")
        shutil.copy2(src_file, dest_file)

def main():
    # 1. Model type and name selection
    MODEL_OPTIONS = {
        "baseline_model": ["fewshot_gemini"],
        "fine_tuned_model": ["google_mt5-small", "UBC-NLP_AraT5-base", "UBC-NLP_AraT5v2-base-1024", "facebook_mbart-large-50-many-to-many-mmt","agemagician_mlong-t5-tglobal-large" , "sultan_ArabicT5-49GB-base","moussaKam_AraBART","moussaKam_AraBART_with_lora","UBC-NLP_AraT5v2-base-1024_with_lora"]
    }
    print("Select MODEL TYPE:")
    for idx, key in enumerate(MODEL_OPTIONS.keys(), 1):
        print(f"  {idx}. {key}")
    while True:
        try:
            type_choice = int(input("Enter number for MODEL TYPE: "))
            MODEL_TYPE = list(MODEL_OPTIONS.keys())[type_choice-1]
            break
        except (ValueError, IndexError):
            print("Invalid choice. Try again.")

    print(f"\nAvailable models for '{MODEL_TYPE}':")
    for idx, name in enumerate(MODEL_OPTIONS[MODEL_TYPE], 1):
        print(f"  {idx}. {name}")
    while True:
        try:
            name_choice = int(input("Enter number for MODEL NAME: "))
            MODEL_NAME = MODEL_OPTIONS[MODEL_TYPE][name_choice-1]
            break
        except (ValueError, IndexError):
            print("Invalid choice. Try again.")

    print(f"\nSelected MODEL_TYPE: {MODEL_TYPE}")
    print(f"Selected MODEL_NAME: {MODEL_NAME}")


    # Create output directory
    output_base_dir = os.path.join("analysis", MODEL_TYPE , MODEL_NAME)   #, "fewshot_model"
    output_dir = create_timestamped_dir(output_base_dir)
    print(f"\nSaving results to: {os.path.abspath(output_dir)}")
    
    # Create log file
    log_file = os.path.join(output_dir, "pipeline.log")
    
    # Run each command in the pipeline
    commands = [
        ## uncomment is you want to rerun the best represnetative exmaple selection from train set
        # f"python models/baseline_gemini/kmeans_selector.py", 
        # f"python models/baseline_gemini/generate_baseline.py", # --sample-size 25

        f"python utils/prepare_metrics_log.py models/{MODEL_TYPE}/results/{MODEL_NAME}/generation_log.json",  #  --sample_mode stratified --sample_size 0.08 --random_state 42
        f"python evaluation/evaluate.py",
        # f"python evaluation/gemini_rater.py",
        # f"python evaluation/compare_metrics.py \"{Path('analysis/baseline_model/fewshot_gemini/run_20250630_162245/per_author_metrics.csv').as_posix()}\" \"{Path('evaluation/per_author_metrics.csv').as_posix()}\"",
        # f"python submission/validate_submission.py --json models/{MODEL_TYPE}/results/{MODEL_NAME}/generation_log_test.json"
    ]

    files_to_archive = [
        # Config.FEWSHOT_EXAMPLES_FILE,  # Few-shot examples
        "submission/predictions.xlsx",      #Config.OUTPUT_FILE,                            # Model predictions
        "models/{MODEL_TYPE}/results/{MODEL_NAME}/generation_log_test.json",                # Config.LOG_FILE,                # Generation logs
        Config.PER_AUTHOR_METRICS_LOG,                                                      # evaluation metrics results
        f"per_author_gemini_rating_{MODEL_NAME}.csv",
        # Config.LABELED_LOG,           # (predictions, label) pairs log data
    ]
    
    for cmd in commands:
        return_code = run_command(cmd, log_file)
        if return_code != 0:
            print(f"\nERROR: Command failed with return code {return_code}")
            print(f"See {log_file} for details.")
            return return_code
    
    # Archive all relevant files
    archive_files(output_dir,files_to_archive)
    
    print(f"\n{'='*80}")
    print(f"Pipeline completed successfully!")
    print(f"Results saved to: {os.path.abspath(output_dir)}")
    print(f"{'='*80}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
