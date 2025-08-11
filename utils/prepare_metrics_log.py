# AraGenEval2025/utils/prepare_metrics_log.py

import os
import sys
import json
import time
import argparse

# Add project root to Python path before any other imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)



import os
import sys
import json
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from config import Config

def extract_and_sample(
    file_path: str,
    sample_mode: str = "all",
    sample_size: float | int | None = None,
    random_state: int = 42
):
    # 1) Load raw JSON
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2) Flatten out only the fields we need
    flat = [{
        "id": item["id"],
        "author": item["author"],
        "output": item["output"],
        "ground_truth": item["ground_truth"]
    } for item in data]
    df = pd.DataFrame(flat)

    # 3) Apply sampling if requested
    if sample_mode == "random":
        if sample_size is None:
            raise ValueError("For 'random' mode, --sample_size must be provided.")
        n = int(sample_size) if isinstance(sample_size, int) else int(len(df) * float(sample_size))
        df = df.sample(n=n, random_state=random_state)

    elif sample_mode == "stratified":
        if sample_size is None:
            raise ValueError("For 'stratified' mode, --sample_size must be provided for stratified splitting.")
        # train_test_split with stratify returns sampled df and discard the rest
        df, _ = train_test_split(
            df,
            train_size=sample_size,
            stratify=df["author"],
            random_state=random_state
        )

    # "all" mode does nothing

    # 4) Ensure output folder exists
    out_csv = Config.LABELED_LOG
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    # 5) Save CSV
    df.to_csv(out_csv, index=False)
    print(f"[DONE] Wrote {len(df)} rows to CSV: {out_csv}")

    # 6) Save JSON
    out_json = os.path.splitext(out_csv)[0] + ".json"
    df.to_json(out_json, orient="records", force_ascii=False, lines=False)
    print(f"[DONE] Wrote {len(df)} rows to JSON: {out_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract labeled data from generation log and optionally sample it"
    )
    parser.add_argument("file_path", type=str, help="Path to the raw generation-log JSON file")
    parser.add_argument(
        "--sample_mode",
        type=str,
        choices=["all", "random", "stratified"],
        default="all",
        help="Sampling strategy: 'all' (no sample), 'random', or 'stratified' on authors"
    )
    parser.add_argument(
        "--sample_size",
        type=float,
        default=None,
        help="If random: float in (0,1] or int count. If stratified: float proportion (0,1]"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    extract_and_sample(
        file_path=args.file_path,
        sample_mode=args.sample_mode,
        sample_size=args.sample_size,
        random_state=args.random_state
    )
