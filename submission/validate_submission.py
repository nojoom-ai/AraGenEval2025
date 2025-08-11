import os
import sys
import pandas as pd
import zipfile
import json
import argparse

PRED_PATH = os.path.join(os.path.dirname(__file__), 'predictions.xlsx')
ZIP_PATH = os.path.join(os.path.dirname(__file__), 'predictions.zip')

def create_xlsx_from_json(json_path, pred_path):
    print(f"Reading JSON: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Accepts list of dicts, or dict with 'results' key
    if isinstance(data, dict) and 'results' in data:
        data = data['results']
    flat = [{
        'id': item['id'],
        'style': item['output']
    } for item in data]
    df = pd.DataFrame(flat)
    df.to_excel(pred_path, index=False)
    print(f"Wrote predictions.xlsx with {len(df)} rows.")

def validate_and_zip(pred_path, zip_path):
    # 1. Validate predictions.xlsx exists
    if not os.path.exists(pred_path):
        print(f"ERROR: {pred_path} not found.")
        sys.exit(1)
    # 2. Validate columns
    try:
        df = pd.read_excel(pred_path)
    except Exception as e:
        print(f"ERROR: Could not read {pred_path}: {e}")
        sys.exit(1)
    expected_cols = ['id', 'style']
    if list(df.columns) != expected_cols:
        print(f"ERROR: predictions.xlsx must have columns {expected_cols}, found {list(df.columns)}")
        sys.exit(1)
    # 3. Create ZIP
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(pred_path, arcname='predictions.xlsx')
    print(f"SUCCESS: Created {zip_path} for submission.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate and package predictions for submission.')
    parser.add_argument('--json', type=str, help='Path to generation log JSON file (optional).')
    args = parser.parse_args()
    if args.json:
        create_xlsx_from_json(args.json, PRED_PATH)
    validate_and_zip(PRED_PATH, ZIP_PATH)
