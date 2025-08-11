import os
import sys
import json
import time

# Add project root to Python path before any other imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import csv
import time
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Any
import google.generativeai as genai
from config import Config

genai.configure(api_key=Config.GEMINI_API_KEY)

# List available Gemini models at startup for debugging
try:
    available_models = genai.list_models()
    print("[Gemini] Available models:")
    for m in available_models:
        print(f" - {m.name}")
except Exception as e:
    print(f"[Gemini] Could not list models: {e}")

model = genai.GenerativeModel('models/gemini-1.5-pro-latest')


# --- Debug utility ---
DEBUG = True

def debug(msg):
    if DEBUG:
        print(msg)

def load_demonstrations(path: str) -> Dict[str, List[Tuple[str, str]]]:
    author_demos = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            neutral = row.get("text_in_msa") or row.get("\ufefftext_in_msa") or row.get("text_in_msa ") or row.get(" text_in_msa")
            styled = row.get("text_in_author_style") or row.get("text_in_author_style ") or row.get(" text_in_author_style")
            author = row.get("author") or row.get("author ") or row.get(" author")
            if neutral is None or styled is None or author is None:
                debug(f"Row {i} keys = {list(row.keys())}")
                debug(f"Row {i} raw = {row}")
                raise KeyError(f"One of the expected keys not found in row: {row}")
            debug(f"Loaded demonstration for author: {author}")
            neutral = neutral.strip()
            styled = styled.strip()
            author = author.strip()
            author_demos[author].append((neutral, styled))
    return author_demos

def make_prompt(author: str, demos: List[Tuple[str, str]], neutral_text: str, generated_text: str) -> str:
    instruction = (
        "[Instruction]\n"
        "You are an impartial judge. Evaluate how well the generated text adopts the target author's style, while preserving meaning and fluency.\n"
        "First, give your score for the generated stylized text in the format [Score]x/10 (where x is an integer from 1 to 10).\n"
        "Then, provide your reasoning for the score, referencing both the style transfer and meaning preservation.\n"
        "The response format MUST be:\n[Score]x/10\nReasoning: ..."
    )

    demo_section = "[The Start of Demonstrations]\n"
    for neutral, styled in demos:
        demo_section += f"Neutral Text: {neutral}\nAuthor-stylized Text: {styled}\n"
    demo_section += "[The End of Demonstrations]"

    eval_section = (
        f"\n\n[Original Neutral Text]\n{neutral_text}\n\n"
        f"[Generated Stylized Text]\n{generated_text}"
    )

    return instruction + "\n" + demo_section + "\n" + eval_section


def rescore_missing_samples(output_path: str, demonstration_path: str, generation_log_path: str):
    debug(f"[Rescore] Loading output file: {output_path}")
    with open(output_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    missing_ids = [entry["id"] for entry in results if entry["score"] is None]
    if not missing_ids:
        debug("[Rescore] No missing samples to rescore.")
        return
    debug(f"[Rescore] Found {len(missing_ids)} missing samples.")
    demos_per_author = load_demonstrations(demonstration_path)
    debug(f"[Rescore] Loading generation log: {generation_log_path}")
    with open(generation_log_path, "r", encoding="utf-8") as f:
        gen_log = json.load(f)
    gen_by_id = {sample["id"]: sample for sample in gen_log}
    for entry in results:
        if entry["score"] is None:
            sample_id = entry["id"]
            gen_sample = gen_by_id.get(sample_id)
            if not gen_sample:
                debug(f"[Rescore] Sample id {sample_id} not found in generation log. Skipping.")
                continue
            author = gen_sample["author"]
            neutral = gen_sample["ground_truth"]
            generated = gen_sample["output"]
            if author in demos_per_author:
                new_result = rate_single_sample(author, demos_per_author[author], neutral, generated, sample_id)
                entry.update(new_result)
                debug(f"[Rescore] Updated sample_id: {sample_id}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    debug("[Rescore] Finished updating missing samples.")

def extract_score(response: str) -> int:
    matches = re.findall(r'\[Score\](\d+)', response)
    if matches:
        return int(matches[-1])
    matches = re.findall(r'(\d+)/10', response)
    if matches:
        return int(matches[-1])
    return None


def rate_single_sample(author: str, demos: List[Tuple[str, str]], neutral_text: str, generated_text: str, sample_id: Any) -> Dict[str, Any]:
    debug(f"[Rate] Creating prompt for author: {author}, sample_id: {sample_id}")
    prompt = make_prompt(author, demos, neutral_text, generated_text)
    max_retries = 5
    attempt = 0
    score = None
    result_text = None
    while score is None and attempt < max_retries:
        try:
            debug(f"[Rate] Sending prompt to Gemini API... (attempt {attempt+1})")
            response = model.generate_content(prompt)
            result_text = response.text
            debug(f"[Rate] Received response for sample_id: {sample_id}")
        except Exception as e:
            result_text = f"ERROR: {e}"
            debug(f"[Rate] ERROR for sample_id {sample_id}: {e}")
        score = extract_score(result_text)
        debug(f"[Rate] Extracted score: {score} for sample_id: {sample_id} (attempt {attempt+1})")
        attempt += 1
        if score is None and attempt < max_retries:
            debug(f"[Rate] Retrying sample_id: {sample_id} (attempt {attempt+1})...")
            time.sleep(1.5)  # Wait a bit before retrying
    return {
        "id": sample_id,
        "author": author,
        "prompt": prompt,
        "response": result_text,
        "score": score
    }


def evaluate_all(generation_log_path: str, demonstration_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
    debug(f"[Evaluate] Loading demonstrations...")
    demos_per_author = load_demonstrations(demonstration_path)

    debug(f"[Evaluate] Loading generation log from: {generation_log_path}")
    with open(generation_log_path, "r", encoding="utf-8") as f:
        generation_data = json.load(f)

    results = []
    score_per_author = defaultdict(list)

    debug(f"[Evaluate] Starting evaluation of {len(generation_data)} samples...")
    total = len(generation_data)
    progress_step = max(1, total // 20)  # 5% steps
    for idx, sample in enumerate(generation_data):
        author = sample["author"]
        neutral = sample["ground_truth"]
        generated = sample["output"]
        sample_id = sample["id"]

        if author not in demos_per_author:
            print(f"⚠️ No demonstrations for author {author}. Skipping.")
            continue

        # Only print progress every 5%
        if idx % progress_step == 0 or idx == total - 1:
            debug(f"[Evaluate] Progress: {idx+1}/{total} ({(idx+1)/total*100:.1f}%)")

        result = rate_single_sample(author, demos_per_author[author], neutral, generated, sample_id)
        results.append(result)

        if result["score"] is not None:
            score_per_author[author].append(result["score"])

        time.sleep(1.2)  # Gemini API rate-limiting

    debug(f"[Evaluate] Evaluation complete. {len(results)} samples rated.")
    return results, score_per_author


def compute_score_summary(score_per_author: Dict[str, List[int]]) -> Dict[str, Any]:
    summary = {}
    total_score = 0
    total_count = 0

    for author, scores in score_per_author.items():
        avg = sum(scores) / len(scores)
        summary[author] = {
            "average": avg,
            "count": len(scores),
            "total": sum(scores)
        }
        total_score += sum(scores)
        total_count += len(scores)

    if total_count > 0:
        summary["overall"] = {
            "average": total_score / total_count,
            "count": total_count,
            "total": total_score
        }
    else:
        summary["overall"] = {
            "average": 0.0,
            "count": 0,
            "total": 0
        }

    return summary


def main():
    debug("[Main] Starting Gemini rater workflow...")
    # 1. Model type and name selection
    MODEL_OPTIONS = {
        "baseline_model": ["fewshot_gemini"],
        "fine_tuned_model": ["google_mt5-small", "UBC-NLP_AraT5-base","facebook_mbart-large-50-many-to-many-mmt"]
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

    DEMONSTRATION_PATH = "evaluation/fewshot_examples_per_author.csv"
    GENERATION_LOG_PATH = f"models/{MODEL_TYPE}/results/{MODEL_NAME}/generation_log.json"
    OUTPUT_LOG_PATH = f"gemini_rater_log_{MODEL_NAME}.json"

    debug(f"[Main] Loading demonstrations from: {DEMONSTRATION_PATH}")

    results, score_per_author = evaluate_all(GENERATION_LOG_PATH, DEMONSTRATION_PATH)
    debug(f"[Main] Finished evaluation. Writing results to: {OUTPUT_LOG_PATH}")

    with open(OUTPUT_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # rescore_missing_samples(OUTPUT_LOG_PATH, DEMONSTRATION_PATH, GENERATION_LOG_PATH)

    # Load updated results for summary
    # with open(OUTPUT_LOG_PATH, "r", encoding="utf-8") as f:
    #     results = json.load(f)
    # score_per_author = defaultdict(list)
    # for entry in results:
    #     if entry["score"] is not None:
    #         score_per_author[entry["author"]].append(entry["score"])

    summary = compute_score_summary(score_per_author)
    print("\n🔍 Evaluation Summary:")
    for author, stats in summary.items():
        print(f"  {author}: Average = {stats['average']:.2f}, Count = {stats['count']}")

    # Save summary to CSV
    csv_path = f"per_author_gemini_rating_{MODEL_NAME}.csv"
    with open(csv_path, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["author", "count", "average"])
        overall_count = 0
        overall_sum = 0.0
        for author, stats in summary.items():
            writer.writerow([author, stats['count'], f"{stats['average']:.2f}"])
            overall_count += stats['count']
            overall_sum += stats['average'] * stats['count']
        overall_avg = overall_sum / overall_count if overall_count else 0.0
        writer.writerow(["overall", overall_count, f"{overall_avg:.2f}"])
    debug(f"[Main] Saved summary CSV to: {csv_path}")
    debug("[Main] Workflow complete.")

if __name__ == "__main__":
    main()