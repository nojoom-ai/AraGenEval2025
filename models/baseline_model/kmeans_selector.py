# models/baseline_gemini/kmeans_selector.py


import os
import sys
import time
import argparse

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
 
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

from config import Config

def select_representative_examples(texts, embeddings, k):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(embeddings)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    representatives = []
    for i in range(k):
        cluster_indices = np.where(labels == i)[0]
        cluster_embeddings = embeddings[cluster_indices]
        distances = np.linalg.norm(cluster_embeddings - cluster_centers[i], axis=1)
        closest_index = cluster_indices[np.argmin(distances)]
        representatives.append(closest_index)
    return representatives

def extract_kmeans_examples_all_authors(train_path, output_csv_path, k=3, embedding_model='all-MiniLM-L6-v2'):
    df = pd.read_excel(train_path)
    model = SentenceTransformer(embedding_model)

    collected_rows = []
    for author in df['author'].unique():
        subset = df[df['author'] == author]
        texts = subset['text_in_msa'].tolist()
        embeddings = model.encode(texts, convert_to_numpy=True)

        reps_idx = select_representative_examples(texts, embeddings, k)
        reps_df = subset.iloc[reps_idx][['text_in_msa', 'text_in_author_style']].copy()
        reps_df['author'] = author

        collected_rows.append(reps_df)

        print(f"✅ Selected {k} examples for author: {author}")

    final_df = pd.concat(collected_rows, ignore_index=True)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    final_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"📦 Saved all few-shot examples to: {output_csv_path}")

if __name__ == "__main__":
    # Ensure the directory for the output file exists
    output_dir = os.path.dirname(Config.FEWSHOT_EXAMPLES_FILE)
    if output_dir:  # Only create directory if there's a path (not just a filename)
        os.makedirs(output_dir, exist_ok=True)
    
    extract_kmeans_examples_all_authors(
        train_path=Config.TRAIN_FILE,
        output_csv_path=Config.FEWSHOT_EXAMPLES_FILE,
        k=Config.NUM_FEWSHOT_EXAMPLES
    )
