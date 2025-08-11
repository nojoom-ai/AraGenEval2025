from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. Load Arabic Sentence-BERT model
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')  # works for Arabic

# 2. Compute style centroid for each author
author_texts = {...}  # Dict: {author_name: [list of real samples]}
author_centroids = {}

for author, texts in author_texts.items():
    embeddings = model.encode(texts, convert_to_numpy=True)
    centroid = np.mean(embeddings, axis=0)
    author_centroids[author] = centroid

# 3. Evaluate style distance for generated samples
generated_samples = [...]  # List of (generated_text, target_author) tuples

for text, target_author in generated_samples:
    embedding = model.encode(text, convert_to_numpy=True)
    centroid = author_centroids[target_author]
    similarity = cosine_similarity([embedding], [centroid])[0][0]
    
    print(f"Sample style similarity to {target_author}: {similarity:.4f}")
