import json
import os
import numpy as np
from sklearn.manifold import TSNE

# Path configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "thought-search-db.json")
VECTOR_PATH = os.path.join(BASE_DIR, "data", "thought-search-db.vectors.npy")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "viz-data.json")

def extract_visualization_data():
    print(f"Loading database from {DB_PATH}...")
    if not os.path.exists(DB_PATH) or not os.path.exists(VECTOR_PATH):
        print("Error: Database or vector file not found.")
        return

    with open(DB_PATH, "r", encoding="utf-8") as f:
        db_data = json.load(f)
    
    vectors = np.load(VECTOR_PATH)
    metadata = db_data.get("metadata", [])
    documents = db_data.get("documents", [])

    print(f"Loaded {len(vectors)} vectors with dimension {vectors.shape[1]}.")

    if len(vectors) < 2:
        print("Not enough data points for visualization.")
        return

    # 1. Dimensionality Reduction using t-SNE (3D for multi-dimensional feel)
    perplexity = min(30, len(vectors) - 1)
    print(f"Running t-SNE 3D (perplexity={perplexity})...")
    
    tsne = TSNE(
        n_components=3, 
        perplexity=perplexity, 
        random_state=42, 
        init='pca', 
        learning_rate='auto'
    )
    vectors_3d = tsne.fit_transform(vectors)

    # 2. Calculate Connectivity (Edges based on Cosine Similarity)
    # Vectors are already normalized in SimpleVectorDB.add_texts, 
    # but let's ensure normalization for safety.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors_norm = vectors / (norms + 1e-10)
    
    # Calculate all-to-all similarity matrix
    sim_matrix = np.dot(vectors_norm, vectors_norm.T)
    
    edges = []
    SIMILARITY_THRESHOLD = 0.70 # Lower threshold to show more connections
    
    for i in range(len(sim_matrix)):
        # Find top 3 most similar documents for each document (excluding itself)
        top_indices = np.argsort(sim_matrix[i])[::-1][1:4] 
        for j in top_indices:
            if sim_matrix[i, j] > SIMILARITY_THRESHOLD:
                # Add edge if not already added (undirected)
                if i < j:
                    edges.append([i, int(j)])

    # 3. Handle Categories and Coloring
    # Extract categories from metadata and assign colors
    all_categories = []
    for m in metadata:
        cat = m.get("categories", ["Uncategorized"])[0] # Use top-level category
        all_categories.append(cat)
    
    unique_cats = sorted(list(set(all_categories)))
    # Premium color palette
    palette = [
        '#38bdf8', # Sky
        '#fb7185', # Rose
        '#34d399', # Emerald
        '#fbbf24', # Amber
        '#a78bfa', # Violet
        '#f472b6', # Pink
        '#2dd4bf', # Teal
        '#60a5fa', # Blue
    ]
    cat_to_color = {cat: palette[i % len(palette)] for i, cat in enumerate(unique_cats)}

    # 4. Combine data for visualization
    nodes = []
    for i in range(len(vectors_3d)):
        cat = metadata[i].get("categories", ["Uncategorized"])[0]
        node = {
            "id": i,
            "x": float(vectors_3d[i, 0]),
            "y": float(vectors_3d[i, 1]),
            "z": float(vectors_3d[i, 2]),
            "size": 5 + min(15, len(documents[i]) / 100),
            "color": cat_to_color[cat],
            "category": cat,
            "text": documents[i],
            "metadata": metadata[i]
        }
        nodes.append(node)

    output_data = {
        "nodes": nodes,
        "edges": edges,
        "categories": cat_to_color
    }

    print(f"Saving visualization data to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Success! (Nodes: {len(nodes)}, Edges: {len(edges)})")

if __name__ == "__main__":
    extract_visualization_data()
