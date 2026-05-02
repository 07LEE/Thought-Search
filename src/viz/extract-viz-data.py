#!/usr/bin/env python3
import json
import os
import numpy as np
from sklearn.manifold import TSNE

# Path configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

    # --- [AGGR] Group Chunks into Documents ---
    print("Grouping chunks into documents...")
    file_groups = {}
    for i, meta in enumerate(metadata):
        path = meta.get("rel_path")
        if not path: continue
        if path not in file_groups:
            file_groups[path] = []
        file_groups[path].append(i)
    
    file_metadata = []
    file_vectors = []
    file_texts = []
    
    for path, indices in file_groups.items():
        # 1. Aggregate vectors (Mean pooling)
        avg_vector = np.mean(vectors[indices], axis=0)
        file_vectors.append(avg_vector)
        
        # 2. Use first chunk's metadata as representative
        file_metadata.append(metadata[indices[0]])
        
        # 3. Combine text for the whole file
        full_text = "\n\n".join([documents[i] for i in indices])
        file_texts.append(full_text)
        
    file_vectors = np.array(file_vectors)
    print(f"Aggregated {len(vectors)} chunks into {len(file_vectors)} documents.")

    if len(file_vectors) < 2:
        print("Not enough documents for visualization.")
        return

    # 1. Dimensionality Reduction using t-SNE (3D)
    norms = np.linalg.norm(file_vectors, axis=1, keepdims=True)
    file_vectors_norm = file_vectors / (norms + 1e-10)
    
    perplexity = min(30, len(file_vectors) - 1)
    print(f"Running t-SNE 3D (perplexity={perplexity})...")
    
    tsne = TSNE(
        n_components=3, 
        perplexity=perplexity, 
        random_state=42, 
        init='pca', 
        learning_rate='auto'
    )
    vectors_3d = tsne.fit_transform(file_vectors_norm)

    # 2. Calculate Connectivity (Document-level)
    sim_matrix = np.dot(file_vectors_norm, file_vectors_norm.T)
    
    edges = []
    SIMILARITY_THRESHOLD = 0.70 
    BROAD_TAGS = {"linux", "ubuntu"} 
    
    for i in range(len(sim_matrix)):
        top_indices = np.argsort(sim_matrix[i])[::-1][1:5] 
        source_tags = set(file_metadata[i].get("tags", []))
        
        for j in top_indices:
            score = sim_matrix[i, j]
            target_tags = set(file_metadata[j].get("tags", []))
            
            shared_tags = source_tags.intersection(target_tags)
            if shared_tags - BROAD_TAGS:
                score += 0.1
                
            if score > SIMILARITY_THRESHOLD:
                if i < j:
                    edges.append([i, int(j)])

    # 3. Categories and Coloring
    all_categories = []
    for m in file_metadata:
        cat = m.get("categories", ["Uncategorized"])[0]
        all_categories.append(cat)
    
    unique_cats = sorted(list(set(all_categories)))
    palette = ['#38bdf8', '#fb7185', '#34d399', '#fbbf24', '#a78bfa', '#f472b6', '#2dd4bf', '#60a5fa']
    cat_to_color = {cat: palette[i % len(palette)] for i, cat in enumerate(unique_cats)}

    # 4. Combine data for visualization
    nodes = []
    for i in range(len(vectors_3d)):
        cat = file_metadata[i].get("categories", ["Uncategorized"])[0]
        source_path = file_metadata[i].get("source_path")
        rel_path = file_metadata[i].get("rel_path")
        
        mtime = 0
        search_paths = []
        if source_path: search_paths.append(source_path)
        if rel_path:
            search_paths.append(os.path.join(BASE_DIR, "posts", rel_path))
            search_paths.append(os.path.join(BASE_DIR, "src", "..", "posts", rel_path))
            
        for path in search_paths:
            if os.path.exists(path):
                mtime = os.path.getmtime(path)
                break
        
        node = {
            "id": i,
            "x": float(vectors_3d[i, 0]),
            "y": float(vectors_3d[i, 1]),
            "z": float(vectors_3d[i, 2]),
            "size": 8 + min(12, len(file_texts[i]) / 1000),
            "color": cat_to_color[cat],
            "category": cat,
            "mtime": mtime,
            "text": file_texts[i],
            "metadata": file_metadata[i]
        }
        nodes.append(node)

    output_data = {
        "nodes": nodes,
        "edges": edges,
        "intra_file_edges": [], 
        "categories": cat_to_color
    }

    print(f"Saving visualization data to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Success! (Nodes: {len(nodes)}, Semantic Edges: {len(edges)})")

if __name__ == "__main__":
    extract_visualization_data()
