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
        # 1. Aggregate vectors using Weighted Mean Pooling
        # Weight chunks by their length (log-scale to prevent long docs from dominating too much)
        chunk_weights = []
        valid_indices = []
        
        for idx in indices:
            text_len = len(documents[idx])
            if text_len < 10: continue # Skip extremely short noise
            
            weight = np.log1p(text_len) # log(1 + length)
            chunk_weights.append(weight)
            valid_indices.append(idx)
        
        if not valid_indices: # Fallback to original if all filtered
            valid_indices = indices
            chunk_weights = [1.0] * len(indices)
            
        weights = np.array(chunk_weights).reshape(-1, 1)
        weighted_vector = np.sum(vectors[valid_indices] * weights, axis=0) / np.sum(weights)
        file_vectors.append(weighted_vector)
        
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

    # 2. Calculate Connectivity (Document-level) with Reciprocal Verification
    sim_matrix = np.dot(file_vectors_norm, file_vectors_norm.T)
    BROAD_TAGS = {"linux", "ubuntu", "optimization", "guide", "setup"} 
    
    # Pre-calculate RAW candidates for all nodes (Semantic-only)
    adj_list = []
    for i in range(len(sim_matrix)):
        top_indices = np.argsort(sim_matrix[i])[::-1][1:16] # Expanded to top 15
        candidates = {int(j): float(sim_matrix[i, j]) for j in top_indices}
        adj_list.append(candidates)

    edges = []
    SIMILARITY_THRESHOLD = 0.85 # One-way connection allowed if extremely strong
    RECIPROCAL_THRESHOLD = 0.80 # Mutual connection allowed at a more reasonable level
    
    for i in range(len(adj_list)):
        for j, raw_score in adj_list[i].items():
            if i >= j: continue
            
            is_reciprocal = i in adj_list[j]
            source_meta = file_metadata[i]
            target_meta = file_metadata[j]
            source_cats = source_meta.get("categories", [])
            target_cats = target_meta.get("categories", [])
            source_tags = set(source_meta.get("tags", []))
            target_tags = set(target_meta.get("tags", []))
            
            # Subtle metadata nudge
            final_score = raw_score
            if source_cats and target_cats:
                if source_cats[0] != target_cats[0]:
                    final_score -= 0.02 # Subtle penalty for domain mismatch
                else:
                    shared_depth = 0
                    for c1, c2 in zip(source_cats, target_cats):
                        if c1 == c2: shared_depth += 1
                        else: break
                    final_score += 0.02 * shared_depth # Increased bonus for same domain/folder
            
            # Tag Bonus
            shared_tags = source_tags.intersection(target_tags) - BROAD_TAGS
            if shared_tags:
                final_score += min(0.05, 0.02 * len(shared_tags))
            
            # Connection Logic:
            # Save edges with score for frontend filtering
            if final_score > 0.50: # Lower threshold for data inclusion
                edges.append([i, j, round(float(final_score), 4)])

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
