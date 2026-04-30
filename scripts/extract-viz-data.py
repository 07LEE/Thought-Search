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
    SIMILARITY_THRESHOLD = 0.72 # Stricter threshold for high-quality connections
    BROAD_TAGS = {"linux", "ubuntu"} # Tags that are too common to provide meaningful specific links
    
    for i in range(len(sim_matrix)):
        # Find top 5 most similar documents (focused search)
        top_indices = np.argsort(sim_matrix[i])[::-1][1:6] 
        source_rel_path = metadata[i].get("rel_path")
        source_tags = set(metadata[i].get("tags", []))
        
        for j in top_indices:
            target_rel_path = metadata[j].get("rel_path")
            target_tags = set(metadata[j].get("tags", []))
            
            # Skip connections between chunks of the SAME file
            if source_rel_path == target_rel_path:
                continue
            
            # Semantic Similarity with Specific Tag Boosting
            score = sim_matrix[i, j]
            
            # Boost if they share SPECIFIC (not broad) tags
            shared_tags = source_tags.intersection(target_tags)
            specific_shared_tags = shared_tags - BROAD_TAGS
            
            if specific_shared_tags:
                score += 0.12 # Stronger boost for specific thematic alignment
                
            if score > SIMILARITY_THRESHOLD:
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
        # Get modification time of the source file
        source_path = metadata[i].get("source_path")
        rel_path = metadata[i].get("rel_path")
        
        # Robust path resolution
        mtime = 0
        search_paths = []
        if source_path: search_paths.append(source_path)
        if rel_path:
            # Try to resolve relative to current project root
            search_paths.append(os.path.join(BASE_DIR, "posts", rel_path))
            # Also try the resolved path from indexer's default
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
            "size": 5 + min(15, len(documents[i]) / 100),
            "color": cat_to_color[cat],
            "category": cat,
            "mtime": mtime,
            "text": documents[i],
            "metadata": metadata[i]
        }
        nodes.append(node)

    # 3. Calculate Structural Edges (Intra-file sequence)
    # This connects chunks of the same file in order to show document structure
    intra_file_edges = []
    file_groups = {}
    for i, meta in enumerate(metadata):
        path = meta.get("rel_path")
        if path:
            if path not in file_groups:
                file_groups[path] = []
            file_groups[path].append(i)
    
    for path, indices in file_groups.items():
        # Sort by chunk index or sequence if available, 
        # but since they are added sequentially in indexer, original index works.
        for k in range(len(indices) - 1):
            intra_file_edges.append([indices[k], indices[k+1]])

    output_data = {
        "nodes": nodes,
        "edges": edges,
        "intra_file_edges": intra_file_edges,
        "categories": cat_to_color
    }

    print(f"Saving visualization data to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Success! (Nodes: {len(nodes)}, Edges: {len(edges)}, Intra-file Edges: {len(intra_file_edges)})")

if __name__ == "__main__":
    extract_visualization_data()
