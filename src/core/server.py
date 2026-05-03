import os
import sys
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS

# Ensure the root directory is in sys.path for relative imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src"))

from cli.indexer import index_markdown_files
from viz.extract_viz_data import extract_visualization_data
from core.config import DB_DEFAULT_PATH, POSTS_DIR
from core.vector_db import SimpleVectorDB

app = Flask(__name__)
CORS(app)

# Initialize and load Vector DB
db = SimpleVectorDB()
if os.path.exists(DB_DEFAULT_PATH):
    db.load(DB_DEFAULT_PATH)

# Pre-load models to avoid delay on first search
db.pre_load_models()

@app.route('/')
def root():
    """Serve the visualization index."""
    return send_from_directory(os.path.join(BASE_DIR, 'visualize'), 'index.html')

@app.route('/data/<path:path>')
def serve_data(path):
    """Serve files from the data directory."""
    return send_from_directory(os.path.join(BASE_DIR, 'data'), path)

@app.route('/posts/<path:path>')
def serve_posts(path):
    """Serve files from the posts directory."""
    return send_from_directory(POSTS_DIR, path)

@app.route('/<path:path>')
def serve_visualize(path):
    """Serve static files from the visualize directory."""
    return send_from_directory(os.path.join(BASE_DIR, 'visualize'), path)

@app.route('/api/sync', methods=['POST'])
def sync_db():
    """API endpoint to trigger manual re-indexing and data extraction."""
    try:
        print("\nLOGE: [Server] Received sync request.")
        # 1. Run Indexer
        print("LOGE: [Server] Running indexer...")
        index_markdown_files(POSTS_DIR, DB_DEFAULT_PATH)
        
        # 2. Run Visualization Data Extractor
        print("LOGE: [Server] Running viz data extractor...")
        extract_visualization_data()
        
        return jsonify({
            "status": "success", 
            "message": "Database and visualization data updated successfully."
        })
    except Exception as e:
        print(f"LOGE: [Server] Sync error: {e}")
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 500

@app.route('/api/search', methods=['GET'])
def search():
    """API endpoint for vector search."""
    query = request.args.get('q', '')
    top_k = int(request.args.get('k', 5))
    rerank = request.args.get('rerank', 'true').lower() == 'true'
    
    if not query:
        return jsonify({"results": []})
        
    try:
        # Increase initial k for hybrid search before reranking
        initial_k = max(top_k * 5, 10)
        results = db.search_hybrid(query, top_k=initial_k)
        
        if rerank and len(results) > 1:
            results = db.rerank(query, results)
            
        # Deduplicate and limit to top_k
        unique_results = []
        seen_paths = set()
        for res in results:
            path = res["metadata"].get("rel_path")
            if path not in seen_paths:
                unique_results.append(res)
                seen_paths.add(path)
                if len(unique_results) >= top_k:
                    break
        
        return jsonify({
            "status": "success",
            "results": unique_results
        })
    except Exception as e:
        print(f"LOGE: [Server] Search error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    print(f"Thought-Search Server starting at http://localhost:8080")
    app.run(host='0.0.0.0', port=8080)
