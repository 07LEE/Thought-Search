import os

# Default AI Model (can be overridden by environment variable)
EMBEDDING_MODEL = os.getenv("THOUGHT_SEARCH_MODEL", "jhgan/ko-sroberta-multitask")

# Database Pathing
DB_DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "thought-search-db.json"
)

# Posts Directory (can be overridden by environment variable)
POSTS_DIR = os.getenv(
    "THOUGHT_SEARCH_POSTS",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "posts")
)

# Reranking Model
RERANK_MODEL = os.getenv("THOUGHT_SEARCH_RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

# Indexing Configuration
# Directories to skip entirely during indexing
EXCLUDED_DIRS = [".git", ".venv", "__pycache__"]

# Specific filenames to skip
EXCLUDED_FILENAMES = ["README.md", "TEMPLATE.md"]

# File extensions to index
SUPPORTED_EXTENSIONS = [".md"]
