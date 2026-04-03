import os

# Default AI Model (can be overridden by environment variable)
EMBEDDING_MODEL = os.getenv("THOUGHT_SEARCH_MODEL", "jhgan/ko-sroberta-multitask")

# Database Pathing
DB_DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "thought-search-db.json"
)

# Posts Directory (can be overridden by environment variable)
POSTS_DIR = os.getenv(
    "THOUGHT_SEARCH_POSTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "posts")
)
