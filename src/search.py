import argparse
import sys
import os
from vector_db import SimpleVectorDB


def main():
    """Main entry point for the CLI search application."""
    parser = argparse.ArgumentParser(description="Thought-Search Vector DB Search Engine")
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        help="The search query (e.g., 'How to install Kubernetes?')",
    )
    parser.add_argument(
        "--top_k", "-k", type=int, default=3, help="Number of results to retrieve."
    )
    from config import DB_DEFAULT_PATH
    parser.add_argument(
        "--db", type=str, default=DB_DEFAULT_PATH, help="Path to the saved vector DB file."
    )
    parser.add_argument(
        "--model", type=str, default=None, help="The embedding model to use."
    )

    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"LOGE: [Search] Error: DB file not found. '{args.db}'")
        print("LOGE: [Search] Hint: Please run 'python src/indexer.py' first to index your documents.")
        sys.exit(1)

    print("LOGE: [Search] Loading Vector DB... (This may take a few seconds.)")
    db = SimpleVectorDB(model_name=args.model)
    db.load(args.db)

    if args.query:
        search_query(db, args.query, args.top_k)
    else:
        print("\n=== Entering Interactive Search Mode (Type 'q' or 'quit' to exit) ===")
        while True:
            query = input("🔍 Enter your query: ")
            if query.lower() in ["q", "quit", "exit"]:
                break
            if not query.strip():
                continue

            search_query(db, query, args.top_k)


def search_query(db, query, top_k):
    """Executes a search query and prints the formatted results.

    Args:
        db (SimpleVectorDB): The loaded vector database instance.
        query (str): The search query text.
        top_k (int): The number of top results to retrieve.
    """
    print(f"\n🔎 [{query}] Search Results (Top {top_k})...\n")
    results = db.search(query, top_k=top_k)

    if not results:
        print("LOGE: [Search] No matching results found.")
        return

    for i, res in enumerate(results, 1):
        score = res["score"]
        meta = res["metadata"]
        title = meta.get("title", meta.get("filename", "Unknown"))
        tags = meta.get("tags", [])
        snippet = res["text"]

        if len(snippet) > 200:
            snippet = snippet[:200] + "..."

        tag_str = f"  |  tags: {', '.join(tags)}" if tags else ""
        print(f"[{i}] {title}{tag_str}  (Score: {score:.4f})")
        print(f"    📝 {snippet}\n")


if __name__ == "__main__":
    main()
