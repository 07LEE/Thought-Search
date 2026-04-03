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


import textwrap

def search_query(db, query, top_k):
    """Executes a search query and prints the formatted, clean results.

    Args:
        db (SimpleVectorDB): The loaded vector database instance.
        query (str): The search query text.
        top_k (int): The number of top results to retrieve.
    """
    # ANSI color codes for sophisticated look
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    print(f"\n{CYAN}{BOLD}🔎 Search Results for: '{query}'{RESET}")
    print(f"{DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")
    
    results = db.search(query, top_k=top_k)

    if not results:
        print(f"{GREEN}LOGE: [Search] No matching results found.{RESET}")
        return

    try:
        terminal_width = min(os.get_terminal_size().columns, 100)
    except (AttributeError, OSError):
        terminal_width = 80

    for i, res in enumerate(results, 1):
        score = res["score"]
        meta = res["metadata"]
        title = meta.get("title", meta.get("filename", "Unknown"))
        tags = meta.get("tags", [])
        categories = meta.get("categories", [])
        snippet = res["text"].strip()

        # Format header with Category and Title
        cat_str = f"[{' > '.join(categories)}] " if categories else ""
        header = f"{BOLD}{i}. {cat_str}{title}{RESET} {DIM}(Score: {score:.4f}){RESET}"
        
        # Format tags
        tag_line = f"{CYAN}#{' #'.join(tags)}{RESET}" if tags else ""

        print(header)
        if tag_line:
            print(f"   {tag_line}")
        
        # Preserve original structure while wrapping long lines
        wrapped_lines = []
        for line in snippet.splitlines():
            if not line.strip():
                wrapped_lines.append("")
                continue
            
            # Wrap only long lines, preserving the indentation for wrapped parts
            sub_lines = textwrap.wrap(
                line, 
                width=terminal_width - 6, 
                initial_indent="   ", 
                subsequent_indent="   "
            )
            wrapped_lines.extend(sub_lines)
        
        print("\n".join(wrapped_lines))
        print(f"{DIM}──────────────────────────────────────────────────────────────{RESET}")

    print(f"{DIM}Total {len(results)} results displayed.{RESET}\n")


if __name__ == "__main__":
    main()
