import argparse
import sys
import os
from core.vector_db import SimpleVectorDB


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
    from core.config import DB_DEFAULT_PATH
    parser.add_argument(
        "--db", type=str, default=DB_DEFAULT_PATH, help="Path to the saved vector DB file."
    )
    parser.add_argument(
        "--model", type=str, default=None, help="The embedding model to use."
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=0.0, help="Similarity threshold (0.0 to 1.0)."
    )
    parser.add_argument(
        "--no-rerank", action="store_true", help="Disable Cross-Encoder re-ranking."
    )
    parser.add_argument(
        "--rerank_k", type=int, default=10, help="Number of candidates for re-ranking."
    )

    args = parser.parse_args()
    rerank = not args.no_rerank

    if not os.path.exists(args.db):
        print(f"LOGE: [Search] Error: DB file not found. '{args.db}'")
        print("LOGE: [Search] Hint: Please run 'python src/indexer.py' first to index your documents.")
        sys.exit(1)

    # print("LOGE: [Search] Loading Vector DB... (This may take a few seconds.)")
    db = SimpleVectorDB(model_name=args.model)
    db.load(args.db)

    if args.query:
        search_query(db, args.query, args.top_k, args.threshold, rerank, args.rerank_k)
    else:
        print("\n=== Entering Interactive Search Mode (Type 'q' or 'quit' to exit) ===")
        print(f"💡 Tip: Re-ranking is {'ENABLED' if rerank else 'DISABLED'}. Enter a result number to open!")
        
        last_results = []
        while True:
            query = input("🔍 Enter your query or #number: ")
            
            if query.lower() in ["q", "quit", "exit"]:
                break
                
            if not query.strip():
                continue

            # Handle file opening request
            if query.startswith('#') and query[1:].isdigit():
                idx = int(query[1:]) - 1
                if 0 <= idx < len(last_results):
                    target_file = last_results[idx]["metadata"].get("source_path")
                    if target_file and os.path.exists(target_file):
                        print(f"Opening: {target_file}")
                        import subprocess
                        subprocess.run(["xdg-open", target_file])
                        continue
                    else:
                        print("Error: Could not find the source file.")
                        continue
                else:
                    print(f"Error: Invalid result number. (1 to {len(last_results)})")
                    continue

            # Execute search
            last_results = search_query(db, query, args.top_k, args.threshold, rerank, args.rerank_k)


import textwrap

def search_query(db, query, top_k, threshold=0.0, rerank=True, rerank_k=10):
    """Executes a search query and prints the formatted, clean results.

    Args:
        db (SimpleVectorDB): The loaded vector database instance.
        query (str): The search query text.
        top_k (int): The number of top results to retrieve.
        threshold (float): Similarity threshold filter.
        rerank (bool): Whether to use re-ranking.
        rerank_k (int): Number of candidates to rerank.
        
    Returns:
        list[dict]: The results found (to allow interactive opening).
    """
    # ANSI color codes for sophisticated look
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    mode_str = f" {GREEN}(Re-ranking ON){RESET}" if rerank else ""
    print(f"\n{CYAN}{BOLD}🔎 Search Results for: '{query}'{RESET}{mode_str} {DIM}(Min: {threshold}){RESET}")
    print(f"{DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")
    
    # Increase k to ensure we have enough unique documents after deduplication
    initial_k = max(top_k * 5, rerank_k) if rerank else top_k * 5
    results = db.search_hybrid(query, top_k=initial_k)
    
    # 1. Apply Threshold filtering BEFORE reranking (optional, but saves compute)
    if threshold > 0:
        results = [r for r in results if r["score"] >= threshold]

    if not results:
        print(f"{GREEN}LOGE: [Search] No matching results found.{RESET}")
        return []

    # 2. Apply Re-ranking
    if rerank and len(results) > 1:
        results = db.rerank(query, results)
        
    # 3. Deduplicate by document (rel_path)
    # We want to show the best chunk for each unique document.
    unique_results = []
    seen_paths = set()
    
    for res in results:
        path = res["metadata"].get("rel_path")
        if path not in seen_paths:
            unique_results.append(res)
            seen_paths.add(path)
            if len(unique_results) >= top_k:
                break
    
    results = unique_results

    try:
        terminal_width = min(os.get_terminal_size().columns, 100)
    except (AttributeError, OSError):
        terminal_width = 80

    for i, res in enumerate(results, 1):
        score = res.get("rerank_score", res["score"])
        score_label = "R-Score" if "rerank_score" in res else "Score"
        meta = res["metadata"]
        title = meta.get("title", meta.get("filename", "Unknown"))
        tags = meta.get("tags", [])
        categories = meta.get("categories", [])
        snippet = res["text"].strip()

        # Format header with Category and Title
        cat_str = f"[{' > '.join(categories)}] " if categories else ""
        header = f"{BOLD}{i}. {cat_str}{title}{RESET} {DIM}({score_label}: {score:.4f}){RESET}"
        
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

    print(f"{DIM}Total {len(results)} results displayed. (Type #1 to open first result){RESET}\n")
    return results


if __name__ == "__main__":
    main()
