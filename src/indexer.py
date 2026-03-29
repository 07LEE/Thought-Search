import os
import glob
import re
from vector_db import SimpleVectorDB


def chunk_text(text, max_chunk_size=800):
    """Splits text semantically by paragraphs while preserving markdown heading context.

    Args:
        text (str): The raw input markdown text.
        max_chunk_size (int): The maximum number of characters per chunk.

    Returns:
        list[str]: A list of valid, context-aware text chunks.
    """
    paragraphs = text.split('\n\n')
    valid_chunks = []
    current_heading = ""
    current_chunk = ""

    for paragraph in paragraphs:
        p_stripped = paragraph.strip()
        if not p_stripped:
            continue
            
        # Update context if a heading is found to help the LLM/Vector model understand semantic belonging.
        if p_stripped.startswith('#'):
            current_heading = p_stripped.split('\n')[0]
            
        # If adding the next paragraph breaches the max limit, flush the current chunk to the results
        # and re-inject the active heading recursively into the new chunk.
        if len(current_chunk) + len(p_stripped) > max_chunk_size and current_chunk:
            valid_chunks.append(current_chunk.strip())
            current_chunk = f"[{current_heading}]\n\n" if current_heading else ""
            
        current_chunk += p_stripped + "\n\n"
        
    if current_chunk.strip():
        valid_chunks.append(current_chunk.strip())
        
    # Remove fragments that are too short to hold standalone semantic meaning.
    return [chunk for chunk in valid_chunks if len(chunk) > 20]


def index_markdown_files(posts_dir, db_path, model_name=None):
    """Indexes all markdown files structurally in the specified directory into the vector DB.

    Args:
        posts_dir (str): The directory containing markdown files.
        db_path (str): The path to the vector database JSON file.
        model_name (str, optional): The embedding model to use.
    """
    db = SimpleVectorDB(model_name=model_name)
    if os.path.exists(db_path):
        db.load(db_path)

    search_pattern = os.path.join(posts_dir, "*.md")
    md_files = glob.glob(search_pattern)

    total_added_chunks = 0
    all_chunks = []
    all_metadata = []

    print(f"LOGE: [Indexer] {len(md_files)} Markdown files found in {posts_dir}")
    print("LOGE: [Indexer] Start processing...\n")

    for md_file in md_files:
        filename = os.path.basename(md_file)
        
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Strict Frontmatter Validation
        frontmatter_match = re.match(r"^---\n(.*?)\n---\n(.*)", content, flags=re.DOTALL)
        
        if not frontmatter_match:
            print(f"LOGE: [Indexer] WARNING: Skipping '{filename}' (Missing YAML Frontmatter). Please run 'python scripts/migrate.py'.")
            continue
            
        yaml_block = frontmatter_match.group(1)
        raw_text = frontmatter_match.group(2)
        
        # Extract title from YAML
        title_match = re.search(r'^title:\s*"(.*?)"', yaml_block, flags=re.MULTILINE)
        title = title_match.group(1) if title_match else filename

        chunks = chunk_text(raw_text)

        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append({"filename": filename, "source_path": md_file, "title": title})

        print(f"LOGE: [Indexer] Processed {filename}: Found {len(chunks)} chunks.")
        total_added_chunks += len(chunks)

    if all_chunks:
        print(f"\nLOGE: [Indexer] Adding total {total_added_chunks} chunks to Vector DB...")
        db.add_texts(all_chunks, metadatas=all_metadata)
        db.save(db_path)
        print("LOGE: [Indexer] Success!")
    else:
        print("LOGE: [Indexer] No texts to add.")


if __name__ == "__main__":
    import argparse
    from config import DB_DEFAULT_PATH

    parser = argparse.ArgumentParser(description="Thought-Search Markdown Indexer")
    parser.add_argument("--model", type=str, default=None, help="The embedding model to use.")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    posts_directory = os.path.join(current_dir, "..", "posts")

    index_markdown_files(posts_directory, DB_DEFAULT_PATH, args.model)
