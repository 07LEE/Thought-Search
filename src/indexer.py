import os
import glob
from vector_db import SimpleVectorDB


def chunk_text(text):
    """Splits raw text into paragraph-level chunks based on double newlines.

    Args:
        text (str): The raw input text.

    Returns:
        list[str]: A list of valid, non-empty text chunks.
    """
    chunks = text.split("\n\n")
    valid_chunks = []

    for chunk in chunks:
        cleaned = chunk.strip()
        if len(cleaned) > 10:
            valid_chunks.append(cleaned)

    return valid_chunks


def index_markdown_files(posts_dir, db_path):
    """Indexes all markdown files in the specified directory into the vector DB.

    Args:
        posts_dir (str): The directory containing markdown files.
        db_path (str): The path to the vector database JSON file.
    """
    db = SimpleVectorDB()
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
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()

        filename = os.path.basename(md_file)
        chunks = chunk_text(content)

        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append({"filename": filename, "source_path": md_file})

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
    current_dir = os.path.dirname(os.path.abspath(__file__))
    posts_directory = os.path.join(current_dir, "..", "posts")
    database_file = os.path.join(current_dir, "..", "data", "thought-search-db.json")

    index_markdown_files(posts_directory, database_file)
