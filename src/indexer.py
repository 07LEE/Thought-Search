import os
import glob
import re
import hashlib
from vector_db import SimpleVectorDB


def compute_file_hash(filepath):
    """Computes the SHA-256 hash of a file's content for change detection.

    Args:
        filepath: The absolute path to the file.

    Returns:
        A hex digest string of the file's SHA-256 hash.
    """
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


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


def parse_markdown(filepath):
    """Parses a markdown file and extracts frontmatter metadata and text chunks.

    Args:
        filepath: The absolute path to the markdown file.

    Returns:
        A tuple of (chunks, metadata_template) on success, or (None, None)
        if the file lacks valid YAML frontmatter.
    """
    filename = os.path.basename(filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    frontmatter_match = re.match(r"^---\n(.*?)\n---\n(.*)", content, flags=re.DOTALL)
    if not frontmatter_match:
        return None, None

    yaml_block = frontmatter_match.group(1)
    raw_text = frontmatter_match.group(2)

    # Extract title and tags from YAML
    title_match = re.search(r'^title:\s*"(.*?)"', yaml_block, flags=re.MULTILINE)
    title = title_match.group(1) if title_match else filename

    tags_match = re.search(r'^tags:\s*\[(.*)\]', yaml_block, flags=re.MULTILINE)
    tags = [t.strip().strip('"').strip("'") for t in tags_match.group(1).split(",") if t.strip()] if tags_match else []

    chunks = chunk_text(raw_text)
    meta = {"filename": filename, "source_path": filepath, "title": title, "tags": tags}

    return chunks, meta


def index_markdown_files(posts_dir, db_path, model_name=None):
    """Incrementally indexes markdown files by detecting changes via content hashing.

    Only new or modified files are embedded. Deleted files are purged from the DB.

    Args:
        posts_dir: The directory containing markdown files.
        db_path: The path to the vector database JSON file.
        model_name: The embedding model to use (optional).
    """
    db = SimpleVectorDB(model_name=model_name)
    if os.path.exists(db_path):
        db.load(db_path)

    md_files = glob.glob(os.path.join(posts_dir, "*.md"))
    current_files = {}

    for md_file in md_files:
        filename = os.path.basename(md_file)
        current_files[filename] = md_file

    # --- Detect Changes ---
    stored_hashes = db.file_hashes
    current_filenames = set(current_files.keys())
    stored_filenames = set(stored_hashes.keys())

    deleted = stored_filenames - current_filenames
    new_files = current_filenames - stored_filenames
    possibly_changed = current_filenames & stored_filenames

    changed = set()
    for filename in possibly_changed:
        current_hash = compute_file_hash(current_files[filename])
        if current_hash != stored_hashes.get(filename):
            changed.add(filename)

    unchanged_count = len(possibly_changed) - len(changed)
    files_to_index = new_files | changed

    print(f"LOGE: [Indexer] {len(current_files)} files found in {posts_dir}")
    print(f"LOGE: [Indexer] New: {len(new_files)}, Changed: {len(changed)}, "
          f"Deleted: {len(deleted)}, Unchanged: {unchanged_count}\n")

    # --- Remove Stale Entries ---
    for filename in deleted | changed:
        removed = db.remove_by_filename(filename)
        if filename in deleted:
            del db.file_hashes[filename]
            print(f"LOGE: [Indexer] Removed: {filename} ({removed} chunks)")

    # --- Index New & Changed Files ---
    all_chunks = []
    all_metadata = []

    for filename in sorted(files_to_index):
        filepath = current_files[filename]
        chunks, meta = parse_markdown(filepath)

        if chunks is None:
            print(f"LOGE: [Indexer] WARNING: Skipping '{filename}' (Missing YAML Frontmatter).")
            continue

        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append(meta)

        db.file_hashes[filename] = compute_file_hash(filepath)
        print(f"LOGE: [Indexer] Indexed: {filename} ({len(chunks)} chunks)")

    if all_chunks:
        db.add_texts(all_chunks, metadatas=all_metadata)

    if all_chunks or deleted:
        db.save(db_path)
        print(f"\nLOGE: [Indexer] Done! Total documents in DB: {len(db.documents)}")
    else:
        print("LOGE: [Indexer] Everything up to date. No changes needed.")


if __name__ == "__main__":
    import argparse
    from config import DB_DEFAULT_PATH

    parser = argparse.ArgumentParser(description="Thought-Search Markdown Indexer")
    parser.add_argument("--model", type=str, default=None, help="The embedding model to use.")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    posts_directory = os.path.join(current_dir, "..", "posts")

    index_markdown_files(posts_directory, DB_DEFAULT_PATH, args.model)
