import os
import glob
import re
import hashlib
import concurrent.futures
from functools import partial
import yaml
from core.vector_db import SimpleVectorDB


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


def parse_markdown(filepath, rel_path=""):
    """Parses a markdown file and extracts frontmatter metadata, text chunks, and categories.

    Args:
        filepath: The absolute path to the markdown file.
        rel_path: The relative path from the base posts directory.

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

    try:
        # Use safe_load to avoid arbitrary code execution from malicious YAML
        frontmatter = yaml.safe_load(yaml_block)
    except yaml.YAMLError:
        print(f"LOGE: [Indexer] WARNING: Failed to parse YAML in '{filepath}'.")
        return None, None

    if not isinstance(frontmatter, dict):
        return None, None

    # Extract title and tags from PyYAML's dict
    title = frontmatter.get("title", filename)
    tags = frontmatter.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]

    # Extract categories from the directory structure
    categories = [p for p in os.path.dirname(rel_path).split(os.sep) if p]

    chunks = chunk_text(raw_text)
    meta = {
        "filename": filename, 
        "rel_path": rel_path,
        "source_path": filepath, 
        "title": title, 
        "tags": tags,
        "categories": categories
    }

    return chunks, meta


def index_markdown_files(posts_dir, db_path, model_name=None):
    """Incrementally indexes markdown files by detecting changes via content hashing.

    Only new or modified files are embedded. Deleted files are purged from the DB.

    Args:
        posts_dir: The directory containing markdown files.
        db_path: The path to the vector database JSON file.
        model_name: The embedding model to use (optional).
    """
    from core.config import EXCLUDED_DIRS, EXCLUDED_FILENAMES, SUPPORTED_EXTENSIONS
    db = SimpleVectorDB(model_name=model_name)
    if os.path.exists(db_path):
        db.load(db_path)

    # Look for markdown files recursively in subdirectories
    md_files = glob.glob(os.path.join(posts_dir, "**", "*"), recursive=True)
    
    valid_files = []
    for f in md_files:
        if os.path.isdir(f):
            continue
            
        basename = os.path.basename(f)
        rel_path = os.path.relpath(f, posts_dir)
        path_parts = rel_path.split(os.sep)
        
        # 1. Check if filename is in excluded list
        if basename in EXCLUDED_FILENAMES:
            continue
            
        # 2. Check if file extension is supported
        if not any(f.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
            continue
            
        # 3. Check if any part of the path is in excluded directories
        # This handles nested exclusions like 'private/secret.md' or 'Engineering/.git/config'
        if any(part in EXCLUDED_DIRS for part in path_parts):
            continue

        # 4. Global hidden folder/file exclusion (standard practice)
        if any(part.startswith('.') for part in path_parts if part != '.'):
            continue

        valid_files.append(f)

    current_files = {}
    for f in valid_files:
        rel_path = os.path.relpath(f, posts_dir)
        current_files[rel_path] = f

    # --- Detect Changes ---
    stored_hashes = db.file_hashes
    current_filenames = set(current_files.keys())
    stored_filenames = set(stored_hashes.keys())

    deleted = stored_filenames - current_filenames
    new_files = current_filenames - stored_filenames
    possibly_changed = current_filenames & stored_filenames

    changed = set()
    for rel_path in possibly_changed:
        current_hash = compute_file_hash(current_files[rel_path])
        if current_hash != stored_hashes.get(rel_path):
            changed.add(rel_path)

    unchanged_count = len(possibly_changed) - len(changed)
    files_to_index = new_files | changed

    print(f"LOGE: [Indexer] {len(current_files)} files found in {posts_dir}")
    print(f"LOGE: [Indexer] New: {len(new_files)}, Changed: {len(changed)}, "
          f"Deleted: {len(deleted)}, Unchanged: {unchanged_count}\n")

    # --- Remove Stale Entries ---
    for identifier in deleted | changed:
        # SimpleVectorDB matches by 'filename' metadata, so we pass rel_path here if we use it as ID.
        # But we've updated metadata to store rel_path as well.
        removed = db.remove_by_filename(identifier)
        if identifier in deleted:
            del db.file_hashes[identifier]
            print(f"LOGE: [Indexer] Removed: {identifier} ({removed} chunks)")

    # --- Index New & Changed Files (Parallel Version) ---
    all_chunks = []
    all_metadata = []

    # Prepare arguments for parallel execution
    targets = sorted(files_to_index)
    
    if targets:
        print(f"LOGE: [Indexer] Indexing {len(targets)} files in parallel...")
        
        # We use a wrapper to pass additional fixed arguments to parse_markdown
        # parse_markdown(filepath, rel_path="")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Map returns results in the order of the inputs
            future_to_file = {
                executor.submit(parse_markdown, current_files[rel_path], rel_path): rel_path 
                for rel_path in targets
            }
            
            for future in concurrent.futures.as_completed(future_to_file):
                rel_path = future_to_file[future]
                filepath = current_files[rel_path]
                try:
                    chunks, meta = future.result()
                    if chunks:
                        all_chunks.extend(chunks)
                        all_metadata.extend([meta] * len(chunks)) # Duplicate meta for each chunk
                        db.file_hashes[rel_path] = compute_file_hash(filepath)
                        print(f"LOGE: [Indexer] Indexed: {rel_path} ({len(chunks)} chunks)")
                    else:
                        print(f"LOGE: [Indexer] WARNING: Skipping '{rel_path}' (Invalid content).")
                except Exception as exc:
                    print(f"LOGE: [Indexer] ERROR: '{rel_path}' generated an exception: {exc}")

    if all_chunks:
        db.add_texts(all_chunks, metadatas=all_metadata)

    if all_chunks or deleted:
        db.save(db_path)
        print(f"\nLOGE: [Indexer] Done! Total documents in DB: {len(db.documents)}")
    else:
        print("LOGE: [Indexer] Everything up to date. No changes needed.")


if __name__ == "__main__":
    import argparse
    from core.config import DB_DEFAULT_PATH, POSTS_DIR, EXCLUDED_DIRS, EXCLUDED_FILENAMES, SUPPORTED_EXTENSIONS

    parser = argparse.ArgumentParser(description="Thought-Search Markdown Indexer")
    parser.add_argument("--model", type=str, default=None, help="The embedding model to use.")
    parser.add_argument("--posts", "-p", type=str, default=POSTS_DIR, help="The directory containing markdown files.")
    args = parser.parse_args()

    index_markdown_files(args.posts, DB_DEFAULT_PATH, args.model)
