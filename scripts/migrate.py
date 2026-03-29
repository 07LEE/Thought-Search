import os
import glob
import re
from datetime import datetime


def extract_title(content, filename):
    """Extracts the first heading # as title, or defaults to filename."""
    match = re.search(r"^#\s+(.+)$", content, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()
    return filename.replace(".md", "").replace("-", " ")


def get_file_date(filepath, filename):
    """Extracts date from filename YYYY-MM-DD-slug.md or falls back to system creation time."""
    match = re.match(r"^(\d{4}-\d{2}-\d{2})", filename)
    if match:
        return match.group(1)
    
    # Fallback to system time
    timestamp = os.path.getctime(filepath)
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")


def migrate_posts():
    """Scans and injects YAML frontmatter to legacy markdown files in posts/ directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    posts_dir = os.path.join(current_dir, "..", "posts")
    
    if not os.path.exists(posts_dir):
        print(f"LOGE: [Migrator] Directory '{posts_dir}' does not exist.")
        return
        
    search_pattern = os.path.join(posts_dir, "*.md")
    md_files = glob.glob(search_pattern)
    
    migrated_count = 0
    skipped_count = 0
    
    for md_file in md_files:
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Check if already has frontmatter
        if content.startswith("---\n"):
            skipped_count += 1
            continue
            
        filename = os.path.basename(md_file)
        title = extract_title(content, filename)
        date_str = get_file_date(md_file, filename)
        
        frontmatter = f"""---
title: "{title}"
date: "{date_str}"
tags: []
---
"""
        new_content = frontmatter + "\n" + content
        
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(new_content)
            
        migrated_count += 1
        print(f"LOGE: [Migrator] Migrated: {filename}")
        
    print(f"LOGE: [Migrator] Migration complete. Migrated: {migrated_count}, Skipped (Already formatted): {skipped_count}")


if __name__ == "__main__":
    migrate_posts()
