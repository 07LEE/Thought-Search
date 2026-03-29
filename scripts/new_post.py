import argparse
import os
import re
from datetime import datetime


def slugify(text):
    """Converts a string to a URL-friendly slug."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text.strip("-")


def create_post(title):
    """Generates a new markdown post with YAML frontmatter."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    slug = slugify(title)
    filename = f"{today_str}-{slug}.md"
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    posts_dir = os.path.join(current_dir, "..", "posts")
    os.makedirs(posts_dir, exist_ok=True)
    
    filepath = os.path.join(posts_dir, filename)
    
    if os.path.exists(filepath):
        print(f"LOGE: [Generator] Error: File '{filename}' already exists.")
        return
        
    template = f"""---
title: "{title}"
date: "{today_str}"
tags: []
---

# {title}

"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(template)
        
    print(f"LOGE: [Generator] Successfully created new post: {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a new TIL markdown post.")
    parser.add_argument("title", type=str, help="The title of the post")
    args = parser.parse_args()
    
    create_post(args.title)
