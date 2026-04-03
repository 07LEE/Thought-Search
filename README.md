# Thought-Search

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)

Thought-Search is a CLI tool that builds a local vector database from Markdown files for semantic search. It now supports hierarchical directory structures and automatic category extraction.

---

## Features

### Text Processing & Embedding

- **Semantic Search:** Uses cosine similarity to match query vectors against indexed document chunks.
- **Markdown Parsing:** Splits raw `.md` files into paragraph-level chunks for indexing.
- **Local Embedding:** Generates text embeddings locally using a configurable `sentence-transformers` model.
- **Hierarchical Support:** Recursively searches for markdown files in subdirectories.
- **Category Extraction:** Automatically extracts folder names from the directory structure and stores them as `categories` metadata.

### Architecture & Storage

- **Offline Execution:** All inference and processing are done locally without external API dependencies.
- **Read-Only Access:** The application only reads markdown files and does not modify the original data.
- **JSON Backend:** Stores vectors and metadata in a JSON file (`data/thought-search-db.json`).
- **Flexible Data Sources:** Supports a default `[REDACTED]/` directory (can be a Git submodule) or any external path via environment variables.

---

## Data Requirements

### Input Format

- **Format:** Markdown (`*.md`)
- **Source Directory:** Target files can be in the `[REDACTED]/` directory or any subfolder within it.
- **Mandatory Metadata:** Every file must start with a YAML Frontmatter block to be indexed successfully:

```yaml
---
title: "Your Document Title"
date: "YYYY-MM-DD"
tags: ["tag1", "tag2"]
---
```

### Storage Output

- **Vector DB:** `data/thought-search-db.json`
- Stores text chunks, source filenames, relative paths, categories, and their corresponding dense vector arrays.

---

## Tech Stack

- **Language:** Python 3.10+
- **Environment:** Conda
- **Embedding:** `sentence-transformers`
- **Vector Operations:** `numpy`

---

## Setup & Execution

### 1. Environment Setup (Conda)

```bash
# Create and activate environment
conda create -n thought-search python=3.10 -y
conda activate thought-search

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

You can override the default posts directory using the `THOUGHT_SEARCH_POSTS` environment variable:

```bash
export THOUGHT_SEARCH_POSTS="/path/to/your/knowledge-base"
```

### 3. Usage

**Quick Demo**
Run the automated script to index files and test the search functionality:

```bash
bash scripts/run_demo.sh
```

**Manual Workflow**

```bash
# 1. Index your markdown files
python src/indexer.py

# 2. Search for a specific query
python src/search.py "Your query here"

# 3. Start an interactive search session
python src/search.py
```

---

## Directory Structure

- `data/`: Output directory for the database JSON.
- `[REDACTED]/`: Default directory for Input Markdown files.
- `scripts/`: Utility scripts.
- `src/`: Application source code.
  - `config.py`: Configuration and environment variable handling.
  - `indexer.py`: Main indexing logic (recursive search).
  - `search.py`: CLI search interface.
  - `vector_db.py`: Simple JSON-based vector database implementation.
