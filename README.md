# Thought-Search

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)

Thought-Search is a CLI tool that builds a local vector database from Markdown files for semantic search.

---

## Features

### Text Processing & Embedding

- **Semantic Search:** Uses cosine similarity to match query vectors against indexed document chunks.
- **Markdown Parsing:** Splits raw `.md` files into paragraph-level chunks for indexing.
- **Local Embedding:** Generates text embeddings locally using a configurable `sentence-transformers` model.

### Architecture & Storage

- **Offline Execution:** All inference and processing are done locally without external API dependencies.
- **Read-Only Access:** The application only reads markdown files and does not modify the original data.
- **JSON Backend:** Stores vectors and metadata in a JSON file (`data/thought-search-db.json`).
- **Git Submodule Support:** Supports markdown directories linked as Git submodules (`[REDACTED]/`), keeping user data separate from application logic.

---

## Data Requirements

### Input Format

- **Format:** Markdown (`*.md`)
- **Source Directory:** All target files must be located in the `[REDACTED]/` directory.
- **Mandatory MetaData:** Every file must start with a YAML Frontmatter block to be indexed successfully:

```yaml
---
title: "Your Document Title"
date: "YYYY-MM-DD"
tags: ["tag1", "tag2"]
---
```

### Storage Output

- **Vector DB:** `data/thought-search-db.json`
- Stores text chunks, source filenames, and their corresponding dense vector arrays.

---

## Tech Stack

- **Language:** Python 3.12+
- **Embedding:** `sentence-transformers`
- **Vector Operations:** `numpy`

---

## Setup & Execution

### 1. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Usage

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
- `[REDACTED]/`: Input Markdown files (Git Submodule).
- `scripts/`: Utility scripts (`run_demo.sh`).
- `src/`: Application source code (`indexer.py`, `search.py`, `vector_db.py`).
