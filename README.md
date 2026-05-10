# Thought-Search

Thought-Search is a CLI tool that builds a local vector database from Markdown files for semantic search. It now supports hierarchical directory structures and automatic category extraction.

---

## Features

### Text Processing & Embedding

- **Semantic Search:** Uses cosine similarity to match query vectors against indexed document chunks.
- **Hybrid Search:** Combines semantic (Vector) and keyword (BM25) search using Reciprocal Rank Fusion (RRF) for superior accuracy.
- **Korean Morphological Analysis:** Integrated with `kiwipiepy` (Kiwi) for precise Korean tokenization and particle removal.
- **Custom Dictionary:** Supports external dictionary management via `Thought-Dictionary` to protect technical terms (e.g., 3DGS, COLMAP).
- **Markdown Parsing:** Splits raw `.md` files into paragraph-level chunks for indexing.
- **Local Embedding:** Generates text embeddings locally using a configurable `sentence-transformers` model.
- **Hierarchical Support:** Recursively searches for markdown files in subdirectories.
- **Category Extraction:** Automatically extracts folder names from the directory structure and stores them as `categories` metadata.
- **3D Knowledge Graph:** Interactive 3D visualization of semantic relationships between documents using t-SNE dimensionality reduction and Plotly.js.

### Architecture & Storage

- **Offline Execution:** All inference and processing are done locally without external API dependencies.
- **Read-Only Access:** The application only reads markdown files and does not modify the original data.
- **Hybrid Storage:** Stores metadata in JSON and high-dimensional vectors in a binary NumPy (`.npy`) file for performance.
- **Flexible Data Sources:** Supports a default `posts/` directory (can be a Git submodule) or any external path via environment variables.

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

- **Vector DB:** `data/thought-search-db.json` & `data/thought-search-db.vectors.npy`
- Stores text chunks, metadata, and their corresponding dense vector arrays separately for efficiency.

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

# Install Custom Dictionary Manager (Optional but Recommended for Korean)
# Path should be where you cloned Thought-Dictionary
pip install -e /path/to/Thought-Dictionary
```

### 2. Configuration

You can override the default posts directory using the `THOUGHT_SEARCH_POSTS` environment variable:

```bash
export THOUGHT_SEARCH_POSTS="/path/to/your/knowledge-base"
```

### 3. Usage

The most convenient way to use Thought-Search is via the `run.sh` script, which automatically activates the environment, indexes your documents, and starts the search engine.

```bash
# Start interactive search (with auto-indexing)
./run.sh

# Search directly with a query
./run.sh "How to install Kubernetes?"
```

#### Manual Workflow

```bash
# 1. Index your markdown files
python src/cli/indexer.py

# 2. Search for a specific query
python src/cli/search.py "Your query here"
```

### 3D Visualization

Visualize your knowledge base in an interactive 3D space:

```bash
# 1. Extract visualization data (t-SNE 3D reduction)
python src/viz/extract_viz_data.py

# 2. Start the integrated search & visualization server
./run.sh --viz

# 3. Open in your browser
# URL: http://localhost:8080
```

**Features:**

- **Domain Coloring:** Nodes colored by top-level categories.
- **Dynamic Sizing:** Larger nodes indicate richer content density.
- **Real-time Search:** Instantly highlight nodes by title, tags, or content.
- **Interactive Panel:** Click nodes to read rendered Markdown previews.

---

## Directory Structure

- `data/`: Output directory for the database files.
- `posts/`: Default directory for Input Markdown files.
- `src/`: Application source code.
  - `core/`: Core logic including search engines and database management.
  - `cli/`: Command-line interfaces for indexing and searching.
  - `viz/`: Visualization data processing logic.
