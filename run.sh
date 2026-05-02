#!/bin/bash

# Thought-Search: General Execution Script
# This script automates environment setup, indexing, and searching.

# 1. Setup path
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

# ANSI color codes
BOLD="\033[1m"
CYAN="\033[36m"
GREEN="\033[32m"
YELLOW="\033[33m"
RESET="\033[0m"
export PYTHONPATH=$BASE_DIR/src

echo -e "${CYAN}${BOLD}🚀 Thought-Search System Initializing...${RESET}"

# 2. Environment Activation
if [ -n "$VIRTUAL_ENV" ]; then
    echo -e "${GREEN}✔ Active Virtual Environment detected: $(basename "$VIRTUAL_ENV")${RESET}"
elif [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo -e "${GREEN}✔ Active Conda Environment detected: $CONDA_DEFAULT_ENV${RESET}"
elif [ -d ".venv" ]; then
    source .venv/bin/activate
    echo -e "${GREEN}✔ Local .venv activated.${RESET}"
else
    echo -e "${YELLOW}⚠ Warning: No virtual environment detected. Proceeding with system python...${RESET}"
fi

# Check for Visualization Mode
if [ "$1" == "--viz" ]; then
    echo -e "${CYAN}${BOLD}🌐 Launching Visualization Server at http://localhost:8080...${RESET}"
    python3 src/core/server.py
    exit 0
fi

# 3. Synchronize Knowledge (Auto-Indexing)
echo -e "${CYAN}${BOLD}📂 Syncing Knowledge Base...${RESET}"
python3 src/cli/indexer.py
if [ $? -ne 0 ]; then
    echo -e "\033[31m✖ Error: Indexing failed. Please check your document structure.${RESET}"
    exit 1
fi

# 4. Launch Search Engine
if [ $# -eq 0 ]; then
    # No arguments: Interactive Mode
    python3 src/cli/search.py
else
    # Arguments provided: Direct Search
    python3 src/cli/search.py "$@"
fi
