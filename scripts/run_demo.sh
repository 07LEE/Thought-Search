#!/bin/bash
cd "$(dirname "$0")/.."

source .venv/bin/activate

echo "LOGE: [Demo] ======================================"
echo "LOGE: [Demo]  1. Markdown Document Indexing        "
echo "LOGE: [Demo] ======================================"
python src/indexer.py

echo ""
echo "LOGE: [Demo] ======================================"
echo "LOGE: [Demo]  2. Search Test                       "
echo "LOGE: [Demo] ======================================"
echo "LOGE: [Demo] Query: 'How to install a deb package on Linux?'"
python src/search.py "How to install a deb package on Linux?"

echo ""
echo "LOGE: [Demo] Ready for interactive search."
echo "LOGE: [Demo] Command: python src/search.py"
