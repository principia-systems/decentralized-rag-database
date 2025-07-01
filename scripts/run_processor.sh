#!/bin/bash
# run_processor.sh
# DESCRIPTION: End‑to‑end pipeline — PDF → Markdown → chunks → embeddings → IPFS/DB
set -e

echo "Running src processor..."
poetry run python -m src.core.processor_main -v 