#!/bin/bash
# run_db_creator.sh
# DESCRIPTION: Build ChromaDB collections by pulling embeddings & metadata from Neo4j/IPFS
set -e

echo "Running src database creator..."
poetry run python -m src.db.db_creator_main -v