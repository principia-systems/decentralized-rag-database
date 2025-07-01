#!/bin/bash
# lint.sh
# DESCRIPTION: Run Black, isort, Flake8, and MyPy on src/ and tests/ for full style + type checks
set -e

echo "Running black formatter..."
poetry run black src tests

echo "Running isort..."
poetry run isort src tests

echo "Running flake8..."
poetry run flake8 src tests

echo "Running mypy..."
poetry run mypy src 