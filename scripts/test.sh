#!/bin/bash
# test.sh
# DESCRIPTION: Execute pytest suite with coverage output

set -e

# Check for specific test flags
if [ "$1" = "--unit" ]; then
    echo "Running unit tests with coverage..."
    poetry run pytest tests/unit -v --cov=src --cov-report=term --cov-report=html
elif [ "$1" = "--integration" ]; then
    echo "Running integration tests with coverage..."
    poetry run pytest tests/integration -v --cov=src --cov-report=term --cov-report=html
# Check if a specific file is provided
elif [ -n "$1" ]; then
    echo "Running tests in file $1..."
    poetry run pytest "$1" -v
else
    echo "Running all tests with coverage..."
    poetry run pytest
fi 