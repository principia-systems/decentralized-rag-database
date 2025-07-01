#!/bin/bash
# start_server.sh
# DESCRIPTION: Start FastAPI server with configurable options
set -e

# Default configuration
PORT=3001
HOST="0.0.0.0"
ENV="development"
WORKERS=1
LOG_LEVEL="info"

# Help message
function show_help {
    echo "Usage: start_server.sh [OPTIONS]"
    echo "Start FastAPI server"
    echo ""
    echo "Options:"
    echo "  -p, --port PORT        Port to run the server on (default: 5000)"
    echo "  -h, --host HOST        Host to bind the server to (default: 0.0.0.0)"
    echo "  -e, --env ENV          Environment: development or production (default: development)"
    echo "  -w, --workers WORKERS  Number of worker processes (default: 1, only used in production)"
    echo "  -l, --log-level LEVEL  Log level: debug, info, warning, error, critical (default: info)"
    echo "  --help                 Show this help message and exit"
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -h|--host)
            HOST="$2"
            shift 2
            ;;
        -e|--env)
            ENV="$2"
            shift 2
            ;;
        -w|--workers)
            WORKERS="$2"
            shift 2
            ;;
        -l|--log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    source .env
fi

# Start the server
echo "Starting FastAPI server..."
echo "Host: $HOST, Port: $PORT, Environment: $ENV"

if [ "$ENV" = "production" ]; then
    echo "Running in production mode with $WORKERS workers"
    poetry run uvicorn src.server.single_server:app --host "$HOST" --port "$PORT" --workers "$WORKERS" --log-level "$LOG_LEVEL"
else
    echo "Running in development mode with auto-reload enabled"
    poetry run uvicorn src.server.single_server:app --host "$HOST" --port "$PORT" --reload --log-level "$LOG_LEVEL"
fi 
