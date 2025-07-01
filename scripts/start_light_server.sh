#!/bin/bash
# start_light_server.sh
# DESCRIPTION: Start Light FastAPI server (evaluation, status)
set -e

# Default configuration
PORT=5001
HOST="0.0.0.0"
ENV="development"
WORKERS=4
LOG_LEVEL="info"

# Help message
function show_help {
    echo "Usage: start_light_server.sh [OPTIONS]"
    echo "Start Light FastAPI server (quick operations)"
    echo ""
    echo "Options:"
    echo "  -p, --port PORT          Port to run the server on (default: 5001)"
    echo "  -h, --host HOST          Host to bind the server to (default: 0.0.0.0)"
    echo "  -e, --env ENV            Environment: development or production (default: development)"
    echo "  -w, --workers WORKERS    Number of worker processes (default: 4)"
    echo "  -l, --log-level LEVEL    Log level: debug, info, warning, error, critical (default: info)"
    echo "  --stop                   Stop any existing light server"
    echo "  --logs                   Show recent logs"
    echo "  --help                   Show this help message and exit"
    echo ""
    echo "Examples:"
    echo "  # Start with defaults"
    echo "  ./scripts/start_light_server.sh"
    echo ""
    echo "  # Production mode with 6 workers"
    echo "  ./scripts/start_light_server.sh --env production --workers 6"
    echo ""
    echo "  # Custom port"
    echo "  ./scripts/start_light_server.sh --port 5003"
    exit 0
}

# Function to stop existing server
function stop_server {
    echo "Stopping existing Light server..."
    
    PID=$(lsof -ti:$PORT 2>/dev/null || true)
    if [ ! -z "$PID" ]; then
        echo "Stopping Light server on port $PORT (PID: $PID)"
        kill -TERM $PID 2>/dev/null || true
        sleep 2
        # Force kill if still running
        if kill -0 $PID 2>/dev/null; then
            echo "Force killing Light server"
            kill -KILL $PID 2>/dev/null || true
        fi
    else
        echo "No Light server running on port $PORT"
    fi
    
    # Also kill any uvicorn processes related to light_app
    pkill -f "src.server.light_app" 2>/dev/null || true
    
    echo "Light server shutdown complete"
    exit 0
}

# Function to show logs
function show_logs {
    echo "Recent Light server logs:"
    if [ -f "logs/light_server.log" ]; then
        tail -n 50 logs/light_server.log
    else
        echo "No log file found at logs/light_server.log"
    fi
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
        --stop)
            stop_server
            ;;
        --logs)
            show_logs
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

# Check if port is available
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "Error: Port $PORT is already in use"
    echo "Use --stop to kill existing server or choose a different port"
    exit 1
fi

echo "Starting Light server..."
echo "Server: $HOST:$PORT"
echo "Environment: $ENV"
echo "Workers: $WORKERS"

# Create log directory
mkdir -p logs

LOG_FILE="logs/light_server.log"

if [ "$ENV" = "production" ]; then
    echo "Starting Light server in production mode with $WORKERS workers"
            poetry run uvicorn src.server.light_app:app \
            --host "$HOST" \
            --port "$PORT" \
            --workers "$WORKERS" \
            --log-level "$LOG_LEVEL" \
        --access-log \
        > "$LOG_FILE" 2>&1 &
else
    echo "Starting Light server in development mode (single worker with reload)"
            poetry run uvicorn src.server.light_app:app \
            --host "$HOST" \
            --port "$PORT" \
            --reload \
            --log-level "$LOG_LEVEL" \
        > "$LOG_FILE" 2>&1 &
fi

PID=$!
echo "Light server started with PID: $PID"
echo "Logs: $LOG_FILE"

# Wait a moment for server to start
sleep 3

# Check if server is running
if ! kill -0 $PID 2>/dev/null; then
    echo "❌ Error: Light server failed to start"
    echo "Check logs: tail -f $LOG_FILE"
    exit 1
fi

echo ""
echo "✅ Light server started successfully!"
echo ""
echo "🚀 Light Server Endpoints:"
echo "  - POST http://$HOST:$PORT/api/evaluate   # Query evaluation"
echo "  - GET  http://$HOST:$PORT/api/status     # User status"
echo "  - GET  http://$HOST:$PORT/health         # Health check"
echo "  - GET  http://$HOST:$PORT/docs           # API documentation"
echo ""
echo "📊 Server Info:"
echo "  - PID: $PID"
echo "  - Workers: $WORKERS (CPU cores available: $(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 'unknown'))"
echo "  - Log file: $LOG_FILE"
echo ""
echo "🔧 Management:"
echo "  - View logs: tail -f $LOG_FILE"
echo "  - Stop server: ./scripts/start_light_server.sh --stop"
echo "  - Test health: curl http://localhost:$PORT/health"
echo ""

# Create a function to handle cleanup on script exit
cleanup() {
    echo ""
    echo "Shutting down Light server..."
    kill $PID 2>/dev/null || true
    wait $PID 2>/dev/null || true
    echo "Light server stopped"
}

# Set trap for cleanup
trap cleanup EXIT

echo "Press Ctrl+C to stop the server"
echo ""

# Monitor server
while true; do
    if ! kill -0 $PID 2>/dev/null; then
        echo "⚠️  Light server (PID: $PID) has stopped unexpectedly!"
        echo "Check logs: tail -f $LOG_FILE"
        break
    fi
    
    sleep 5
done 