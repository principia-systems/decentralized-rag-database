#!/bin/bash
# start_database_server.sh
# DESCRIPTION: Start Database FastAPI server (database operations)
set -e

# Default configuration
PORT=5003
HOST="0.0.0.0"
ENV="development"
WORKERS=4
LOG_LEVEL="info"

# Help message
function show_help {
    echo "Usage: start_database_server.sh [OPTIONS]"
    echo "Start Database FastAPI server (database creation and management)"
    echo ""
    echo "Options:"
    echo "  -p, --port PORT          Port to run the server on (default: 5003)"
    echo "  -h, --host HOST          Host to bind the server to (default: 0.0.0.0)"
    echo "  -e, --env ENV            Environment: development or production (default: development)"
    echo "  -w, --workers WORKERS    Number of worker processes (default: 4)"
    echo "  -l, --log-level LEVEL    Log level: debug, info, warning, error, critical (default: info)"
    echo "  --stop                   Stop any existing database server"
    echo "  --logs                   Show recent logs"
    echo "  --help                   Show this help message and exit"
    echo ""
    echo "Examples:"
    echo "  # Start with defaults"
    echo "  ./scripts/start_database_server.sh"
    echo ""
    echo "  # Production mode with 6 workers"
    echo "  ./scripts/start_database_server.sh --env production --workers 6"
    echo ""
    echo "  # Custom port"
    echo "  ./scripts/start_database_server.sh --port 5004"
    exit 0
}

# Function to stop existing server
function stop_server {
    echo "Stopping existing Database server..."
    
    PID=$(lsof -ti:$PORT 2>/dev/null || true)
    if [ ! -z "$PID" ]; then
        echo "Stopping Database server on port $PORT (PID: $PID)"
        kill -TERM $PID 2>/dev/null || true
        sleep 2
        # Force kill if still running
        if kill -0 $PID 2>/dev/null; then
            echo "Force stopping Database server..."
            kill -KILL $PID 2>/dev/null || true
        fi
        echo "Database server stopped."
    else
        echo "No Database server running on port $PORT"
    fi
}

# Function to show logs
function show_logs {
    LOG_FILE="logs/database_server.log"
    if [ -f "$LOG_FILE" ]; then
        echo "Recent Database server logs:"
        tail -n 50 "$LOG_FILE"
    else
        echo "No log file found at $LOG_FILE"
    fi
}

# Function to check if server is already running
function check_running {
    PID=$(lsof -ti:$PORT 2>/dev/null || true)
    if [ ! -z "$PID" ]; then
        echo "Database server is already running on port $PORT (PID: $PID)"
        echo "Use --stop to stop it first or choose a different port"
        exit 1
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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
            exit 0
            ;;
        --logs)
            show_logs
            exit 0
            ;;
        --help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate environment
if [[ "$ENV" != "development" && "$ENV" != "production" ]]; then
    echo "Error: Environment must be 'development' or 'production'"
    exit 1
fi

# Validate port
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || [[ "$PORT" -lt 1 || "$PORT" -gt 65535 ]]; then
    echo "Error: Port must be a number between 1 and 65535"
    exit 1
fi

# Check if server is already running (unless we're stopping it)
check_running

echo "Starting Database FastAPI server..."
echo "Configuration:"
echo "  - Port: $PORT"
echo "  - Host: $HOST"
echo "  - Environment: $ENV"
echo "  - Workers: $WORKERS"  
echo "  - Log Level: $LOG_LEVEL"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Set up logging
LOG_FILE="logs/database_server.log"
PID_FILE="logs/database_server.pid"

# Export environment variables
export PORT=$PORT
export HOST=$HOST
export LOG_LEVEL=$LOG_LEVEL
export PYTHONPATH=$(pwd):$PYTHONPATH

# Start the server
if [[ "$ENV" == "production" ]]; then
    echo "Starting in production mode with $WORKERS workers..."
    nohup uvicorn src.server.database_app:app \
        --host $HOST \
        --port $PORT \
        --workers $WORKERS \
        --log-level $LOG_LEVEL \
        --access-log \
        --use-colors \
        > $LOG_FILE 2>&1 &
else
    echo "Starting in development mode..."
    nohup uvicorn src.server.database_app:app \
        --host $HOST \
        --port $PORT \
        --reload \
        --log-level $LOG_LEVEL \
        --access-log \
        --use-colors \
        > $LOG_FILE 2>&1 &
fi

# Save PID for later reference
echo $! > $PID_FILE

echo "Database server started with PID: $(cat $PID_FILE)"
echo "Server running at: http://$HOST:$PORT"
echo "Health check: http://$HOST:$PORT/health"
echo "API docs: http://$HOST:$PORT/docs"
echo ""
echo "View logs with: tail -f $LOG_FILE"
echo "Stop server with: ./scripts/start_database_server.sh --stop"

# Wait a moment and check if server started successfully
sleep 3
if kill -0 $(cat $PID_FILE) 2>/dev/null; then
    echo "✅ Database server started successfully!"
    
    # Test health endpoint
    if command -v curl >/dev/null 2>&1; then
        echo "Testing health endpoint..."
        if curl -s "http://$HOST:$PORT/health" >/dev/null; then
            echo "✅ Health check passed!"
        else
            echo "⚠️  Health check failed - server may still be starting"
        fi
    fi
else
    echo "❌ Failed to start Database server"
    echo "Check logs: tail $LOG_FILE"
    exit 1
fi

echo ""
echo "Database server is ready to handle database operations!"
echo "Main endpoints:"
echo "  - POST /api/database/create - Create user databases"
echo "  - POST /api/evaluate - Query databases"
echo "  - GET /health - Health check" 