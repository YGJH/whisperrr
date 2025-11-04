#!/bin/bash

# Trap EXIT and INT signals to ensure cleanup
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    
    # Kill Flask process and all its children
    if [ ! -z "$FLASK_PID" ]; then
        echo "Stopping Flask backend (PID: $FLASK_PID)..."
        pkill -P $FLASK_PID 2>/dev/null  # Kill child processes
        kill $FLASK_PID 2>/dev/null
        wait $FLASK_PID 2>/dev/null
    fi
    
    # Kill Next.js process and all its children
    if [ ! -z "$NEXTJS_PID" ]; then
        echo "Stopping Next.js frontend (PID: $NEXTJS_PID)..."
        pkill -P $NEXTJS_PID 2>/dev/null  # Kill child processes
        kill $NEXTJS_PID 2>/dev/null
        wait $NEXTJS_PID 2>/dev/null
    fi
    
    # Double-check: kill any process still using port 8080
    echo "Checking for remaining processes on port 8080..."
    PORT_PID=$(lsof -ti:8080 2>/dev/null)
    if [ ! -z "$PORT_PID" ]; then
        echo "Force killing process on port 8080 (PID: $PORT_PID)..."
        kill -9 $PORT_PID 2>/dev/null
    fi
    
    echo "✅ All services stopped"
    exit 0
}

# Set up trap to catch Ctrl+C (SIGINT) and script exit
trap cleanup EXIT INT TERM

# Start Flask backend in the background
echo "🚀 Starting Flask backend..."
uv run app.py &
FLASK_PID=$!
echo "Flask PID: $FLASK_PID"

# Wait a moment for Flask to start
sleep 2

# Start Next.js frontend in the background
echo "🚀 Starting Next.js frontend..."
npm run dev &
NEXTJS_PID=$!
echo "Next.js PID: $NEXTJS_PID"

# Wait for both processes
wait
