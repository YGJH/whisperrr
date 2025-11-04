#!/bin/bash

echo "🛑 Stopping all Whisper services..."

# Kill Flask app
echo "Stopping Flask (uv run app.py)..."
pkill -f "uv run app.py" 2>/dev/null
pkill -f "app.py" 2>/dev/null

# Kill Next.js dev server
echo "Stopping Next.js (npm run dev)..."
pkill -f "next dev" 2>/dev/null
pkill -f "next-server" 2>/dev/null

# Kill any process on port 8080 (Flask)
PORT_8080=$(lsof -ti:8080 2>/dev/null)
if [ ! -z "$PORT_8080" ]; then
    echo "Killing process on port 8080 (PID: $PORT_8080)..."
    kill -9 $PORT_8080 2>/dev/null
fi

# Kill any process on port 3000 (Next.js)
PORT_3000=$(lsof -ti:3000 2>/dev/null)
if [ ! -z "$PORT_3000" ]; then
    echo "Killing process on port 3000 (PID: $PORT_3000)..."
    kill -9 $PORT_3000 2>/dev/null
fi

# Kill any running main.py transcription jobs
echo "Stopping any running transcription jobs..."
pkill -f "main.py" 2>/dev/null

echo "✅ All services stopped"
