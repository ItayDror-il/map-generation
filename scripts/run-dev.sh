#!/bin/bash
# Development startup script
# Runs both backend API and frontend dev server

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Starting Sequence Map Generator (Development)"
echo "=============================================="

# Check for API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Error: ANTHROPIC_API_KEY not set"
    echo "Run: export ANTHROPIC_API_KEY='your-key'"
    exit 1
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

# Start backend
echo ""
echo "Starting backend API on http://localhost:8000..."
cd "$PROJECT_ROOT/api"
python server.py &
BACKEND_PID=$!

# Wait for backend to be ready
sleep 2

# Start frontend
echo ""
echo "Starting frontend on http://localhost:3000..."
cd "$PROJECT_ROOT/frontend"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "=============================================="
echo "Services running:"
echo "  - Backend API: http://localhost:8000"
echo "  - Frontend:    http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all services"
echo "=============================================="

# Wait for both processes
wait
