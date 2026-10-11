#!/usr/bin/env bash
# ==============================================================================
# VoxContrast - Interactive Web Application Launch Script
# Launches FastAPI server with integrated interactive single-page dashboard.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo " Starting VoxContrast Interactive Dashboard & API"
echo "======================================================================"

if [ -d ".venv" ]; then
    PYTHON=".venv/bin/python"
    UVICORN=".venv/bin/uvicorn"
else
    PYTHON="python3"
    UVICORN="uvicorn"
fi

export PYTHONPATH="."
PORT=8000
HOST="127.0.0.1"

echo "Verifying dataset manifest..."
if [ ! -f "data/manifest.json" ]; then
    echo "Manifest not found. Generating contrastive speech dataset..."
    $PYTHON scripts/generate_dataset.py
fi

echo ""
echo "Server starting on: http://${HOST}:${PORT}"
echo "API Documentation:  http://${HOST}:${PORT}/docs"
echo "Press Ctrl+C to terminate the server."
echo "======================================================================"

# Automatically open browser on macOS if interactive
if command -v open &> /dev/null && [ -t 1 ]; then
    (sleep 1.5 && open "http://${HOST}:${PORT}") &
fi

exec $UVICORN backend.main:app --host "$HOST" --port "$PORT" --log-level info
