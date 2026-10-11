#!/usr/bin/env bash
# ==============================================================================
# VoxContrast - Test Suite & Temporal Grounding Benchmark Execution Script
# Runs multi-tiered pytest suite and evaluate_temporal_iou.py acceptance benchmark.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo " VoxContrast Verification Suite"
echo "======================================================================"

# Activate Python virtual environment
if [ -d ".venv" ]; then
    PYTHON=".venv/bin/python"
    PYTEST=".venv/bin/pytest"
else
    PYTHON="python3"
    PYTEST="pytest"
fi

export PYTHONPATH="."

echo ""
echo "[1/2] Executing Pytest Verification Suite (Unit, Integration, Adversarial)..."
echo "----------------------------------------------------------------------"
$PYTEST tests/ -v

echo ""
echo "[2/2] Executing Temporal Flaw Grounding Benchmark (IoU & Boundary Gates)..."
echo "----------------------------------------------------------------------"
$PYTHON tests/evaluate_temporal_iou.py

echo ""
echo "======================================================================"
echo " All tests and temporal grounding benchmark gates passed successfully!"
echo "======================================================================"
