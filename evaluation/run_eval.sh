#!/bin/bash
# One-command eval runner
# Usage: bash evaluation/run_eval.sh
# Usage with flags: bash evaluation/run_eval.sh --sample 10

set -e

echo "=============================="
echo "  ACSH-RAG Evaluation Runner"
echo "=============================="

# Activate venv if not already active
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    elif [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
    else
        echo "ERROR: venv not found. Run from project root."
        exit 1
    fi
fi

python evaluation/eval.py "$@"
