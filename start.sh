#!/bin/bash
# PaperForge AI - One Click Launcher (Unix/Mac)

echo ""
echo "===================================================="
echo "  PaperForge AI - One Click Launcher"
echo "===================================================="
echo ""

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 not found. Please install Python 3.10+"
    exit 1
fi

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "[WARNING] OPENAI_API_KEY not set"
    echo ""
    read -p "Enter your OpenAI API key: " OPENAI_API_KEY
    export OPENAI_API_KEY
fi

# Change to script directory
cd "$(dirname "$0")"

# Run the launcher
python3 run.py
