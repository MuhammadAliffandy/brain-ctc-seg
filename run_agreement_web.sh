#!/bin/bash
# ==============================================================================
# Brain CT Multi-Model Agreement & Comparison Web Application Launcher
# ==============================================================================
# Usage:
#   Local (Mac/PC): ./run_agreement_web.sh
#   DGX Server    : CUDA_VISIBLE_DEVICES=5 ./run_agreement_web.sh
# ==============================================================================

PORT=${1:-8502}

echo "🧠 Starting Brain CT Multi-Model Agreement Dashboard on port $PORT..."
echo "📂 Working directory: $(pwd)"

# Check NumPy version (escnn / lie_learn strictly requires NumPy < 2)
if ! python3 -c "import numpy as np; exit(0 if int(np.__version__.split('.')[0]) < 2 else 1)" 2>/dev/null; then
    echo "⚠️ NumPy 2.x detected! Downgrading to 'numpy<2' for escnn / lie_learn compatibility..."
    pip install "numpy<2" --quiet
fi

# Check and install plotly if missing (preserving numpy<2)
if ! python3 -c "import plotly" &> /dev/null; then
    echo "📦 Installing plotly for interactive charts..."
    pip install plotly "numpy<2" --quiet || true
fi

# Check and install python-docx if missing (preserving numpy<2)
if ! python3 -c "import docx" &> /dev/null; then
    echo "📦 Installing python-docx for clinical Word report downloads..."
    pip install python-docx "numpy<2" --quiet || true
fi

# Check if streamlit is available
if ! command -v streamlit &> /dev/null
then
    echo "⚠️ Streamlit not found in current PATH. Attempting python3 -m streamlit..."
    python3 -m streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false
else
    streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false
fi
