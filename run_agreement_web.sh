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

# Check if streamlit is available
if ! command -v streamlit &> /dev/null
then
    echo "⚠️ Streamlit not found in current PATH. Attempting python3 -m streamlit..."
    python3 -m streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
else
    streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
fi
