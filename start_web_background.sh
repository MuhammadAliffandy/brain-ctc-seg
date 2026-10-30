#!/usr/bin/env bash
# ==============================================================================
# Start Streamlit & Ngrok in Background (nohup)
# ==============================================================================
# Usage:
#   ./start_web_background.sh [PORT] [GPU_ID]
# Examples:
#   ./start_web_background.sh           # default port 8501
#   ./start_web_background.sh 8501 5     # port 8501 on GPU 5
# ==============================================================================

PORT=${1:-8501}
GPU_ID=${2:-""}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

mkdir -p logs

echo "=========================================================="
echo "🚀 Starting Brain CT Streamlit & Ngrok in Background"
echo "=========================================================="
echo "📂 Project Dir : $SCRIPT_DIR"
echo "🔌 Port        : $PORT"
if [ -n "$GPU_ID" ]; then
    echo "🎮 GPU Device : CUDA_VISIBLE_DEVICES=$GPU_ID"
fi
echo "=========================================================="

# 1. Stop existing processes on the same port / existing ngrok if running
echo "🧹 Checking & cleaning up old instances..."
pkill -f "streamlit run app.py.*$PORT" 2>/dev/null || true
pkill -f "ngrok http $PORT" 2>/dev/null || true
sleep 1

# 2. Check NumPy version (NumPy < 2 for escnn)
python3 -c "import numpy as np; exit(0 if int(np.__version__.split('.')[0]) < 2 else 1)" 2>/dev/null || {
    echo "⚠️ Downgrading to numpy<2 for escnn compatibility..."
    pip install "numpy<2" --quiet
}

# 3. Start Streamlit in background with nohup
STREAMLIT_LOG="logs/streamlit_${PORT}.log"
echo "▶️ Launching Streamlit (log -> $STREAMLIT_LOG)..."

if [ -n "$GPU_ID" ]; then
    nohup env CUDA_VISIBLE_DEVICES="$GPU_ID" python3 -m streamlit run app.py \
        --server.port "$PORT" \
        --server.address 0.0.0.0 \
        --server.headless true \
        --server.enableCORS false \
        --server.enableXsrfProtection false \
        > "$STREAMLIT_LOG" 2>&1 &
else
    nohup python3 -m streamlit run app.py \
        --server.port "$PORT" \
        --server.address 0.0.0.0 \
        --server.headless true \
        --server.enableCORS false \
        --server.enableXsrfProtection false \
        > "$STREAMLIT_LOG" 2>&1 &
fi

STREAMLIT_PID=$!
echo "✅ Streamlit running with PID: $STREAMLIT_PID"

# Wait briefly for Streamlit to bind
sleep 3

# 4. Start Ngrok in background with nohup
NGROK_LOG="logs/ngrok_${PORT}.log"
echo "▶️ Launching Ngrok tunnel (log -> $NGROK_LOG)..."

# Check for custom ngrok config, fallback to default if not found
NGROK_CONF_ARG=""
if [ -f "$HOME/.config/ngrok/ngrok2.yml" ]; then
    NGROK_CONF_ARG="--config $HOME/.config/ngrok/ngrok2.yml"
elif [ -f "$HOME/.config/ngrok/ngrok.yml" ]; then
    NGROK_CONF_ARG="--config $HOME/.config/ngrok/ngrok.yml"
fi

nohup ngrok http "$PORT" $NGROK_CONF_ARG --host-header="localhost:$PORT" --log=stdout > "$NGROK_LOG" 2>&1 &
NGROK_PID=$!
echo "✅ Ngrok running with PID: $NGROK_PID"

echo "⏳ Waiting for Ngrok tunnel URL (5 seconds)..."
sleep 5

# 5. Retrieve and print Ngrok Public URL from local API
NGROK_URL=$(curl -s http://127.0.0.1:4040/api/tunnels | grep -o 'https://[^"]*\.ngrok[^"]*' | head -n 1)

echo "=========================================================="
if [ -n "$NGROK_URL" ]; then
    echo "🎉 WEB DASHBOARD IS LIVE!"
    echo "🔗 Public URL : $NGROK_URL"
    echo "🏠 Local URL  : http://localhost:$PORT"
else
    echo "⚠️ Ngrok started, but URL could not be automatically retrieved."
    echo "🔍 Check ngrok log: tail -n 20 $NGROK_LOG"
    echo "ℹ️ Or run: curl -s http://127.0.0.1:4040/api/tunnels"
fi
echo "=========================================================="
echo "📋 Helper commands:"
echo "   - View Streamlit log : tail -f $STREAMLIT_LOG"
echo "   - View Ngrok log     : tail -f $NGROK_LOG"
echo "   - Check URL anytime  : ./check_web_status.sh"
echo "   - Stop both services : ./stop_web_background.sh"
echo "=========================================================="
