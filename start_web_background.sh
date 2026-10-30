#!/usr/bin/env bash
# ==============================================================================
# Start Streamlit & Ngrok in Background (nohup) with Conda Environment & Auto-Ngrok
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

# 1. Activate conda environment 'brain_env' if available
if command -v conda &> /dev/null; then
    # shellcheck disable=SC1091
    eval "$(conda shell.bash hook 2>/dev/null)"
    if conda info --envs | grep -q "brain_env"; then
        echo "🐍 Activating Conda environment: brain_env..."
        conda activate brain_env
    else
        echo "ℹ️ Conda found, but environment 'brain_env' not detected. Continuing with current env..."
    fi
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate brain_env 2>/dev/null && echo "🐍 Activated Conda environment: brain_env" || true
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate brain_env 2>/dev/null && echo "🐍 Activated Conda environment: brain_env" || true
fi

# 2. Stop existing processes on the same port / existing ngrok if running
echo "🧹 Checking & cleaning up old instances..."
pkill -f "streamlit run app.py.*$PORT" 2>/dev/null || true
pkill -f "ngrok http $PORT" 2>/dev/null || true
pkill -f "pyngrok.*$PORT" 2>/dev/null || true
sleep 1

# 3. Check and install ngrok if not installed on system
if ! command -v ngrok &> /dev/null; then
    echo "⚠️ 'ngrok' binary not found in PATH."
    # Check if user already has ngrok in ~/bin or /usr/local/bin or current folder
    if [ -f "$HOME/bin/ngrok" ]; then
        export PATH="$HOME/bin:$PATH"
    elif [ -f "$SCRIPT_DIR/ngrok" ]; then
        export PATH="$SCRIPT_DIR:$PATH"
    else
        echo "📦 Attempting to install ngrok binary locally (no root required)..."
        mkdir -p "$HOME/bin"
        if command -v curl &> /dev/null; then
            ARCH=$(uname -m)
            if [ "$ARCH" = "x86_64" ]; then
                curl -s https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz -o /tmp/ngrok.tgz 2>/dev/null || true
            elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
                curl -s https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-arm64.tgz -o /tmp/ngrok.tgz 2>/dev/null || true
            fi
            if [ -f /tmp/ngrok.tgz ]; then
                tar -xzf /tmp/ngrok.tgz -C "$HOME/bin" 2>/dev/null || true
                rm -f /tmp/ngrok.tgz
                chmod +x "$HOME/bin/ngrok" 2>/dev/null || true
                export PATH="$HOME/bin:$PATH"
            fi
        fi
    fi
fi

# If ngrok CLI is still not available, ensure pyngrok is installed in python
USE_PYNGROK=false
if ! command -v ngrok &> /dev/null; then
    echo "⚠️ Standalone ngrok binary unavailable. Using pyngrok fallback..."
    if ! python3 -c "import pyngrok" &> /dev/null; then
        echo "📦 Installing pyngrok in $(which python3)..."
        pip install pyngrok --quiet || true
    fi
    USE_PYNGROK=true
else
    echo "✅ Found ngrok executable: $(command -v ngrok)"
fi

# 4. Check NumPy version (NumPy < 2 for escnn)
python3 -c "import numpy as np; exit(0 if int(np.__version__.split('.')[0]) < 2 else 1)" 2>/dev/null || {
    echo "⚠️ Downgrading to numpy<2 for escnn compatibility..."
    pip install "numpy<2" --quiet || true
}

# 5. Start Streamlit in background with nohup
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

# 6. Start Ngrok in background with nohup
NGROK_LOG="logs/ngrok_${PORT}.log"
echo "▶️ Launching Ngrok tunnel (log -> $NGROK_LOG)..."

if [ "$USE_PYNGROK" = true ]; then
    # Launch pyngrok tunnel in background
    nohup python3 -c "
import time, os
from pyngrok import ngrok, conf
# Check config if present
if os.path.exists(os.path.expanduser('~/.config/ngrok/ngrok2.yml')):
    conf.get_default().config_path = os.path.expanduser('~/.config/ngrok/ngrok2.yml')
elif os.path.exists(os.path.expanduser('~/.config/ngrok/ngrok.yml')):
    conf.get_default().config_path = os.path.expanduser('~/.config/ngrok/ngrok.yml')

port = int('$PORT')
tunnel = ngrok.connect(port, 'http', host_header=f'localhost:{port}')
print(f'NGROK_URL={tunnel.public_url}', flush=True)
while True:
    time.sleep(5)
" > "$NGROK_LOG" 2>&1 &
    NGROK_PID=$!
else
    # Launch native ngrok CLI in background
    NGROK_CONF_ARG=""
    if [ -f "$HOME/.config/ngrok/ngrok2.yml" ]; then
        NGROK_CONF_ARG="--config $HOME/.config/ngrok/ngrok2.yml"
    elif [ -f "$HOME/.config/ngrok/ngrok.yml" ]; then
        NGROK_CONF_ARG="--config $HOME/.config/ngrok/ngrok.yml"
    fi

    nohup ngrok http "$PORT" $NGROK_CONF_ARG --host-header="localhost:$PORT" --log=stdout > "$NGROK_LOG" 2>&1 &
    NGROK_PID=$!
fi

echo "✅ Ngrok running with PID: $NGROK_PID"
echo "⏳ Waiting for Ngrok tunnel URL (5 seconds)..."
sleep 5

# 7. Retrieve and print Ngrok Public URL
NGROK_URL=$(curl -s http://127.0.0.1:4040/api/tunnels 2>/dev/null | grep -o 'https://[^"]*\.ngrok[^"]*' | head -n 1)
if [ -z "$NGROK_URL" ] && [ -f "$NGROK_LOG" ]; then
    NGROK_URL=$(grep -o 'https://[^ ]*\.ngrok[^ ]*' "$NGROK_LOG" | tail -n 1)
fi

echo "=========================================================="
if [ -n "$NGROK_URL" ]; then
    echo "🎉 WEB DASHBOARD IS LIVE!"
    echo "🔗 Public URL : $NGROK_URL"
    echo "🏠 Local URL  : http://localhost:$PORT"
else
    echo "⚠️ Ngrok process started, but tunnel URL is not yet ready."
    echo "🔍 Check ngrok log: tail -n 20 $NGROK_LOG"
    echo "ℹ️ Or run: ./check_web_status.sh"
fi
echo "=========================================================="
echo "📋 Helper commands:"
echo "   - View Streamlit log : tail -f $STREAMLIT_LOG"
echo "   - View Ngrok log     : tail -f $NGROK_LOG"
echo "   - Check URL anytime  : ./check_web_status.sh"
echo "   - Stop both services : ./stop_web_background.sh"
echo "=========================================================="
