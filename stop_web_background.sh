#!/usr/bin/env bash
# ==============================================================================
# Stop Streamlit & Ngrok background instances
# ==============================================================================

PORT=${1:-""}

echo "🛑 Stopping Streamlit and Ngrok background processes..."

if [ -n "$PORT" ]; then
    echo "🎯 Targeting port $PORT..."
    pkill -f "streamlit.*app.py.*$PORT" 2>/dev/null && echo "✅ Streamlit on port $PORT stopped." || echo "ℹ️ No Streamlit process found on port $PORT."
    pkill -f "ngrok http $PORT" 2>/dev/null && echo "✅ Ngrok for port $PORT stopped." || echo "ℹ️ No Ngrok process found for port $PORT."
else
    pkill -f "streamlit.*app.py" 2>/dev/null && echo "✅ All Streamlit app.py processes stopped." || echo "ℹ️ No Streamlit processes found."
    pkill -f "ngrok http" 2>/dev/null && echo "✅ All Ngrok tunnels stopped." || echo "ℹ️ No Ngrok tunnels found."
fi

echo "✨ Done."
