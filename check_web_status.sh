#!/usr/bin/env bash
# ==============================================================================
# Check Streamlit & Ngrok background status and URL
# ==============================================================================

echo "=========================================================="
echo "📊 Checking Background Web & Ngrok Status"
echo "=========================================================="

echo "🖥️ Running Processes:"
ps aux | grep -E "(streamlit run app.py|ngrok http)" | grep -v grep || echo "   (No active processes found)"
echo ""

echo "🌐 Active Ngrok Public URL(s):"
TUNNELS=$(curl -s http://127.0.0.1:4040/api/tunnels 2>/dev/null)
if [ -n "$TUNNELS" ]; then
    URLS=$(echo "$TUNNELS" | grep -o 'https://[^"]*\.ngrok[^"]*')
    if [ -n "$URLS" ]; then
        echo "$URLS" | while read -r url; do
            echo "   🔗 $url"
        done
    else
        echo "   ⚠️ Ngrok API reachable, but no active tunnels found."
    fi
else
    echo "   ⚠️ Ngrok is not running or API (port 4040) is offline."
fi
echo "=========================================================="
