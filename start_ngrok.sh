#!/bin/bash
# ==============================================================================
# Helper Script: Expose Brain CT Agreement Dashboard via Ngrok
# ==============================================================================
# Usage:
#   ./start_ngrok.sh [PORT] [NGROK_AUTHTOKEN]
#
# Examples:
#   ./start_ngrok.sh 8502
#   ./start_ngrok.sh 8502 2abcdef... (if authtoken is needed)
# ==============================================================================

PORT=${1:-8502}
TOKEN=${2}

echo "🌐 Setting up Ngrok tunnel for Streamlit on port $PORT..."

# 1. If authtoken is passed
if [ -n "$TOKEN" ]; then
    echo "🔑 Configuring ngrok authtoken..."
    if command -v ngrok &> /dev/null; then
        ngrok config add-authtoken "$TOKEN"
    elif python3 -c "import pyngrok" &> /dev/null; then
        python3 -c "from pyngrok import ngrok; ngrok.set_auth_token('$TOKEN')"
    fi
fi

# 2. Check if ngrok CLI is available
if command -v ngrok &> /dev/null; then
    echo "🚀 Launching native ngrok on port $PORT..."
    ngrok http $PORT --host-header="localhost:$PORT"
    exit 0
fi

# 3. If ngrok CLI is not found, check pyngrok
echo "⚠️ Native 'ngrok' command not found. Checking pyngrok..."
if ! python3 -c "import pyngrok" &> /dev/null; then
    echo "📦 Installing pyngrok via pip..."
    pip install pyngrok --quiet
fi

if python3 -c "import pyngrok" &> /dev/null; then
    echo "🚀 Starting tunnel via pyngrok on port $PORT..."
    python3 -c "
import sys, time
from pyngrok import ngrok, conf

port = int('$PORT')
try:
    public_url = ngrok.connect(port, 'http', host_header=f'localhost:{port}')
    print('\n' + '='*60)
    print(f'🎉 NGROK TUNNEL ONLINE!')
    print(f'🔗 Public Dashboard URL: {public_url}')
    print('='*60 + '\n')
    print('Press Ctrl+C to stop the tunnel.\n')
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print('\n🛑 Stopping ngrok tunnel...')
    ngrok.kill()
except Exception as e:
    print(f'❌ Error starting ngrok: {e}')
    if 'authtoken' in str(e).lower():
        print('👉 Tip: Ngrok requires a free account authtoken.')
        print('   Run: ./start_ngrok.sh $PORT <YOUR_NGROK_TOKEN>')
"
    exit 0
fi

echo "❌ Failed to start ngrok. Please install ngrok or run with SSH port-forwarding."
