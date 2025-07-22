#!/bin/bash
set -e

echo ""
echo "=== Anthropic Proxy (with LiteLLM): Environment Setup ==="
echo ""

# Step 1: Create and activate virtual environment
echo "👉 Creating Python virtual environment (.venv)..."
python3 -m venv .venv

echo "👉 Activating virtual environment..."
source .venv/bin/activate

# Step 2: Upgrade pip
echo "👉 Upgrading pip..."
pip install --upgrade pip

# Step 3: Install dependencies
echo ""
echo "👉 Installing LiteLLM (with proxy support)..."
pip install "litellm[proxy]"

echo "👉 Installing FastAPI, httpx, uvicorn (proxy dependencies)..."
pip install fastapi httpx uvicorn

# Step 4: Patch for Copilot debug (optional)
echo ""
echo "👉 Patching Copilot authenticator for debug output (if present)..."
PATCH_PATH=".venv/lib/python3.11/site-packages/litellm/llms/github_copilot/authenticator.py"
PATCH_CONTENT='--- a/litellm/llms/github_copilot/authenticator.py
+++ b/litellm/llms/github_copilot/authenticator.py
@@ -19,6 +19,8 @@
    RefreshAPIKeyError,
 )
 
+from litellm._logging import _turn_on_debug; _turn_on_debug()
+
 # Constants
 GITHUB_CLIENT_ID = "Iv1.b507a08c87ecfe98"
 GITHUB_DEVICE_CODE_URL = "https://github.com/login/device/code"
'
echo "$PATCH_CONTENT" | patch "$PATCH_PATH" && echo "✅ Patch applied to $PATCH_PATH" || echo "⚠️ Patch not applied (file may not exist, safe to ignore if not using Copilot)"

# Step 5: Print next steps
echo ""
echo "✅ Environment setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit 'config.yaml' for LiteLLM customization if needed."
echo "  2. Set your OpenAI API Key (if using an OpenAI backend):"
echo "       export OPENAI_API_KEY=your-key-here"
echo "  3. Launch the proxy and backend servers:"
echo "       ./start_servers.sh"
echo "  4. Proxy will listen at:    http://localhost:3000"
echo "     LiteLLM backend at:      http://localhost:4000"
echo ""
echo "If using Copilot, complete browser authentication when prompted (GitHub OAuth)."
echo ""