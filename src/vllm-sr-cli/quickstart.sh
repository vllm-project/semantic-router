#!/bin/bash
# Quick start script for vLLM Semantic Router CLI

set -e

echo "=========================================="
echo "vLLM Semantic Router - Quick Start"
echo "=========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

echo "✓ Docker is installed"

# Check if we're in the right directory
if [ ! -f "Dockerfile" ]; then
    echo "❌ Please run this script from src/vllm-sr-cli directory"
    exit 1
fi

echo "✓ In correct directory"
echo ""

# Step 1: Build Docker image
echo "Step 1: Building Docker image..."
echo "This may take a few minutes on first run..."
cd ../..
docker build -t vllm-sr:dev -f src/vllm-sr-cli/Dockerfile .
echo "✓ Docker image built successfully"
echo ""

# Step 2: Install CLI
echo "Step 2: Installing CLI..."
cd src/vllm-sr-cli
pip install -e . > /dev/null 2>&1
echo "✓ CLI installed successfully"
echo ""

# Step 3: Verify installation
echo "Step 3: Verifying installation..."
vllm-sr --version
echo ""

# Step 4: Instructions
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Start vLLM Semantic Router:"
echo "   vllm-sr serve ../../config/config.yaml"
echo ""
echo "2. In another terminal, test it:"
echo "   curl http://localhost:8801/v1/chat/completions \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"model\": \"qwen3\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'"
echo ""
echo "3. View logs:"
echo "   vllm-sr logs --follow"
echo ""
echo "4. Stop when done:"
echo "   vllm-sr stop"
echo ""
echo "For more information, see README.md"
echo ""

