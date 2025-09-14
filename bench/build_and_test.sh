#!/bin/bash

# Build and test script for vLLM Semantic Router Bench PyPI package

set -e

echo "🔨 Building vLLM Semantic Router Bench Package"
echo "=============================================="

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/
find vllm_semantic_router_bench/ -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find vllm_semantic_router_bench/ -name "*.pyc" -delete 2>/dev/null || true

# Build the package
echo "📦 Building package..."
python -m build

# Test installation in virtual environment
echo "🧪 Testing installation..."
python -m venv test_env
source test_env/bin/activate

# Install the built package
pip install dist/*.whl

# Test imports
echo "🔍 Testing imports..."
python -m vllm_semantic_router_bench.test_package

# Test CLI commands
echo "🖥️  Testing CLI commands..."
echo "Available commands:"
vllm-semantic-router-bench --help | head -10

# Clean up
deactivate
rm -rf test_env/

echo ""
echo "✅ Package build and test completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Review the built package in dist/"
echo "2. Test installation: pip install dist/*.whl"
echo "3. Upload to PyPI: twine upload dist/*"
echo ""
echo "📦 Files ready for PyPI:"
ls -la dist/
