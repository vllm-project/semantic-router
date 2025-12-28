#!/bin/bash
# Script to help with publishing vllm-sr to PyPI
# Usage: ./scripts/publish.sh [test|prod|check]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get current version from pyproject.toml
CURRENT_VERSION=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')

echo -e "${BLUE}=== vllm-sr Publishing Tool ===${NC}"
echo -e "Current version: ${GREEN}${CURRENT_VERSION}${NC}"
echo ""

# Parse command
COMMAND=${1:-check}

case $COMMAND in
  check)
    echo -e "${BLUE}Checking package...${NC}"
    
    # Install dependencies
    echo "Installing build dependencies..."
    pip3 install -q build twine
    
    # Clean previous builds
    echo "Cleaning previous builds..."
    rm -rf dist/ build/ ./*.egg-info
    
    # Build package
    echo "Building package..."
    python -m build
    
    # Check with twine
    echo "Checking package with twine..."
    twine check dist/*
    
    echo ""
    echo -e "${GREEN}✓ Package check passed!${NC}"
    echo ""
    echo "Package contents:"
    ls -lh dist/
    echo ""
    echo "To publish:"
    echo "  Test PyPI:  ./scripts/publish.sh test"
    echo "  Production: ./scripts/publish.sh prod"
    ;;
    
  test)
    echo -e "${YELLOW}Publishing to Test PyPI...${NC}"
    
    # Check if package is built
    if [ ! -d "dist" ]; then
      echo "Building package first..."
      ./scripts/publish.sh check
    fi
    
    # Upload to Test PyPI
    echo "Uploading to Test PyPI..."
    twine upload --repository testpypi dist/* --verbose
    
    echo ""
    echo -e "${GREEN}✓ Published to Test PyPI!${NC}"
    echo ""
    echo "Test installation:"
    echo "  pip3 install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ vllm-sr==${CURRENT_VERSION}"
    echo ""
    echo "View on Test PyPI:"
    echo "  https://test.pypi.org/project/vllm-sr/${CURRENT_VERSION}/"
    ;;
    
  prod)
    echo -e "${RED}Publishing to Production PyPI...${NC}"
    echo ""
    echo -e "${YELLOW}WARNING: This will publish version ${CURRENT_VERSION} to PyPI!${NC}"
    echo "This action cannot be undone."
    echo ""
    read -p "Are you sure you want to continue? (yes/no): " -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
      echo "Aborted."
      exit 1
    fi
    
    # Check if package is built
    if [ ! -d "dist" ]; then
      echo "Building package first..."
      ./scripts/publish.sh check
    fi
    
    # Upload to PyPI
    echo "Uploading to PyPI..."
    twine upload dist/* --verbose
    
    echo ""
    echo -e "${GREEN}✓ Published to PyPI!${NC}"
    echo ""
    echo "Installation:"
    echo "  pip3 install vllm-sr==${CURRENT_VERSION}"
    echo ""
    echo "View on PyPI:"
    echo "  https://pypi.org/project/vllm-sr/${CURRENT_VERSION}/"
    echo ""
    echo "Next steps:"
    echo "  1. Create git tag: git tag v${CURRENT_VERSION}"
    echo "  2. Push tag: git push origin v${CURRENT_VERSION}"
    echo "  3. Create GitHub release"
    ;;
    
  *)
    echo -e "${RED}Unknown command: $COMMAND${NC}"
    echo ""
    echo "Usage: ./scripts/publish.sh [check|test|prod]"
    echo ""
    echo "Commands:"
    echo "  check - Build and check package (default)"
    echo "  test  - Publish to Test PyPI"
    echo "  prod  - Publish to Production PyPI"
    exit 1
    ;;
esac

