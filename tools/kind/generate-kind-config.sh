#!/usr/bin/env bash
# Generate kind-config.yaml with dynamic project root path
# This script auto-detects the project root and generates the configuration

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get project root (two levels up from tools/kind/)
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Template and output files
TEMPLATE_FILE="${SCRIPT_DIR}/kind-config.yaml.template"
OUTPUT_FILE="${SCRIPT_DIR}/kind-config.yaml"

# Check if template exists
if [[ ! -f "${TEMPLATE_FILE}" ]]; then
    echo -e "${RED}Error: Template file not found: ${TEMPLATE_FILE}${NC}" >&2
    exit 1
fi

# Check if models directory exists
MODELS_DIR="${PROJECT_ROOT}/models"
if [[ ! -d "${MODELS_DIR}" ]]; then
    echo -e "${YELLOW}Warning: Models directory does not exist: ${MODELS_DIR}${NC}" >&2
    echo -e "${YELLOW}Creating models directory...${NC}"
    mkdir -p "${MODELS_DIR}"
fi

# Generate the configuration file
echo -e "${GREEN}Generating kind configuration...${NC}"
echo "  Project root: ${PROJECT_ROOT}"
echo "  Models dir:   ${MODELS_DIR}"
echo "  Output file:  ${OUTPUT_FILE}"

# Use envsubst to replace ${PROJECT_ROOT} in template
export PROJECT_ROOT
envsubst < "${TEMPLATE_FILE}" > "${OUTPUT_FILE}"

echo -e "${GREEN}✓ Generated ${OUTPUT_FILE}${NC}"
echo ""
echo "You can now create the kind cluster with:"
echo "  kind create cluster --config ${OUTPUT_FILE}"
