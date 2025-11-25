#!/bin/bash
# Convert all LoRA models from HuggingFace format to OpenVINO IR format

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${MODELS_DIR:-../models}"
OPENVINO_DIR="${OPENVINO_DIR:-${MODELS_DIR}/openvino}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================================"
echo "OpenVINO LoRA Model Conversion Script"
echo "================================================"
echo ""
echo "Models Directory: $MODELS_DIR"
echo "Output Directory: $OPENVINO_DIR"
echo ""

# Check if Python and required packages are available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python3 not found${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python3 found${NC}"

# Check for required Python packages
if ! python3 -c "import torch; import openvino; import transformers" 2>/dev/null; then
    echo -e "${YELLOW}⚠ Required Python packages not found${NC}"
    echo "Installing required packages..."
    pip install torch openvino transformers --quiet
fi

echo -e "${GREEN}✓ Required packages available${NC}"
echo ""

# Create output directory
mkdir -p "$OPENVINO_DIR"

# Models to convert
MODELS=(
    "lora_intent_classifier_bert-base-uncased_model:bert"
    "lora_intent_classifier_modernbert-base_model:modernbert"
    "lora_jailbreak_classifier_bert-base-uncased_model:bert"
    "lora_jailbreak_classifier_modernbert-base_model:modernbert"
    "lora_pii_detector_bert-base-uncased_model:bert"
    "lora_pii_detector_modernbert-base_model:modernbert"
)

SUCCESS_COUNT=0
TOTAL_COUNT=0

# Convert each model
for model_entry in "${MODELS[@]}"; do
    IFS=':' read -r model_name model_type <<< "$model_entry"
    
    TOTAL_COUNT=$((TOTAL_COUNT + 1))
    
    INPUT_PATH="${MODELS_DIR}/${model_name}"
    OUTPUT_PATH="${OPENVINO_DIR}/${model_name}"
    
    echo "================================================"
    echo "Converting: $model_name ($model_type)"
    echo "================================================"
    
    # Check if input exists
    if [ ! -d "$INPUT_PATH" ]; then
        echo -e "${YELLOW}⚠ Skipping: Model not found at $INPUT_PATH${NC}"
        echo ""
        continue
    fi
    
    # Skip if already converted
    if [ -f "${OUTPUT_PATH}/openvino_model.xml" ]; then
        echo -e "${YELLOW}⚠ Already converted: $OUTPUT_PATH${NC}"
        echo ""
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        continue
    fi
    
    # Run conversion
    if python3 "${SCRIPT_DIR}/convert_lora_models.py" \
        --input "$INPUT_PATH" \
        --output "$OUTPUT_PATH" \
        --type base; then
        echo -e "${GREEN}✓ Successfully converted: $model_name${NC}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo -e "${RED}✗ Failed to convert: $model_name${NC}"
    fi
    
    echo ""
done

# Summary
echo "================================================"
echo "Conversion Summary"
echo "================================================"
echo "Total models: $TOTAL_COUNT"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $((TOTAL_COUNT - SUCCESS_COUNT))"
echo ""

if [ $SUCCESS_COUNT -eq $TOTAL_COUNT ]; then
    echo -e "${GREEN}✓✓✓ All models converted successfully! ✓✓✓${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠ Some models failed to convert${NC}"
    exit 1
fi

