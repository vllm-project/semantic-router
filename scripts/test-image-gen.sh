#!/bin/bash
# Test image generation with vLLM-Omni
# Usage: ./scripts/test-image-gen.sh

set -e

VLLM_OMNI_URL="${VLLM_OMNI_URL:-http://localhost:8001}"
OUTPUT_FILE="${OUTPUT_FILE:-/tmp/generated_image.png}"

echo "=============================================="
echo "  Image Generation Test"
echo "=============================================="
echo ""

# Check vLLM-Omni health
echo "1. Checking vLLM-Omni health..."
if curl -sf "${VLLM_OMNI_URL}/health" > /dev/null 2>&1; then
    echo "   ✓ vLLM-Omni is healthy at ${VLLM_OMNI_URL}"
else
    echo "   ✗ vLLM-Omni is not running at ${VLLM_OMNI_URL}"
    exit 1
fi

echo ""
echo "2. Generating image..."
echo "   Prompt: 'A beautiful sunset over mountains with orange and purple sky'"
echo "   Size: 512x512"
echo "   Steps: 4"
echo ""

START_TIME=$(date +%s)

RESPONSE=$(curl -sS "${VLLM_OMNI_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "black-forest-labs/FLUX.1-schnell",
        "messages": [
            {"role": "user", "content": "A beautiful sunset over mountains with orange and purple sky"}
        ],
        "extra_body": {
            "width": 512,
            "height": 512,
            "num_inference_steps": 4
        }
    }')

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "   Generation took ${DURATION}s"
echo ""

# Check if response contains image
if echo "$RESPONSE" | grep -q "image_url"; then
    echo "   ✓ Image generated successfully!"
    
    # Extract and save the image
    IMAGE_DATA=$(echo "$RESPONSE" | python3 -c "
import json
import sys
import base64

data = json.load(sys.stdin)
content = data['choices'][0]['message']['content']
if isinstance(content, list):
    for part in content:
        if part.get('type') == 'image_url':
            url = part['image_url']['url']
            if url.startswith('data:image/'):
                # Extract base64 data
                b64_data = url.split(',')[1]
                print(b64_data)
                break
")
    
    if [ -n "$IMAGE_DATA" ]; then
        echo "$IMAGE_DATA" | base64 -d > "$OUTPUT_FILE"
        echo "   ✓ Image saved to: $OUTPUT_FILE"
        echo ""
        echo "   Image info:"
        file "$OUTPUT_FILE" 2>/dev/null || echo "   (file command not available)"
        ls -lh "$OUTPUT_FILE" 2>/dev/null | awk '{print "   Size:", $5}'
    fi
else
    echo "   ✗ No image in response"
    echo ""
    echo "   Response:"
    echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
    exit 1
fi

echo ""
echo "=============================================="
echo "  ✓ Test Passed"
echo "=============================================="
