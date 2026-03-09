#!/bin/bash
# Start script for router service
# Generates router configuration and starts the selected router backend.

set -euo pipefail

CONFIG_FILE="${1:-/app/config.yaml}"
OUTPUT_DIR="${2:-/app/.vllm-sr}"
AI_BINDING="${AI_BINDING:-candle}"

echo "Generating router configuration..."
echo "  Config file: $CONFIG_FILE"
echo "  Output dir: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate router config using Python
python3 - "$CONFIG_FILE" "$OUTPUT_DIR" << 'EOF'
import sys
from pathlib import Path
from cli.commands.serve import generate_router_config, copy_defaults_reference

config_file = sys.argv[1]
output_dir = sys.argv[2]

try:
    # Generate router config
    router_config_path = generate_router_config(config_file, output_dir, force=True)
    print(f"✓ Generated router config: {router_config_path}")

    # Copy defaults for reference
    defaults_path = copy_defaults_reference(output_dir)
    print(f"✓ Copied defaults: {defaults_path}")
except Exception as e:
    print(f"ERROR: Failed to generate config: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

# Select router binary. ONNX is only built for the amd64 image variant today,
# so fall back to candle if the requested binary is not present.
case "$AI_BINDING" in
    onnx)
        ROUTER_BINARY="/usr/local/bin/router-onnx"
        ;;
    candle|"")
        ROUTER_BINARY="/usr/local/bin/router-candle"
        ;;
    *)
        echo "ERROR: Unknown AI_BINDING='$AI_BINDING'. Valid values: candle, onnx" >&2
        exit 1
        ;;
esac

if [[ ! -x "$ROUTER_BINARY" ]]; then
    echo "Requested router binary not found: $ROUTER_BINARY (AI_BINDING=$AI_BINDING)" >&2
    echo "Falling back to candle router..." >&2
    ROUTER_BINARY="/usr/local/bin/router-candle"
    AI_BINDING="candle"
fi

echo "Starting router with AI_BINDING=$AI_BINDING..."
exec "$ROUTER_BINARY" \
    -config="$OUTPUT_DIR/router-config.yaml" \
    -port=50051 \
    -enable-api=true \
    -api-port=8080
