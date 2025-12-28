#!/bin/bash
# Start script for router service
# Generates router configuration and starts the router

set -e

CONFIG_FILE="${1:-/app/config.yaml}"
OUTPUT_DIR="${2:-/app/.vllm-sr}"

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

# Start router
echo "Starting router..."
exec /usr/local/bin/router \
    -config="$OUTPUT_DIR/router-config.yaml" \
    -port=50051 \
    -enable-api=true \
    -api-port=8080

