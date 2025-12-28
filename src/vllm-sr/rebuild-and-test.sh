#!/bin/bash
set -e

echo "=========================================="
echo "Rebuild and Test vLLM Semantic Router"
echo "=========================================="
echo ""
echo "This script will:"
echo "  1. Clean up old containers"
echo "  2. Rebuild Docker image with all dependencies"
echo "  3. Start the service"
echo "  4. Verify multi-listener support"
echo ""
echo "Dependencies included in the image:"
echo "  ✓ pydantic>=2.0.0"
echo "  ✓ huggingface_hub[cli]>=0.20.0"
echo "  ✓ Multi-listener support"
echo ""
read -r -p "Press Enter to continue or Ctrl+C to cancel..."
echo ""

# Clean up old containers
echo "1. Cleaning up old containers..."
docker rm -f vllm-sr-container 2>/dev/null || echo "  No container to remove"
echo ""

# Rebuild Docker image
echo "2. Rebuilding Docker image..."
echo "  Building from: $(pwd)/../.."
echo "  Note: Use 'make docker-buildx' for multi-platform builds"
echo ""
cd ../..
docker build -t ghcr.io/vllm-project/semantic-router/vllm-sr:latest -f src/vllm-sr/Dockerfile .
cd src/vllm-sr
echo ""
echo "✓ Image built: ghcr.io/vllm-project/semantic-router/vllm-sr:latest"
echo ""

# Verify dependencies in image
echo "3. Verifying dependencies in image..."
docker run --rm vllm-sr:dev /bin/sh -c "
    echo '  Checking pydantic...'
    python -c 'import pydantic; print(f\"    ✓ pydantic {pydantic.__version__}\")'
    echo '  Checking huggingface_hub...'
    python -c 'import huggingface_hub; print(f\"    ✓ huggingface_hub {huggingface_hub.__version__}\")'
    echo '  Checking huggingface-cli...'
    huggingface-cli --version | sed 's/^/    ✓ /'
"
echo ""

# Show config listeners
echo "4. Checking config.yaml listeners..."
python -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    listeners = config.get('listeners', [])
    print(f'  Found {len(listeners)} listener(s):')
    for listener in listeners:
        name = listener.get('name', 'unknown')
        port = listener.get('port', 'unknown')
        print(f'    - {name}: port {port}')
"
echo ""

# Start service
echo "5. Starting service with local image..."
python -m cli.main serve config.yaml --image vllm-sr:dev --image-pull-policy never
echo ""

# Wait a bit for startup
echo "6. Waiting for services to start (30 seconds)..."
sleep 30
echo ""

# Check container status
echo "7. Container status:"
docker ps --filter name=vllm-sr-container --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""

# Show logs
echo "8. Recent logs:"
docker logs vllm-sr-container --tail 100
echo ""

echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo ""
echo "Check health:"
echo "  curl http://localhost:8000/healthz"
echo "  curl http://localhost:8080/healthz"
echo ""
echo "View logs:"
echo "  docker logs -f vllm-sr-container"
echo ""
echo "Stop service:"
echo "  vllm-sr stop"
echo ""

