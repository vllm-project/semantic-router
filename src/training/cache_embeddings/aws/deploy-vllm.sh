#!/bin/bash

set -e

# Navigate to the directory containing the playbooks
cd "$(dirname "$0")"

# Function to display usage
usage() {
    echo "Usage: $0 [deploy|cleanup|help]"
    echo ""
    echo "Commands:"
    echo "  deploy              Launch and configure AWS GPU instance for vLLM training"
    echo "  cleanup             Terminate AWS instances and clean up inventory files"
    echo "  help                Show this help message"
    echo ""
    echo "Instance Configuration:"
    echo "  - Default: g5.12xlarge (4x A10G GPUs, ~$5/hr)"
    echo "  - Edit launch-vllm-instance.yaml to change instance type"
    echo ""
    echo "After deployment:"
    echo "  1. Check vllm-instance-*.txt for SSH and SCP commands"
    echo "  2. Upload data to instance"
    echo "  3. Run training script (see README.md)"
    exit 1
}

# Function to deploy instance
deploy_instance() {
    echo "=== Deploying vLLM Training Instance on AWS ==="
    echo ""
    echo "Configuration:"
    echo "  - Instance Type: g5.12xlarge (4x A10G GPUs)"
    echo "  - Region: us-east-2"
    echo "  - Root Volume: 200GB"
    echo ""

    # Check if ansible is installed
    if ! command -v ansible-playbook &> /dev/null; then
        echo "Error: ansible-playbook not found!"
        echo "Install with: pip install ansible boto3 botocore"
        exit 1
    fi

    # Check if AWS credentials are configured
    if [ ! -f ~/.aws/credentials ] && [ -z "$AWS_ACCESS_KEY_ID" ]; then
        echo "Error: AWS credentials not configured!"
        echo "Configure with: aws configure"
        exit 1
    fi

    echo "Launching instance..."
    ansible-playbook launch-vllm-instance.yaml

    echo ""
    echo "=== Deployment Complete ==="
    echo ""

    # Find the most recent instance details file
    DETAILS_FILE=$(ls -rt vllm-instance-*.txt 2>/dev/null | tail -1)

    if [ -n "$DETAILS_FILE" ]; then
        echo "Instance details saved to: $DETAILS_FILE"
        echo ""
        cat "$DETAILS_FILE"
        echo ""
        echo "=== Next Steps ==="
        echo ""
        echo "1. Upload training data to instance:"
        grep "SCP Upload" "$DETAILS_FILE"
        echo ""
        echo "2. SSH to instance and run training:"
        grep "ssh -i" "$DETAILS_FILE"
        echo ""
        echo "3. Run training (4 GPUs):"
        echo "   cd ~/semantic-router"
        echo "   python3 src/training/cache_embeddings/generate_training_data.py \\"
        echo "     --input data/cache_embeddings/medical/unlabeled_queries.jsonl \\"
        echo "     --output data/cache_embeddings/medical/augmented_full.jsonl \\"
        echo "     --backend vllm \\"
        echo "     --model Qwen/Qwen2.5-1.5B-Instruct \\"
        echo "     --paraphrases 3 \\"
        echo "     --negatives 2 \\"
        echo "     --batch-size 64 \\"
        echo "     --gpu-memory 0.9 \\"
        echo "     --tensor-parallel 4 \\"
        echo "     --checkpoint-interval 50"
        echo ""
        echo "4. Download results when complete:"
        grep "SCP Download" "$DETAILS_FILE"
        echo ""
        echo "5. Cleanup when done:"
        echo "   ./deploy-vllm.sh cleanup"
    else
        echo "Warning: Could not find instance details file"
    fi
}

# Function to cleanup instances
cleanup_instances() {
    echo "=== Cleaning up vLLM Training Instances ==="
    echo ""

    # Check if there are any inventory files
    if ! ls vllm-inventory-*.ini 1> /dev/null 2>&1; then
        echo "No vLLM inventory files found. Nothing to cleanup."
        exit 0
    fi

    echo "Found inventory files. Running cleanup playbook..."
    ansible-playbook cleanup-vllm-instance.yaml

    echo ""
    echo "=== Cleanup Complete ==="
}

# Main script logic
case "${1:-}" in
    deploy)
        deploy_instance
        ;;
    cleanup)
        cleanup_instances
        ;;
    help|-h|--help)
        usage
        ;;
    "")
        # Default behavior - deploy
        deploy_instance
        ;;
    *)
        echo "Unknown command: $1"
        usage
        ;;
esac
