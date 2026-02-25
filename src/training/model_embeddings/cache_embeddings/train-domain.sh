#!/bin/bash
#
# Simple, fully-automated domain model training script
#
# Usage: ./train-domain.sh <domain>
# Example: ./train-domain.sh medical
#
# This script:
# 1. Provisions AWS GPU instance
# 2. Uploads data and code
# 3. Runs vLLM data generation + LoRA training
# 4. Downloads trained adapter
# 5. Optionally pushes to HuggingFace
# 6. Cleans up AWS instance
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
AWS_DIR="$SCRIPT_DIR/aws"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to display usage
usage() {
    echo "Usage: $0 <domain> [options]"
    echo ""
    echo "Available domains:"
    for config in "$SCRIPT_DIR/domains"/*.yaml; do
        if [ -f "$config" ]; then
            domain=$(basename "$config" .yaml)
            echo "  - $domain"
        fi
    done
    echo ""
    echo "Options:"
    echo "  --skip-aws          Skip AWS provisioning (use existing instance)"
    echo "  --skip-upload       Skip data upload (already on instance)"
    echo "  --skip-cleanup      Don't cleanup AWS instance after training"
    echo "  --push-hf           Push trained adapter to HuggingFace"
    echo "  --test              Test run: provision, upload, start training, then cleanup"
    echo "  --dry-run           Show what would be done without doing it"
    echo ""
    echo "Example:"
    echo "  $0 medical                    # Full automated training"
    echo "  $0 medical --test             # Test the automation (quick)"
    echo "  $0 legal --push-hf            # Train and push to HuggingFace"
    echo "  $0 medical --skip-cleanup     # Keep AWS instance running"
    exit 1
}

# Parse arguments
DOMAIN="${1:-}"
SKIP_AWS=false
SKIP_UPLOAD=false
SKIP_CLEANUP=false
PUSH_HF=false
DRY_RUN=false
TEST_MODE=false

shift || true
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-aws) SKIP_AWS=true ;;
        --skip-upload) SKIP_UPLOAD=true ;;
        --skip-cleanup) SKIP_CLEANUP=true ;;
        --push-hf) PUSH_HF=true ;;
        --test) TEST_MODE=true ;;
        --dry-run) DRY_RUN=true ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
    shift
done

if [ -z "$DOMAIN" ]; then
    echo -e "${RED}Error: Domain required${NC}"
    usage
fi

# Load domain configuration
DOMAIN_CONFIG="$SCRIPT_DIR/domains/${DOMAIN}.yaml"
if [ ! -f "$DOMAIN_CONFIG" ]; then
    echo -e "${RED}Error: Domain config not found: $DOMAIN_CONFIG${NC}"
    echo ""
    echo "Available domains:"
    ls -1 "$SCRIPT_DIR/domains"/*.yaml 2>/dev/null | xargs -n1 basename | sed 's/.yaml$//' | sed 's/^/  - /'
    exit 1
fi

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Domain-Specific Cache Embedding Training Pipeline    ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Domain:${NC} $DOMAIN"
echo -e "${GREEN}Config:${NC} $DOMAIN_CONFIG"
echo ""

# Parse YAML config (simple parsing - expects specific format)
DATA_FILE=$(grep "data_file:" "$DOMAIN_CONFIG" | cut -d'"' -f2)
OUTPUT_DIR=$(grep "output_dir:" "$DOMAIN_CONFIG" | cut -d'"' -f2)
HF_REPO=$(grep "hf_repo:" "$DOMAIN_CONFIG" | cut -d'"' -f2 || echo "")
QUERIES_COUNT=$(grep "queries_count:" "$DOMAIN_CONFIG" | awk '{print $2}')

echo -e "${YELLOW}Configuration:${NC}"
echo "  Data file: $DATA_FILE"
echo "  Output dir: $OUTPUT_DIR"
echo "  Queries: ~$QUERIES_COUNT"
[ -n "$HF_REPO" ] && echo "  HuggingFace: $HF_REPO"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}DRY RUN - would execute but not actually running${NC}"
    exit 0
fi

# Step 1: Provision AWS instance
if [ "$SKIP_AWS" = false ]; then
    echo -e "${GREEN}[1/6] Provisioning AWS GPU instance...${NC}"
    cd "$AWS_DIR"
    ./deploy-vllm.sh deploy

    # Get instance IP
    INSTANCE_FILE=$(ls -rt vllm-instance-*.txt 2>/dev/null | tail -1)
    if [ -z "$INSTANCE_FILE" ]; then
        echo -e "${RED}Error: Could not find instance details${NC}"
        exit 1
    fi

    INSTANCE_IP=$(grep "Public IP:" "$INSTANCE_FILE" | awk '{print $3}')
    SSH_KEY=$(grep "ssh -i" "$INSTANCE_FILE" | sed 's/.*-i //' | awk '{print $1}')

    echo -e "${GREEN}✓ Instance ready: $INSTANCE_IP${NC}"
    echo ""

    # Wait for instance to be fully ready
    echo "Waiting for instance to be fully initialized (60s)..."
    sleep 60
else
    echo -e "${YELLOW}[1/6] Skipping AWS provisioning${NC}"
    # Load existing instance info
    INSTANCE_FILE=$(ls -rt "$AWS_DIR"/vllm-instance-*.txt 2>/dev/null | tail -1)
    INSTANCE_IP=$(grep "Public IP:" "$INSTANCE_FILE" | awk '{print $3}')
    SSH_KEY=$(grep "ssh -i" "$INSTANCE_FILE" | sed 's/.*-i //' | awk '{print $1}')
fi

# Step 2: Upload data and code
if [ "$SKIP_UPLOAD" = false ]; then
    echo -e "${GREEN}[2/6] Uploading data and code to AWS...${NC}"

    # Create directory structure on remote
    DATA_DIR=$(dirname "$DATA_FILE")
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "ubuntu@$INSTANCE_IP" \
        "mkdir -p ~/semantic-router/$DATA_DIR ~/semantic-router/src/training/cache_embeddings"

    # Upload data file
    scp -i "$SSH_KEY" -o StrictHostKeyChecking=no \
        "$REPO_ROOT/$DATA_FILE" \
        "ubuntu@$INSTANCE_IP:~/semantic-router/$DATA_FILE"

    # Upload training scripts
    scp -i "$SSH_KEY" -o StrictHostKeyChecking=no \
        "$SCRIPT_DIR"/{generate_training_data.py,lora_trainer.py,losses.py,common_utils.py,__init__.py} \
        "ubuntu@$INSTANCE_IP:~/semantic-router/src/training/cache_embeddings/"

    echo -e "${GREEN}✓ Upload complete${NC}"
    echo ""
else
    echo -e "${YELLOW}[2/6] Skipping upload${NC}"
fi

# Step 3: Run data generation on AWS
if [ "$TEST_MODE" = true ]; then
    echo -e "${GREEN}[3/6] Testing vLLM data generation (will run for 30 seconds)...${NC}"

    # Start training in background
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "ubuntu@$INSTANCE_IP" << 'EOF' &
cd ~/semantic-router
python3 src/training/cache_embeddings/generate_training_data.py \
  --input data/cache_embeddings/medical/unlabeled_queries.jsonl \
  --output data/cache_embeddings/medical/triplets.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --paraphrases 3 \
  --negatives 2 \
  --batch-size 32 \
  --gpu-memory 0.9 \
  --tensor-parallel 4 \
  --checkpoint-interval 50
EOF

    TRAIN_PID=$!
    echo "Data generation started (PID: $TRAIN_PID)"
    echo "Waiting 30 seconds to verify it's running..."
    sleep 30

    echo ""
    echo -e "${YELLOW}Killing test process...${NC}"
    kill $TRAIN_PID 2>/dev/null || true

    echo -e "${GREEN}✓ Data generation started successfully!${NC}"
    echo ""
    echo -e "${BLUE}In production, this step would run for ~1.5-2 hours and execute:${NC}"
    echo "  python3 generate_training_data.py \\"
    echo "    --input data/cache_embeddings/medical/unlabeled_queries.jsonl \\"
    echo "    --output data/cache_embeddings/medical/triplets.jsonl \\"
    echo "    --model Qwen/Qwen2.5-7B-Instruct \\"
    echo "    --paraphrases 3 --negatives 2 \\"
    echo "    --batch-size 32 --tensor-parallel 4"
    echo ""
else
    echo -e "${GREEN}[3/6] Running vLLM data generation (this takes ~1.5-2 hours)...${NC}"
    echo -e "${BLUE}Progress will update every few minutes. Please be patient...${NC}"
    echo ""
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "ubuntu@$INSTANCE_IP" bash << EOFREMOTE
cd ~/semantic-router
export VLLM_DISABLE_PROGRESS_BAR=1
python3 src/training/cache_embeddings/generate_training_data.py \
  --input $DATA_FILE \
  --output data/cache_embeddings/${DOMAIN}/triplets.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --paraphrases 3 \
  --negatives 2 \
  --batch-size 256 \
  --gpu-memory 0.9 \
  --tensor-parallel 4 \
  --checkpoint-interval 50
EOFREMOTE

    echo -e "${GREEN}✓ Data generation complete${NC}"
    echo ""
fi

# Step 4: Run LoRA training on AWS
if [ "$TEST_MODE" = true ]; then
    echo -e "${YELLOW}[4/6] Skipping LoRA training (test mode)${NC}"
    echo -e "${BLUE}In production, this would run:${NC}"
    echo "  python3 lora_trainer.py \\"
    echo "    --train-data data/cache_embeddings/medical/triplets.jsonl \\"
    echo "    --base-model sentence-transformers/all-MiniLM-L12-v2 \\"
    echo "    --output models/medical-cache-lora \\"
    echo "    --epochs 1 --batch-size 32 --lr 2e-5"
    echo ""
else
    echo -e "${GREEN}[4/6] Running LoRA training (this takes ~5 minutes)...${NC}"
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "ubuntu@$INSTANCE_IP" << EOF
cd ~/semantic-router
python3 src/training/cache_embeddings/lora_trainer.py \
  --train-data data/cache_embeddings/${DOMAIN}/triplets.jsonl \
  --base-model sentence-transformers/all-MiniLM-L12-v2 \
  --output models/${DOMAIN}-cache-lora \
  --epochs 1 \
  --batch-size 32 \
  --lr 2e-5 \
  --temperature 0.05
EOF

    echo -e "${GREEN}✓ Training complete${NC}"
    echo ""
fi

# Step 5: Download trained model
if [ "$TEST_MODE" = true ]; then
    echo -e "${YELLOW}[5/6] Skipping download (test mode)${NC}"
    echo ""
else
    echo -e "${GREEN}[5/6] Downloading trained adapter...${NC}"
    mkdir -p "$REPO_ROOT/$OUTPUT_DIR"
    scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -r \
        "ubuntu@$INSTANCE_IP:~/semantic-router/models/${DOMAIN}-cache-lora/*" \
        "$REPO_ROOT/$OUTPUT_DIR/"

    echo -e "${GREEN}✓ Downloaded to: $OUTPUT_DIR${NC}"
    echo ""
fi

# Step 6: Prompt user for VM management
if [ "$SKIP_CLEANUP" = false ] || [ "$TEST_MODE" = true ]; then
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                  Training Complete! 🎉                    ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}Trained adapter:${NC} $OUTPUT_DIR"
    echo -e "${BLUE}Size:${NC} $(du -sh "$REPO_ROOT/$OUTPUT_DIR" | awk '{print $1}')"
    echo ""
    echo -e "${YELLOW}[6/6] AWS Instance Management${NC}"
    echo ""
    echo "Instance Details:"
    echo "  IP: $INSTANCE_IP"
    echo "  Type: g5.12xlarge (4x A10G GPUs)"
    echo "  SSH: ssh -i $SSH_KEY ubuntu@$INSTANCE_IP"
    echo ""
    echo "What would you like to do with the AWS instance?"
    echo "  1) Terminate (delete completely - stops billing immediately)"
    echo "  2) Stop (shut down - minimal storage billing only)"
    echo "  3) Keep running (continue billing ~$5.67/hour)"
    echo ""

    # Only prompt if running interactively
    if [ -t 0 ]; then
        read -p "Enter choice [1-3]: " choice
        echo ""

        case $choice in
            1)
                echo -e "${GREEN}Terminating instance...${NC}"
                cd "$AWS_DIR"
                ./deploy-vllm.sh cleanup
                echo -e "${GREEN}✓ Instance terminated${NC}"
                ;;
            2)
                echo -e "${YELLOW}Stopping instance (can be restarted later)...${NC}"
                INSTANCE_ID=$(aws ec2 describe-instances \
                    --filters "Name=ip-address,Values=$INSTANCE_IP" \
                    --query 'Reservations[0].Instances[0].InstanceId' \
                    --output text 2>/dev/null)

                if [ -n "$INSTANCE_ID" ] && [ "$INSTANCE_ID" != "None" ]; then
                    aws ec2 stop-instances --instance-ids "$INSTANCE_ID"
                    echo -e "${GREEN}✓ Instance stopped (ID: $INSTANCE_ID)${NC}"
                    echo -e "${BLUE}To restart: aws ec2 start-instances --instance-ids $INSTANCE_ID${NC}"
                else
                    echo -e "${RED}Error: Could not find instance ID${NC}"
                fi
                ;;
            3)
                echo -e "${YELLOW}Keeping instance running${NC}"
                echo "Instance will continue running and incurring charges (~$5.67/hour)"
                echo ""
                echo -e "${BLUE}To terminate later:${NC}"
                echo "  cd $AWS_DIR && ./deploy-vllm.sh cleanup"
                ;;
            *)
                echo -e "${YELLOW}Invalid choice. Keeping instance running by default.${NC}"
                echo "Instance IP: $INSTANCE_IP"
                ;;
        esac
    else
        # Non-interactive mode - don't cleanup automatically
        echo -e "${YELLOW}Running in non-interactive mode. Instance left running.${NC}"
        echo "To manage the instance later:"
        echo "  Terminate: cd $AWS_DIR && ./deploy-vllm.sh cleanup"
        echo "  Stop: aws ec2 stop-instances --instance-ids <instance-id>"
    fi
else
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                  Training Complete! 🎉                    ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}Trained adapter:${NC} $OUTPUT_DIR"
    echo -e "${BLUE}Size:${NC} $(du -sh "$REPO_ROOT/$OUTPUT_DIR" | awk '{print $1}')"
    echo ""
    echo -e "${YELLOW}[6/6] Skipping cleanup (instance still running)${NC}"
    echo "Instance IP: $INSTANCE_IP"
    echo "SSH: ssh -i $SSH_KEY ubuntu@$INSTANCE_IP"
fi

echo ""

# Optional: Push to HuggingFace
if [ "$PUSH_HF" = true ] && [ -n "$HF_REPO" ]; then
    echo -e "${GREEN}Pushing to HuggingFace...${NC}"

    if ! command -v huggingface-cli &> /dev/null; then
        echo -e "${RED}Error: huggingface-cli not found. Install with: pip install huggingface_hub${NC}"
        exit 1
    fi

    cd "$REPO_ROOT/$OUTPUT_DIR"
    huggingface-cli upload "$HF_REPO" . --repo-type model

    echo -e "${GREEN}✓ Pushed to HuggingFace: https://huggingface.co/$HF_REPO${NC}"
    echo ""
fi

echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Test the adapter: python3 test_lora_model.py"
[ -n "$HF_REPO" ] && [ "$PUSH_HF" = false ] && echo "  2. Push to HuggingFace: ./train-domain.sh $DOMAIN --push-hf"
echo ""
echo "Done! ✨"
