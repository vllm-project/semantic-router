# Simple Domain Model Training

**One command. Fully automated. Production ready.**

## 🚀 Quick Start

```bash
cd src/training/cache_embeddings
./train-domain.sh medical
```

Done! The script automatically:
1. ✅ Provisions AWS GPU instance
2. ✅ Uploads data and code
3. ✅ Runs training (~2 hours)
4. ✅ Downloads trained model
5. ✅ Cleans up AWS
6. ✅ Optionally pushes to HuggingFace

## 💡 Why This is Simple

**Old way** (what we had before):
```bash
# Step 1: Manual AWS provisioning
cd aws
./deploy-vllm.sh

# Step 2: Manually get instance IP and SSH key
cat vllm-instance-*.txt

# Step 3: Manually upload data
scp -i ~/.ssh/key.pem data.jsonl ubuntu@ip:~/

# Step 4: Manually upload scripts
scp -i ~/.ssh/key.pem *.py ubuntu@ip:~/

# Step 5: SSH and run data generation
ssh -i ~/.ssh/key.pem ubuntu@ip
python3 generate_training_data.py --input ... --output ...

# Step 6: SSH and run LoRA training
python3 lora_trainer.py --train-data ... --output ...

# Step 7: Manually download results
scp -i ~/.ssh/key.pem -r ubuntu@ip:~/models ./

# Step 8: Manual cleanup
./deploy-vllm.sh cleanup
```

**New way** (what we have now):
```bash
./train-domain.sh medical
```

**That's it!** 🎉

## 📋 Training Multiple Domains

For your 13 planned domains:

```bash
# Train all at once (run overnight)
for domain in medical legal financial scientific programming history philosophy psychology engineering business education mathematics literature; do
    ./train-domain.sh $domain --push-hf
    sleep 60  # Small delay between trainings
done
```

Or train individually as needed:
```bash
./train-domain.sh legal
./train-domain.sh financial --push-hf
./train-domain.sh scientific
```

## 🎯 Adding a New Domain

### 1. Prepare data
Create `data/cache_embeddings/<domain>/unlabeled_queries.jsonl`:
```jsonl
{"query": "Your first query"}
{"query": "Your second query"}
...
```

### 2. Create config
```bash
cd domains/
cp TEMPLATE.yaml <domain>.yaml
# Edit the file with your domain details
```

### 3. Train
```bash
./train-domain.sh <domain>
```

## 🔧 Advanced Options

```bash
# Push to HuggingFace after training
./train-domain.sh medical --push-hf

# Keep AWS instance running (for debugging)
./train-domain.sh medical --skip-cleanup

# Reuse existing instance
./train-domain.sh legal --skip-aws --skip-upload

# See what would happen without running
./train-domain.sh medical --dry-run
```

## 💰 Cost & Time

| Queries | Time | Cost (@$5/hr) |
|---------|------|---------------|
| 10K | 30 min | ~$2.50 |
| 50K | 2.5 hrs | ~$12.50 |
| 100K | 5 hrs | ~$25 |

**For 13 domains @ 50K queries each:** ~$162 total, ~32 hours runtime

## 📦 HuggingFace Integration

Once trained, adapters can be pushed to HuggingFace:

```bash
./train-domain.sh medical --push-hf
```

Then in semantic-router deployment, users can download:
```python
from huggingface_hub import hf_hub_download

# Download specific domain adapter
adapter_path = hf_hub_download(
    repo_id="your-org/semantic-router-medical-cache",
    filename="adapter_model.safetensors"
)

# Use in semantic router
from sentence_transformers import SentenceTransformer
from peft import PeftModel

base = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
base[0].auto_model = PeftModel.from_pretrained(
    base[0].auto_model,
    adapter_path
)
```

## 🎯 Production Workflow

1. **Data Collection**: Gather unlabeled domain queries
2. **Config Creation**: Copy template, fill in details
3. **Training**: Run `./train-domain.sh <domain>`
4. **Validation**: Test with `test_lora_model.py`
5. **HuggingFace**: Push with `--push-hf` flag
6. **Deployment**: semantic-router downloads from HF automatically

## 📁 File Structure

```
src/training/cache_embeddings/
├── train-domain.sh              ⭐ Main script (use this!)
├── domains/
│   ├── medical.yaml             # Medical domain config
│   ├── legal.yaml               # Legal domain config
│   ├── ...                      # More domains
│   └── TEMPLATE.yaml            # Template for new domains
├── generate_training_data.py    # vLLM data generation
├── lora_trainer.py              # LoRA training
├── test_lora_model.py           # Validation
└── aws/                         # AWS deployment (used internally)
```

## 🎓 Example Session

```bash
$ ./train-domain.sh medical

╔═══════════════════════════════════════════════════════════╗
║     Domain-Specific Cache Embedding Training Pipeline    ║
╚═══════════════════════════════════════════════════════════╝

Domain: medical
Config: domains/medical.yaml

Configuration:
  Data file: data/cache_embeddings/medical/unlabeled_queries.jsonl
  Output dir: models/medical-cache-lora
  Queries: ~44603

[1/6] Provisioning AWS GPU instance...
✓ Instance ready: 3.85.123.45

[2/6] Uploading data and code to AWS...
✓ Upload complete

[3/6] Running vLLM data generation (this takes ~1.5-2 hours)...
Processing batches: 100%|███████████████| 1394/1394 [1:48:23<00:00]
✓ Data generation complete

[4/6] Running LoRA training (this takes ~5 minutes)...
Epoch 1/1: 100%|███████████████| 2217/2217 [4:32<00:00, 8.13it/s]
✓ Training complete

[5/6] Downloading trained adapter...
✓ Downloaded to: models/medical-cache-lora

[6/6] Cleaning up AWS instance...
✓ Cleanup complete

╔═══════════════════════════════════════════════════════════╗
║                  Training Complete! 🎉                    ║
╚═══════════════════════════════════════════════════════════╝

Trained adapter: models/medical-cache-lora
Size: 582 KB

Next steps:
  1. Test the adapter: python3 test_lora_model.py

Done! ✨
```

## 📚 See Also

- [domains/README.md](domains/README.md) - Domain configuration guide
- [docs/README.md](docs/README.md) - Complete technical documentation
- [docs/index.html](docs/index.html) - Interactive documentation with tooltips
