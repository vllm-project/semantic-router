# Usage Guide

Technical guide for training and using domain-adapted embedding models.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare data (using MedQuAD as example)
python prepare_data.py --source huggingface --dataset keivalya/MedQuad-MedicalQnADataset

# 3. Train
python train.py --data-dir data --output-dir models/trained

# 4. Use trained model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("models/trained/best", trust_remote_code=True)
embeddings = model.encode(["Your query here"])
```

---

## Data Preparation

### Option 1: HuggingFace Dataset

```bash
python prepare_data.py --source huggingface --dataset keivalya/MedQuad-MedicalQnADataset
```

### Option 2: Custom JSON File

Create a JSON file with Q&A pairs:

```json
[
  {"question": "What are the symptoms of diabetes?", "answer": "Diabetes symptoms include..."},
  {"question": "How is hypertension treated?", "answer": "Hypertension treatment involves..."}
]
```

Then run:

```bash
python prepare_data.py --source json --input-file your_data.json
```

### Option 3: JSONL File

```bash
python prepare_data.py --source jsonl --input-file your_data.jsonl
```

### Output Files

```
data/
├── corpus_chunks.pkl      # Document chunks for retrieval
├── train_queries.pkl      # Training queries with ground-truth
├── test_queries.pkl       # Test queries for evaluation
├── corpus_sample.json     # Sample for inspection
└── queries_sample.json    # Sample for inspection
```

---

## Training

### Basic Training

```bash
python train.py --data-dir data --output-dir models/trained
```

### Custom Configuration

```bash
python train.py \
    --data-dir data \
    --output-dir models/trained \
    --base-model llm-semantic-router/mmbert-embed-32k-2d-matryoshka \
    --iterations 2 \
    --learning-rate 5e-5 \
    --margin 0.1
```

### All Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | (required) | Directory with prepared data |
| `--output-dir` | `models/trained` | Where to save trained model |
| `--base-model` | `llm-semantic-router/mmbert-embed-32k-2d-matryoshka` | Base model to fine-tune |
| `--iterations` | `2` | Number of training iterations |
| `--learning-rate` | `5e-5` | Learning rate (**critical - use 5e-5, not lower**) |
| `--epochs` | `2` | Epochs per iteration |
| `--batch-size` | `8` | Training batch size |
| `--margin` | `0.1` | TripletLoss margin |
| `--easy-to-hard-ratio` | `2` | Ratio of easy:hard triplets |
| `--num-queries` | all | Limit training queries |

---

## Critical Hyperparameters

These settings are essential for good results:

| Parameter | Value | Why It Matters |
|-----------|-------|----------------|
| **learning-rate** | `5e-5` | Lower values (5e-7) result in minimal improvement |
| **margin** | `0.1` | Default margin (~5.0) causes severe forgetting |
| easy:hard ratio | `2:1` | Easy triplets reinforce existing knowledge |
| Accumulate triplets | Yes | Don't replace, add to training set each iteration |
| iterations | `2` | Second iteration provides additional gains |

---

## Using the Trained Model

```python
from sentence_transformers import SentenceTransformer

# Load trained model
model = SentenceTransformer("models/trained/best", trust_remote_code=True)

# Encode queries and documents
query_embedding = model.encode("What are the symptoms of diabetes?")
doc_embeddings = model.encode([
    "Diabetes symptoms include increased thirst...",
    "The weather today is sunny...",
])

# Compute similarity
similarities = query_embedding @ doc_embeddings.T
print(similarities)  # Higher score = more relevant
```

---

## Adapting to Other Domains

1. **Collect Q&A pairs** for your domain
   - Question: Natural language query
   - Answer: Document/passage that answers the question

2. **Prepare data**

   ```bash
   python prepare_data.py --source json --input-file your_domain_qa.json
   ```

3. **Train**

   ```bash
   python train.py --data-dir data --output-dir models/your_domain
   ```

4. **Evaluate** - Check training_summary.json for metrics

---

## Requirements

- Python 3.8+
- PyTorch 2.0+
- sentence-transformers 2.2+
- GPU recommended (NVIDIA L4 or better with 24GB+ VRAM)

Install:

```bash
pip install -r requirements.txt
```

---

## Files

```
domain_adapted_embeddings/
├── README.md           # Overview and results
├── USAGE.md            # This file
├── train.py            # Main training script
├── prepare_data.py     # Data preparation (JSON, JSONL, HuggingFace)
├── requirements.txt    # Python dependencies
└── .gitignore          # Ignore data/models
```
