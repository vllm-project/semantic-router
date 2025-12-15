import pandas as pd
import numpy as np
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import matplotlib
matplotlib.use('Agg') # Headless mode for cluster
import matplotlib.pyplot as plt
import torch
# Disable compilation to prevent "Python.h" errors on the cluster
torch.compile = lambda *args, **kwargs: lambda x: x

CSV_FILE = "rag_decision_data.csv"       
QUESTION_COL = "question"          
DECISION_COL = "rag_decision"         
MODEL_PATH = "ModernBERT-complexity-regression/checkpoint-291"


# 1. Load Data & Model
print(f"Loading data from {CSV_FILE}...")
df = pd.read_csv(CSV_FILE)
print(f"Loaded {len(df)} examples.")

print(f"Loading Gatekeeper model from {MODEL_PATH}...")
# Use GPU if available (device=0), otherwise CPU (-1)
import torch
device = 0 if torch.cuda.is_available() else -1
gatekeeper = pipeline("text-classification", model=MODEL_PATH, tokenizer="my_model/", device=device)

# 2. Run Inference (Get Raw Scores)
print("Running inference to get complexity scores...")
# This returns a list of dicts: [{'label': 'LABEL_0', 'score': 0.12}, ...]
results = gatekeeper(df[QUESTION_COL].tolist(), batch_size=32)

# Extract raw scores (ModernBERT regression output)
# Note: For regression, the 'score' key holds the value directly
df['predicted_score'] = [r['score'] for r in results]

# 3. The "Grid Search" for Best Threshold
thresholds = np.arange(0.20, 0.85, 0.05) # Test from 0.20 to 0.80 in steps of 0.05
metrics = []

print("\n--- EVALUATION RESULTS ---")
print(f"{'Threshold':<10} | {'Accuracy':<10} | {'F1 Score':<10} | {'Recall (Safety)':<15}")
print("-" * 55)

best_f1 = 0
best_thresh = 0.5

for t in thresholds:
    # Apply logic: If Score > Threshold -> 1 (RAG), Else -> 0 (No RAG)
    preds = (df['predicted_score'] > t).astype(int)
    
    acc = accuracy_score(df[DECISION_COL], preds)
    f1 = f1_score(df[DECISION_COL], preds)
    rec = recall_score(df[DECISION_COL], preds) # High Recall means we rarely miss a 'Hard' question
    
    metrics.append({"threshold": t, "f1": f1, "accuracy": acc, "recall": rec})
    
    print(f"{t:.2f}       | {acc:.4f}     | {f1:.4f}     | {rec:.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print("-" * 55)
print(f"\nWINNER: Best Threshold is {best_thresh:.2f} (F1: {best_f1:.4f})")

# 4. Save a Plot (Optional visual check)
metrics_df = pd.DataFrame(metrics)
plt.figure(figsize=(10, 6))
plt.plot(metrics_df['threshold'], metrics_df['f1'], label='F1 Score (Balance)', marker='o')
plt.plot(metrics_df['threshold'], metrics_df['recall'], label='Recall (Safety)', linestyle='--')
plt.axvline(best_thresh, color='red', alpha=0.5, label=f'Best ({best_thresh:.2f})')
plt.title("Performance vs. Threshold Setting")
plt.xlabel("Threshold Knob")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.savefig("threshold_analysis.png")
print("Saved analysis plot to 'threshold_analysis.png'")