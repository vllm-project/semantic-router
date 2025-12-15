import matplotlib
matplotlib.use('Agg') # Headless plotting for cluster
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, DataCollatorWithPadding
from datasets import load_from_disk

# 1. Load Model
model_path = "ModernBERT-complexity-regression/checkpoint-291"
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1, reference_compile=False)
tokenizer = AutoTokenizer.from_pretrained("my_model/")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 2. Load Test Data
dataset = load_from_disk("tokenized_dataset_regression")
trainer = Trainer(model=model, data_collator=data_collator)

# 3. Get Predictions
print("Predicting on full test set...")
output = trainer.predict(dataset["test"])
scores = output.predictions.flatten() # Flatten [N,1] to [N]

# 4. Plot Histogram
plt.figure(figsize=(10, 6))
sns.histplot(scores, bins=50, kde=True, color="blue")
plt.axvline(x=0.5, color='red', linestyle='--', label="RAG Threshold (0.5)")
plt.title("Distribution of Complexity Scores")
plt.xlabel("Predicted Complexity (0=Easy, 1=Hard)")
plt.ylabel("Count")
plt.legend()

plt.savefig("complexity_histogram.png")
print("✅ Saved plot to complexity_histogram.png")