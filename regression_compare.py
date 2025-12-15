import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, DataCollatorWithPadding
from datasets import load_from_disk

# 1. Load Model & Data
model_path = "ModernBERT-complexity-regression/checkpoint-291"
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1, reference_compile=False)
tokenizer = AutoTokenizer.from_pretrained("my_model/")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
dataset = load_from_disk("tokenized_dataset_regression")

# 2. Get Predictions
trainer = Trainer(model=model, data_collator=data_collator)
print("Predicting...")
output = trainer.predict(dataset["test"])

# 3. Get Data Pairs
predicted_scores = output.predictions.flatten()
actual_scores = output.label_ids

# 4. Plot Scatter
plt.figure(figsize=(8, 8))
sns.scatterplot(x=actual_scores, y=predicted_scores, alpha=0.3, color="blue")

# Add a "Perfect Line" (Target)
plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Perfect Prediction")

plt.title("Actual vs. Predicted Complexity")
plt.xlabel("Actual Score (Ground Truth)")
plt.ylabel("Model Prediction")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend()

plt.savefig("complexity_scatter.png")
print("✅ Saved to complexity_scatter.png")