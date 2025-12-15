import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, DataCollatorWithPadding
from datasets import load_from_disk

# --- 1. SETUP HEADLESS PLOTTING ---
# Critical for clusters: prevents errors trying to open a window
import matplotlib
matplotlib.use('Agg') 

# --- 2. LOAD BEST MODEL ---
checkpoint_path = "ModernBERT-domain-classifier/checkpoint-198"
print(f"Loading model from {checkpoint_path}...")

model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint_path,
    num_labels=3,
    reference_compile=False
)
tokenizer = AutoTokenizer.from_pretrained("my_model/")
tokenized_dataset = load_from_disk("tokenized_dataset")

# --- 3. RUN PREDICTIONS ---
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
trainer = Trainer(model=model, data_collator=data_collator)

print("Running prediction on test set...")
predictions_output = trainer.predict(tokenized_dataset["test"])

y_preds = np.argmax(predictions_output.predictions, axis=1)
y_true = predictions_output.label_ids

# --- 4. GENERATE & SAVE PLOT ---
# Calculate matrix
cm = confusion_matrix(y_true, y_preds)
labels = ["Easy (0)", "Medium (1)", "Hard (2)"]

# Create the plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix: Question Complexity')

# Save to file
output_file = "confusion_matrix_results.png"
plt.savefig(output_file)
print(f"✅ Success! Plot saved to: {output_file}")