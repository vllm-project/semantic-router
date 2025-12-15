from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score # <--- CHANGED: Regression metrics
from transformers import DataCollatorWithPadding

# --- CHANGED: Regression Metrics ---
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Note: We do NOT use np.argmax here because the model outputs a single raw number.
    # predictions are already the complexity scores (e.g., 0.65)
    
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, predictions)
    
    return {"rmse": rmse, "r2": r2}

# Load model from memory
model = AutoModelForSequenceClassification.from_pretrained(
    "my_model/", 
    num_labels=1,           # <--- CRITICAL: 1 Node triggers Regression (MSE Loss)
    reference_compile=False # Keep this to prevent cluster errors
)
tokenizer = AutoTokenizer.from_pretrained("my_model/")

# Load the NEW regression dataset
# This must match the folder name you used in the notebook
tokenized_dataset = load_from_disk("tokenized_dataset_regression")

# Define training args
training_args = TrainingArguments(
    output_dir="ModernBERT-complexity-regression", # <--- Save to new folder
    
    # Batch Sizes & Speed (Kept your stable settings)
    per_device_train_batch_size=32, 
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=4,
    dataloader_num_workers=4,
    bf16=True, 
    optim="adamw_torch_fused", 
    
    # Optimization
    learning_rate=5e-5,
    num_train_epochs=3,
    
    # Logging & Saving
    logging_strategy="steps",
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    
    # Best Model Logic (Optimizing for RMSE)
    load_best_model_at_end=True,
    metric_for_best_model="rmse",  # <--- Track Error
    greater_is_better=False        # <--- Lower error is better!
)

print("Training Args set...")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

print("Starting Training...")
trainer.train()
print("Training Completed.")