from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
import numpy as np
from sklearn.metrics import accuracy_score, f1_score # <--- Added imports
from transformers import DataCollatorWithPadding

# --- FIX 1: Define how to calculate F1 ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Convert raw scores to the predicted class (0, 1, or 2)
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate metrics
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro") # Macro is best for balanced classes
    
    return {"accuracy": acc, "f1": f1}

# Load model from memory
model = AutoModelForSequenceClassification.from_pretrained(
    "my_model/", 
    num_labels=3,  # <--- FIX 2: Explicitly set 3 classes (0, 1, 2)
    reference_compile=False
)
tokenizer = AutoTokenizer.from_pretrained("my_model/")

# Load tokenized dataset
tokenized_dataset = load_from_disk("tokenized_dataset")

# Define training args (16 GB VRAM Optimized)
training_args = TrainingArguments(
    output_dir="ModernBERT-domain-classifier",
    
    # Batch Sizes & Speed
    per_device_train_batch_size=32, # Aggressive but good for short text on 16GB
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=4,
    dataloader_num_workers=4,        # <--- Added to prevent CPU bottleneck
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
    
    # Best Model Logic
    load_best_model_at_end=True,
    metric_for_best_model="f1",      # Now this will work!
    greater_is_better=True           # F1 is better when higher
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