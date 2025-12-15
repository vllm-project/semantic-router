import os
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

MODEL_CHECKPOINT = "answerdotai/ModernBERT-base" 

DATASET_ID = "zjunlp/FactCHD" 

TEXT_COLUMN_1 = "query"
TEXT_COLUMN_2 = "response"

LABEL_COLUMN = "label"

MAX_LENGTH = 1024 

LABEL_MAP = {"NON-FACTUAL": 0, "FACTUAL": 1} 
NUM_LABELS = len(LABEL_MAP)

def load_and_preprocess_dataset():
    """
    Loads the FactCHD dataset, tokenizes the text, encodes the labels, 
    and prepares the dataset for fine-tuning.
    """
    print(f"1. Loading dataset: {DATASET_ID}")
    raw_datasets = load_dataset(DATASET_ID)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    
    def tokenize_function(examples):
        """
        Tokenizes the input text and prepares it for a sequence-pair classification task.
        We concatenate the 'query' and 'response' columns, separated by the special [SEP] token.
        """
        tokenized_examples = tokenizer(
            examples[TEXT_COLUMN_1],
            examples[TEXT_COLUMN_2],
            padding="max_length", # Pad to max_length for uniform tensor size
            truncation=True,      # Truncate to MAX_LENGTH if necessary
            max_length=MAX_LENGTH
        )
        
        if "token_type_ids" in tokenized_examples:
             del tokenized_examples["token_type_ids"]
        
        tokenized_examples["labels"] = [
            LABEL_MAP[label] for label in examples[LABEL_COLUMN]
        ] # Need to encode labels
        
        return tokenized_examples

    print(f"2. Tokenizing and processing dataset with MAX_LENGTH={MAX_LENGTH}...")

    processed_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )
    
    print("3. Preprocessing complete. Final dataset structure:")
    print(processed_datasets)
    
    return processed_datasets


if __name__ == "__main__":
    processed_data = load_and_preprocess_dataset()
    
    OUTPUT_DIR = "factchd_modernbert_processed"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"4. Saving processed dataset to local folder: {OUTPUT_DIR}")
    processed_data.save_to_disk(OUTPUT_DIR)
    

    print(f"Data saved successfully. Your finetuning script can now use the model ID '{MODEL_CHECKPOINT}' and load data from '{OUTPUT_DIR}'.")
    
