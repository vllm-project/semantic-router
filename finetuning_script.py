import evaluate
import random
import wandb
import torch
import pandas as pd
import json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from dotenv import load_dotenv

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

SEED = 42
set_seed(SEED)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

load_dotenv() # huggingface and wandb credentials

wandb.login()

run = wandb.init(
    project='Driving with Mistral-7B',
    job_type="training",
    anonymous="allow"
)

# ordered in priority (earlier labels supercede later ones)
# labels and their key-phrases
LABELS = {
    "STOPPED": (["not moving", "stopped", "still", "sitting", "stop", "isn't moving", "not moving", "stops", "parked","stationary"]),
    "REVERSE": (["reverse", "reverses", "reversing"]),
    "ACCELERATE": (["accelerates", "resumes", "picks up", "speeding up", "speeds up", "accelerate", "accelerating"]),
    "RIGHT": (["merge", "turn", "turns", "veer", "makes", "veers", "steers", "steering", "merges",  "moves", "moving", "shifting", "switching", "merging", "turning"],["right"]), # needs a word in BOTH to match
    "LEFT": (["merge", "turn", "turns", "veer", "makes", "veers", "steers", "steering", "merges",  "moves", "moving", "shifting", "switching", "merging", "turning"],["left"]), # needs a word in BOTH to match
    "MAINTAIN": (["travelling down", "going fast", "continues", "continue", "steady", "moves down", "moves forward", "moving down", "moving forward", "drives slowly", "stays", "inches", "drives forward", "steadily", "driving slowly", "driving", "drives", "driving forward", "maintains"]),
    "SLOW": (["slows", "slowing", "brakes", "braking", "slow", "is stopping"]),
    "OTHER": ([])
}

BUCKET_MAPPING = {
    "MAINTAIN": "MAINTAIN",
    "ACCELERATE": "MAINTAIN",
    "STOPPED": "SLOW",
    "SLOW": "SLOW",
    "RIGHT": "TURN",
    "LEFT": "TURN",
    "REVERSE": "REVERSE",
    "OTHER": "OTHER"
}

def check_label_match(full_string, label_key):
    if label_key == "OTHER":
        return True

    if label_key == "LEFT" or label_key == "RIGHT":
        match = True
        for key_word_set in LABELS[label_key]:
            set_match = False
            for key_word in key_word_set:
                if key_word in full_string:
                    set_match = True
            match = (match and set_match)
        if match:
            return True

    else:
        for key_word in LABELS[label_key]:
            if key_word in full_string:
                return True

    return False

def get_label(full_string):
    for k in LABELS.keys():
        if (check_label_match(full_string, k)):
            return k

def clean_reason(reason):
    words_to_remove = ["because", "to", "since", "as", "due", "for"]
    # Split the reason into words
    words = reason.split()
    # Check if the first word is one of the words to remove
    if words and words[0] in words_to_remove:
        # Remove the first word
        words = words[1:]
    if words and words[0] in words_to_remove:
        # Remove the first word
        words = words[1:]
    # Join the words back into a string
    return " ".join(words)

def load_examples():

    df = pd.read_csv('dataset.csv')

    df = df.iloc[:, 3:] # remove first 3 cols

    examples = []

    for index, row in df.iterrows():

        # Iterate through columns in steps of 2 (pairing columns)
        for i in range(0, len(row), 2):  # Start from 0, step by 2 to get pairs
            first_value = row[i]
            second_value = row[i + 1] if (i + 1) < len(row) else None  # Handle case where second value might be missing

            # Check if the first value is NaN
            if pd.isna(first_value):
                break  # Stop processing once the first value in the pair is NaN

            if second_value == None or pd.isna(second_value):
                break

            if not isinstance(first_value, (int, float)):
                # Append the pair to the processed list
                examples.append((first_value, second_value))

    return examples

def preprocess_examples(raw_examples):
    examples = []
    for e in raw_examples:
        examples.append((get_label(e[0]), clean_reason(e[1])))
    return examples

def load_data_and_collator(
    dataset_dict,
    split="train",
    tokenizer=None,
    response_template="#### Answer:",
):

    if split not in dataset_dict:
        raise ValueError(f"Invalid split '{split}'. Must be one of {list(dataset_dict.keys())}.")
    dataset = dataset_dict[split]
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )

    return dataset, collator

def response_template():
    return f"#### Answer:"

def prompt_instruction(situation, label=None):
    if label is None:
        return f"#### Choose one single instruction from [MAINTAIN,SLOW,ACCELERATE,RIGHT,LEFT,REVERSE,OTHER] for this situation: {situation}\n #### Answer: "
    else:
        return f"#### Choose one single instruction from [MAINTAIN,SLOW,ACCELERATE,RIGHT,LEFT,REVERSE,OTHER] for this situation: {situation}\n #### Answer: {label}"


def format_prompts(example):
    prompts = []
    for i in range(len(example['description'])):
      instruction = prompt_instruction(example['description'][i], example['label'][i])
      prompts.append(instruction)
    return prompts

def initialize_model_and_tokenizer():

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3") # download MISTRAL model
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.3",
        quantization_config=quant_config,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    model.config.use_cache = False

    return model, tokenizer


def train(
    model,
    dataset,
    tokenizer,
    collator,
    format_prompts_function,
):

    ####### Tune any hyperparameters here ########

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["gate_proj", "down_proj", "up_proj", "q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1
    )

    training_arguments = SFTConfig(
        run_name="driving-mistral",
        output_dir="./results",
        num_train_epochs=4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        do_eval=False,
        logging_steps=50,
        learning_rate=4e-5,
        fp16=True,
        max_grad_norm=0.25,
        warmup_ratio=0.025,
        group_by_length=True,
        lr_scheduler_type="linear",
        seed=42,
        report_to="wandb",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        formatting_func=format_prompts_function,
        data_collator=collator,
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_arguments,
    )

    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    trainer.train()

    return trainer, model


def test(model, tokenizer, dataset, predictions_file='predictions.torch'):

    results = []

    for example in dataset:
        prompt = prompt_instruction(example['description'], None)

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}  # Move values to the device

        output = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)

        def get_label(full_prediction):
            for label in LABELS.keys():
                if label in full_prediction[111:]:
                    return label
            return "OTHER"

        prediction = get_label(tokenizer.decode(output[0], skip_special_tokens=True).strip())

        # Store the prediction and ground truth
        results.append({
            "description": example['description'],
            "prediction": prediction,
            "ground_truth": example['label'],
            "prediction_bucket": BUCKET_MAPPING[prediction],
            "ground_truth_bucket": BUCKET_MAPPING[example['label']]
        })

    def compute_accuracy(results):
        count = 0
        bucket_count = 0
        for result in results:
            # Check if prediction matches the label
            if result['prediction'] == result['ground_truth']: # first 111 chars are the original prompt & don't contain new output
                count += 1

            # Check if pred vs ground truth buckets match
            if result['prediction_bucket'] == result['ground_truth_bucket']:
                bucket_count += 1

        return ((count / len(results)),(bucket_count / len(results)))

    accuracy, bucket_accuracy = compute_accuracy(results)

    print("Exact Accuracy:", accuracy)
    print("Bucket Accuracy:", bucket_accuracy)

    # Save results to a JSON file for later use (to compute additional metrics from this run)
    with open("results_finetuning.json", "w") as file:
        json.dump(results, file, indent=4)

    print("Results saved to results.json")


def main():

    raw_examples = load_examples()
    examples = preprocess_examples(raw_examples)

    df = pd.DataFrame(examples, columns=["label", "description"])
    dataset = Dataset.from_pandas(df)
    shuffled_dataset = dataset.shuffle(seed=42) # shuffle to account for implicit order in existing scenario-reason pairs
    sample_size = int(0.50 * len(shuffled_dataset)) # choose 50% of the dataset for fine-tuning in under 2 hours on gpu_mig40
    sampled_dataset = shuffled_dataset.select(range(sample_size))
    train_test_split = sampled_dataset.train_test_split(test_size=0.3, seed=42)
    validation_test_split = train_test_split['test'].train_test_split(test_size=0.5, seed=42)
    dataset_dict = DatasetDict({
        'train': train_test_split['train'],
        'validation': validation_test_split['train'],
        'test': validation_test_split['test']
    })

    model, tokenizer = initialize_model_and_tokenizer()
    response_token = response_template()
    tokenizer.add_special_tokens({'additional_special_tokens': [response_token]})
    model.resize_token_embeddings(len(tokenizer))

    train_set, collator = load_data_and_collator(dataset_dict, split="train", tokenizer=tokenizer, response_template=response_token)
    trainer, model = train(model, train_set, tokenizer, collator, format_prompts_function=format_prompts)

    # Use validation set in test() for debugging & improving the model
    # validation_set, _ = load_data_and_collator(dataset_dict, split="validation", tokenizer=tokenizer, response_template=response_token)
    # test(model, tokenizer, validation_set)

    # Use test set in test() for reporting
    test_set, _ = load_data_and_collator(dataset_dict, split="test", tokenizer=tokenizer, response_template=response_token)
    test(model, tokenizer, test_set)

    return model


if __name__ == "__main__":
  model = main()
