import argparse
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
import wikipediaapi
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- CONFIGURATION (Remains the same) ---
HIGH_LEVEL_CATEGORIES = [
    "Science",
    "History",
    "Technology",
    "Art",
    "Sports",
    "Geography",
    "Philosophy",
    "Mathematics",
]
MMLU_TO_HIGH_LEVEL = {
    "anatomy": "Science",
    "astronomy": "Science",
    "college_biology": "Science",
    "college_chemistry": "Science",
    "college_physics": "Science",
    "conceptual_physics": "Science",
    "electrical_engineering": "Science",
    "high_school_biology": "Science",
    "high_school_chemistry": "Science",
    "high_school_physics": "Science",
    "medical_genetics": "Science",
    "nutrition": "Science",
    "virology": "Science",
    "clinical_knowledge": "Science",
    "college_medicine": "Science",
    "professional_medicine": "Science",
    "human_aging": "Science",
    "human_sexuality": "Science",
    "high_school_us_history": "History",
    "high_school_world_history": "History",
    "prehistory": "History",
    "college_computer_science": "Technology",
    "computer_security": "Technology",
    "high_school_computer_science": "Technology",
    "machine_learning": "Technology",
    "high_school_geography": "Geography",
    "business_ethics": "Philosophy",
    "formal_logic": "Philosophy",
    "jurisprudence": "Philosophy",
    "logical_fallacies": "Philosophy",
    "moral_disputes": "Philosophy",
    "moral_scenarios": "Philosophy",
    "philosophy": "Philosophy",
    "world_religions": "Philosophy",
    "professional_psychology": "Philosophy",
    "high_school_psychology": "Philosophy",
    "sociology": "Philosophy",
    "abstract_algebra": "Mathematics",
    "college_mathematics": "Mathematics",
    "econometrics": "Mathematics",
    "elementary_mathematics": "Mathematics",
    "high_school_mathematics": "Mathematics",
    "high_school_statistics": "Mathematics",
    "professional_accounting": "Mathematics",
}
WIKI_DATA_DIR = "wikipedia_data"
ARTICLES_PER_CATEGORY = 1000


# --- MODEL AND DATASET CLASSES (Remain the same) ---
class MultitaskBertModel(nn.Module):
    def __init__(self, base_model_name, task_configs):
        super(MultitaskBertModel, self).__init__()
        self.bert = AutoModel.from_pretrained(base_model_name)
        self.dropout = nn.Dropout(0.1)
        self.task_heads = nn.ModuleDict()
        self.task_configs = task_configs
        hidden_size = self.bert.config.hidden_size
        for task_name, config in task_configs.items():
            self.task_heads[task_name] = nn.Linear(hidden_size, config["num_classes"])

    def forward(self, input_ids, attention_mask, task_name=None):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = bert_output.last_hidden_state
        attention_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * attention_mask_expanded, 1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        pooled_output = self.dropout(pooled_output)
        outputs = {}
        if task_name:
            outputs[task_name] = self.task_heads[task_name](pooled_output)
        else:
            for task in self.task_heads:
                outputs[task] = self.task_heads[task](pooled_output)
        return outputs


class MultitaskDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, task_name, label = self.samples[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "task_name": task_name,
            "label": torch.tensor(label, dtype=torch.long),
        }


# --- MultitaskTrainer CLASS (Unchanged) ---
class MultitaskTrainer:
    def __init__(self, model, tokenizer, task_configs, device="cuda"):
        self.model = model.to(device) if model is not None else None
        self.tokenizer = tokenizer
        self.task_configs = task_configs
        self.device = device
        self.jailbreak_label_mapping = None
        self.loss_fns = {
            task: nn.CrossEntropyLoss() for task in task_configs if task_configs
        }

    def _get_wiki_articles_recursive(
        self, category_page, max_articles, visited_categories=None
    ):
        if visited_categories is None:
            visited_categories = set()
        articles = []
        visited_categories.add(category_page.title)
        for member in category_page.categorymembers.values():
            if len(articles) >= max_articles:
                break
            if (
                member.ns == wikipediaapi.Namespace.CATEGORY
                and member.title not in visited_categories
            ):
                articles.extend(
                    self._get_wiki_articles_recursive(
                        member, max_articles - len(articles), visited_categories
                    )
                )
            elif member.ns == wikipediaapi.Namespace.MAIN:
                articles.append(member.title)
        return articles

    def download_wikipedia_articles(self, categories, articles_per_category, data_dir):
        logger.info("--- Starting Wikipedia Data Download and Preparation ---")
        os.makedirs(data_dir, exist_ok=True)
        wiki_wiki = wikipediaapi.Wikipedia(
            "MyMultitaskProject (youremail@example.com)", "en"
        )
        for category in categories:
            category_path = os.path.join(data_dir, category)
            os.makedirs(category_path, exist_ok=True)
            logger.info(f"\nFetching articles for category: {category}")
            cat_page = wiki_wiki.page(f"Category:{category}")
            if not cat_page.exists():
                logger.warning(
                    f"  Category page '{category}' does not exist. Skipping."
                )
                continue
            logger.info(
                f"  Recursively searching for up to {articles_per_category} articles..."
            )
            article_titles = self._get_wiki_articles_recursive(
                cat_page, articles_per_category
            )
            logger.info(
                f"  Found {len(article_titles)} potential articles. Now downloading..."
            )
            random.shuffle(article_titles)
            saved_count = 0
            for title in tqdm(article_titles, desc=f"  Downloading '{category}'"):
                safe_filename = (
                    "".join(
                        [c for c in title if c.isalpha() or c.isdigit() or c == " "]
                    ).rstrip()
                    + ".txt"
                )
                file_path = os.path.join(category_path, safe_filename)
                if os.path.exists(file_path):
                    continue
                page = wiki_wiki.page(title)
                if page.exists() and len(page.text) > 500:
                    try:
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(page.text)
                        saved_count += 1
                    except Exception as e:
                        logger.error(f"    Could not save article '{title}': {e}")
            logger.info(f"  Saved {saved_count} new articles for '{category}'.")
        logger.info("\n--- Wikipedia Data Download Complete ---")

    def _load_wikipedia_samples(self, data_dir, category_to_idx):
        samples = []
        if not os.path.exists(data_dir):
            logger.warning(f"Wikipedia data directory '{data_dir}' not found.")
            return samples
        logger.info("Loading Wikipedia articles for category classification...")
        for category_name in category_to_idx.keys():
            category_path = os.path.join(data_dir, category_name)
            if not os.path.isdir(category_path):
                continue
            for filename in os.listdir(category_path):
                if filename.endswith(".txt"):
                    file_path = os.path.join(category_path, filename)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            text = f.read()
                        label_idx = category_to_idx[category_name]
                        samples.append((text, "category", label_idx))
                    except Exception as e:
                        logger.warning(f"Could not read file {file_path}: {e}")
        return samples

    def prepare_datasets(self):
        all_samples = []
        datasets = {}
        logger.info("Preparing combined dataset for category classification...")
        category_to_idx = {cat: idx for idx, cat in enumerate(HIGH_LEVEL_CATEGORIES)}
        try:
            mmlu_dataset = load_dataset("TIGER-Lab/MMLU-Pro")
            questions, categories = (
                mmlu_dataset["test"]["question"],
                mmlu_dataset["test"]["category"],
            )
            mapped_count = 0
            for question, category in zip(questions, categories):
                if category in MMLU_TO_HIGH_LEVEL:
                    high_level_cat = MMLU_TO_HIGH_LEVEL[category]
                    all_samples.append(
                        (question, "category", category_to_idx[high_level_cat])
                    )
                    mapped_count += 1
            logger.info(
                f"Added {mapped_count} mapped samples from MMLU-Pro to category task."
            )
        except Exception as e:
            logger.warning(f"Failed to load MMLU-Pro: {e}")
        wiki_samples = self._load_wikipedia_samples(WIKI_DATA_DIR, category_to_idx)
        all_samples.extend(wiki_samples)
        logger.info(
            f"Added {len(wiki_samples)} samples from Wikipedia to category task."
        )
        datasets["category"] = {
            "label_mapping": {
                "label_to_idx": category_to_idx,
                "idx_to_label": {v: k for k, v in category_to_idx.items()},
            }
        }
        logger.info("Loading PII dataset...")
        try:
            pii_samples = self._load_pii_dataset()
            if pii_samples:
                pii_labels = sorted(list(set([label for _, label in pii_samples])))
                pii_to_idx = {label: idx for idx, label in enumerate(pii_labels)}
                for text, label in pii_samples:
                    all_samples.append((text, "pii", pii_to_idx[label]))
                datasets["pii"] = {
                    "label_mapping": {
                        "label_to_idx": pii_to_idx,
                        "idx_to_label": {v: k for k, v in pii_to_idx.items()},
                    }
                }
                logger.info(f"Added {len(pii_samples)} PII samples to training.")
        except Exception as e:
            logger.warning(f"Failed to load PII dataset: {e}")
        logger.info("Loading real jailbreak dataset...")
        jailbreak_samples = self._load_jailbreak_dataset()
        for text, label in jailbreak_samples:
            all_samples.append((text, "jailbreak", label))
        datasets["jailbreak"] = {"label_mapping": self.jailbreak_label_mapping}
        train_samples, val_samples = train_test_split(
            all_samples, test_size=0.2, random_state=42
        )
        return train_samples, val_samples, datasets

    def _load_pii_dataset(self):
        url = "https://raw.githubusercontent.com/microsoft/presidio-research/refs/heads/master/data/synth_dataset_v2.json"
        dataset_path = "presidio_synth_dataset_v2.json"
        if not Path(dataset_path).exists():
            logger.info(f"Downloading Presidio dataset...")
            response = requests.get(url)
            response.raise_for_status()
            with open(dataset_path, "w", encoding="utf-8") as f:
                f.write(response.text)
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_samples = []
        for sample in data:
            text = sample["full_text"]
            spans = sample.get("spans", [])
            label = (
                "NO_PII"
                if not spans
                else max(
                    set(s["entity_type"] for s in spans),
                    key=[s["entity_type"] for s in spans].count,
                )
            )
            all_samples.append((text, label))
        return all_samples

    def _load_jailbreak_dataset(self):
        try:
            logger.info(
                "Loading jailbreak dataset from HuggingFace: jackhhao/jailbreak-classification"
            )
            jailbreak_dataset = load_dataset("jackhhao/jailbreak-classification")
            texts = list(jailbreak_dataset["train"]["prompt"]) + list(
                jailbreak_dataset["test"]["prompt"]
            )
            labels = list(jailbreak_dataset["train"]["type"]) + list(
                jailbreak_dataset["test"]["type"]
            )
            logger.info(f"Loaded {len(texts)} jailbreak samples")
            unique_labels = sorted(list(set(labels)))
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            logger.info(
                f"Jailbreak label distribution: {dict(sorted([(label, labels.count(label)) for label in set(labels)], key=lambda x: x[1], reverse=True))}"
            )
            logger.info(f"Jailbreak labels: {unique_labels}")
            label_indices = [label_to_idx[label] for label in labels]
            samples = list(zip(texts, label_indices))
            self.jailbreak_label_mapping = {
                "label_to_idx": label_to_idx,
                "idx_to_label": {idx: label for label, idx in label_to_idx.items()},
            }
            return samples
        except Exception as e:
            import traceback

            logger.error(f"Failed to load jailbreak dataset: {e}")
            traceback.print_exc()
            return []

    def _save_checkpoint(
        self, epoch, global_step, optimizer, scheduler, checkpoint_dir, label_mappings
    ):
        os.makedirs(checkpoint_dir, exist_ok=True)
        latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
        state = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "task_configs": self.task_configs,
            "label_mappings": label_mappings,
        }
        torch.save(state, latest_checkpoint_path)
        logger.info(
            f"Checkpoint saved for step {global_step} at {latest_checkpoint_path}"
        )

    def train(
        self,
        train_samples,
        val_samples,
        label_mappings,
        num_epochs=3,
        batch_size=16,
        learning_rate=2e-5,
        checkpoint_dir="checkpoints",
        resume=False,
        save_steps=500,
        checkpoint_to_load=None,
    ):
        train_dataset = MultitaskDataset(train_samples, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = MultitaskDataset(val_samples, self.tokenizer)
        val_loader = (
            DataLoader(val_dataset, batch_size=batch_size) if val_samples else None
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        start_epoch, global_step, steps_to_skip = 0, 0, 0

        if resume and checkpoint_to_load:
            logger.info("Loading state from checkpoint for training...")
            optimizer.load_state_dict(checkpoint_to_load["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint_to_load["scheduler_state_dict"])
            start_epoch = checkpoint_to_load["epoch"]
            global_step = checkpoint_to_load["global_step"]
            steps_in_epoch = len(train_loader)
            steps_to_skip = global_step % steps_in_epoch if steps_in_epoch > 0 else 0
            logger.info(
                f"Resuming from epoch {start_epoch}, global step {global_step}."
            )
            logger.info(
                f"Will skip the first {steps_to_skip} steps of epoch {start_epoch + 1}."
            )

        self.model.train()
        for epoch in range(start_epoch, num_epochs):
            pbar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch+1}",
            )
            for step, batch in pbar:
                if steps_to_skip > 0 and step < steps_to_skip:
                    continue

                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                )
                batch_loss = 0
                for i, task_name in enumerate(batch["task_name"]):
                    task_logits = outputs[task_name][i : i + 1]
                    task_label = batch["label"][i : i + 1].to(self.device)
                    task_weight = self.task_configs[task_name].get("weight", 1.0)
                    batch_loss += (
                        self.loss_fns[task_name](task_logits, task_label) * task_weight
                    )
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                global_step += 1
                if global_step > 0 and global_step % save_steps == 0:
                    self._save_checkpoint(
                        epoch,
                        global_step,
                        optimizer,
                        scheduler,
                        checkpoint_dir,
                        label_mappings,
                    )

            if val_loader:
                self.evaluate(val_loader)
            self._save_checkpoint(
                epoch + 1,
                global_step,
                optimizer,
                scheduler,
                checkpoint_dir,
                label_mappings,
            )
            steps_to_skip = 0

    def evaluate(self, val_loader):
        self.model.eval()
        task_correct, task_total = defaultdict(int), defaultdict(int)
        with torch.no_grad():
            for batch in val_loader:
                outputs = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                )
                for i, task_name in enumerate(batch["task_name"]):
                    predicted = torch.argmax(outputs[task_name][i : i + 1], dim=1)
                    task_correct[task_name] += (
                        (predicted == batch["label"][i : i + 1].to(self.device))
                        .sum()
                        .item()
                    )
                    task_total[task_name] += 1
        logger.info("Validation Results:")
        for task_name in task_correct:
            accuracy = task_correct[task_name] / task_total[task_name]
            logger.info(f"  {task_name} accuracy: {accuracy:.4f}")
        self.model.train()

    def save_model(self, output_path):
        os.makedirs(output_path, exist_ok=True)
        torch.save(
            self.model.state_dict(), os.path.join(output_path, "pytorch_model.bin")
        )
        self.tokenizer.save_pretrained(output_path)
        with open(os.path.join(output_path, "task_configs.json"), "w") as f:
            json.dump(self.task_configs, f, indent=2)
        model_config = {"base_model_name": self.model.bert.config.name_or_path}
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(model_config, f, indent=2)
        logger.info(f"Model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Multitask BERT Trainer")
    parser.add_argument(
        "--download-wiki", action="store_true", help="Run in data download mode."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./multitask_checkpoints",
        help="Dir to save/load checkpoints.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint.",
    )
    parser.add_argument(
        "--save-steps", type=int, default=500, help="Save a checkpoint every N steps."
    )
    args = parser.parse_args()

    base_model_name = "sentence-transformers/all-MiniLM-L12-v2"
    output_path = "./multitask_bert_model"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- FIX IS HERE ---
    if args.download_wiki:
        # -------------------
        temp_trainer = MultitaskTrainer(None, tokenizer, {}, device)
        temp_trainer.download_wikipedia_articles(
            HIGH_LEVEL_CATEGORIES, ARTICLES_PER_CATEGORY, WIKI_DATA_DIR
        )
        return

    logger.info("--- Starting Model Training ---")

    task_configs, label_mappings, checkpoint_to_load = {}, {}, None

    if args.resume:
        latest_checkpoint_path = os.path.join(
            args.checkpoint_dir, "latest_checkpoint.pt"
        )
        if os.path.exists(latest_checkpoint_path):
            logger.info(f"Resuming training from checkpoint: {latest_checkpoint_path}")
            checkpoint_to_load = torch.load(latest_checkpoint_path, map_location=device)

            task_configs = checkpoint_to_load.get("task_configs")
            label_mappings = checkpoint_to_load.get("label_mappings")

            if task_configs is None or label_mappings is None:
                logger.warning(
                    "Checkpoint is from an older version and is missing required model configuration."
                )
                logger.warning("Cannot safely resume. Starting a fresh training run.")
                args.resume = False
                checkpoint_to_load = None
                task_configs = {}
            else:
                logger.info("Loaded model configuration from checkpoint.")

        else:
            logger.warning(
                f"Resume flag is set, but no checkpoint found in '{args.checkpoint_dir}'. Starting fresh run."
            )
            args.resume = False

    if not args.resume:
        logger.info("Preparing datasets to discover model configuration...")
        temp_trainer = MultitaskTrainer(None, tokenizer, {}, device)
        _, _, label_mappings = temp_trainer.prepare_datasets()
        if "category" in label_mappings:
            task_configs["category"] = {
                "num_classes": len(
                    label_mappings["category"]["label_mapping"]["label_to_idx"]
                ),
                "weight": 1.5,
            }
        if "pii" in label_mappings:
            task_configs["pii"] = {
                "num_classes": len(
                    label_mappings["pii"]["label_mapping"]["label_to_idx"]
                ),
                "weight": 3.0,
            }
        if "jailbreak" in label_mappings:
            task_configs["jailbreak"] = {
                "num_classes": len(
                    label_mappings["jailbreak"]["label_mapping"]["label_to_idx"]
                ),
                "weight": 2.0,
            }

    if not task_configs:
        logger.error("No tasks configured. Exiting.")
        return

    logger.info(f"Final task configurations: {task_configs}")

    model = MultitaskBertModel(base_model_name, task_configs)

    if args.resume and checkpoint_to_load:
        model.load_state_dict(checkpoint_to_load["model_state_dict"])
        logger.info("Loaded model weights from checkpoint.")

    logger.info("Preparing datasets for training...")
    trainer_for_data = MultitaskTrainer(None, tokenizer, {}, device)
    if args.resume and label_mappings:
        trainer_for_data.jailbreak_label_mapping = label_mappings.get(
            "jailbreak", {}
        ).get("label_mapping")
    train_samples, val_samples, final_label_mappings = (
        trainer_for_data.prepare_datasets()
    )

    active_label_mappings = (
        label_mappings if (args.resume and label_mappings) else final_label_mappings
    )

    trainer = MultitaskTrainer(model, tokenizer, task_configs, device)

    logger.info(f"Total training samples: {len(train_samples)}")

    trainer.train(
        train_samples,
        val_samples,
        active_label_mappings,
        num_epochs=10,
        batch_size=16,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        save_steps=args.save_steps,
        checkpoint_to_load=checkpoint_to_load,
    )

    trainer.save_model(output_path)
    with open(os.path.join(output_path, "label_mappings.json"), "w") as f:
        json.dump(active_label_mappings, f, indent=2)
    logger.info("Multitask training completed!")


if __name__ == "__main__":
    main()
