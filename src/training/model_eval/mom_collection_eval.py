"""Evaluate current Router classifiers, or explicitly select legacy MoM models.

The historical entrypoint name is retained for compatibility. Default model
identities follow served native artifacts; datasets are historical diagnostics,
not claims of independent Vela release qualification.
"""

import argparse
import json
import logging
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn as sns
import torch
from datasets import Dataset, load_dataset
from seqeval.metrics import accuracy_score as seqe_accuracy_score
from seqeval.metrics import classification_report as seq_classification_report
from seqeval.metrics import f1_score as seq_f1_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm import tqdm

try:
    from .constants import COLLECTIONS, LANGUAGE_CODES, MODEL_REGISTRY, model_registry
    from .dataset_contracts import classification_label_id, require_default_dataset
    from .model_loading import load_registered_model, tokenize_complete
except ImportError:
    from constants import COLLECTIONS, LANGUAGE_CODES, MODEL_REGISTRY, model_registry
    from dataset_contracts import classification_label_id, require_default_dataset
    from model_loading import load_registered_model, tokenize_complete
import warnings

# suppress prf warnings
warnings.filterwarnings(
    "ignore", category=UserWarning, module="sklearn.metrics._classification"
)

#  config logging


def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"evaluation_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger("MoMEval")


#                                                                     CORE Functions


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Served classifier evaluation (with explicit legacy MoM mode)"
    )
    parser.add_argument("--collection", choices=list(COLLECTIONS), default="served")
    parser.add_argument(
        "--revision", help="Explicit revision for --model_id or legacy artifacts"
    )
    parser.add_argument(
        "--dtype", choices=["float32", "bfloat16", "float16"], default="float32"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=32768,
        help="Full-input token budget; over-budget rows fail, never truncate",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model(s) to evaluate. Can specify multiple.",
    )
    parser.add_argument(
        "--model_id", type=str, default=None, help="Override default model path/repo."
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Evaluate LoRA variants instead of merged.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for inference."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit samples for quick testing."
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=list(LANGUAGE_CODES.keys()),
        help="Filter dataset by language.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="src/training/model_eval/results"
    )
    parser.add_argument(
        "--custom_dataset",
        type=str,
        default=None,
        help="Path to local .json/.csv file.",
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Evaluate multiple models in parallel."
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Max retries for network/loading operations.",
    )
    args = parser.parse_args(argv)
    if args.model_id and len(args.model) != 1:
        parser.error("--model_id requires exactly one --model")
    if args.max_length <= 0:
        parser.error("--max_length must be positive")
    registry = model_registry(args.collection)
    if args.use_lora and any("lora_id" not in registry[name] for name in args.model):
        parser.error(
            "Vela entries support merged evaluation only; use --collection legacy-mom for legacy adapters"
        )
    return args


def retry_operation(func, max_retries=3, delay=2):
    """Retry wrapper with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = delay * (2**attempt)
            logging.getLogger("MoMEval").warning(
                f"Attempt {attempt+1} failed: {e}. Retrying in {wait_time}s..."
            )
            time.sleep(wait_time)


def detect_language(text: str, target_lang: str) -> bool:
    """Detect if text matches target lang using char ranges."""
    if not text:
        return False

    #                  For Non-Latin Scripts
    char_checks = {
        "zh": lambda t: any("\u4e00" <= c <= "\u9fff" for c in t),
        "ja": lambda t: any("\u3040" <= c <= "\u30ff" for c in t),
        "ar": lambda t: any("\u0600" <= c <= "\u06ff" for c in t),
        "hi": lambda t: any("\u0900" <= c <= "\u097f" for c in t),
    }

    if target_lang in char_checks:
        return char_checks[target_lang](text)

    #  simple heuristic for latin

    return any("a" <= c.lower() <= "z" for c in text)


def filter_dataset_by_lang(dataset: Dataset, lang: str, text_col: str) -> Dataset:
    """Filter dataset by language."""
    logger = logging.getLogger("MoMEval")
    original_len = len(dataset)
    filtered = dataset.filter(lambda x: detect_language(x.get(text_col, ""), lang))
    logger.info(f"Language filter ({lang}): {original_len} -> {len(filtered)} samples")
    return filtered


def load_eval_data(model_name: str, args) -> Dataset:
    """Load eval dataset."""
    config = dict(model_registry(args.collection)[model_name])
    logger = logging.getLogger("MoMEval")

    try:
        # Custom Dataset
        if args.custom_dataset:
            logger.info(f"Loading custom dataset: {args.custom_dataset}")
            ext = Path(args.custom_dataset).suffix.lower()
            if ext not in [".json", ".csv"]:
                raise ValueError(f"Unsupported format: {ext}. Use .json or .csv")

            def load_custom_ds(split_name):
                return load_dataset(
                    "json" if ext == ".json" else "csv",
                    data_files=args.custom_dataset,
                    split=split_name,
                )

            ds = None
            for split in ["test", "validation", "eval", "train"]:
                try:
                    ds = retry_operation(
                        lambda split=split: load_custom_ds(split),
                        max_retries=args.max_retries,
                    )
                    logger.info(
                        f"Loaded custom dataset from '{split}' split ({len(ds)} samples"
                    )
                    if split == "train":
                        logger.warning(
                            "No test/val split found in custom dataset. "
                            "Evaluating on train split, results may be optimistic."
                        )
                    break
                except Exception as e:
                    logger.debug(
                        f"split '{split}' not found or failed: {e}. Trying next..."
                    )
            if ds is None:
                raise ValueError(
                    "Could not load any split from custom dataset. "
                    "Please ensure the file contains at least one of: test, validation, eval, or train split. "
                )
            if args.limit:
                ds = ds.select(range(min(len(ds), args.limit)))
            if config["type"] == "text_classification":
                ds = ds.map(
                    lambda row: {"label": classification_label_id(row["label"], config)}
                )
            return ds
        require_default_dataset(config)
        #  da  PII
        if model_name == "pii":
            logger.info("Fetching Presidio dataset...")
            url = "https://raw.githubusercontent.com/microsoft/presidio-research/master/data/synth_dataset_v2.json"

            def fetch_presidio():
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                return resp.json()

            data = retry_operation(fetch_presidio, max_retries=args.max_retries)
            if args.limit:
                data = data[: args.limit]

            processed = []
            for entry in tqdm(data, desc="Processing PII data"):
                text = entry["full_text"]
                spans = entry.get("spans", [])
                words = list(re.finditer(r"\S+", text))
                tokens = []
                labels = ["O"] * len(words)

                for idx, m in enumerate(words):
                    tokens.append(m.group())
                    for s in spans:
                        if (
                            m.start() >= s["start_position"]
                            and m.end() <= s["end_position"]
                        ):
                            prefix = "B-" if m.start() == s["start_position"] else "I-"
                            labels[idx] = f"{prefix}{s['entity_type']}"
                            break
                processed.append({"tokens": tokens, "labels": labels})
            return Dataset.from_list(processed)

        # Intent da
        if model_name == "intent":
            logger.info("Loading MMLU-Pro dataset...")
            ds = retry_operation(
                lambda: load_dataset(config["hf_dataset"], split=config["split"]),
                max_retries=args.max_retries,
            )
            ds = ds.filter(lambda x: x["category"] in config["labels"])
            if args.limit:
                ds = ds.select(range(min(len(ds), args.limit)))
            l2id = {label: i for i, label in enumerate(config["labels"])}
            ds = ds.map(lambda x: {"label": l2id[x["category"]]})
            ds = ds.rename_column("question", "text")
            return ds

        #                 Standard Classification
        logger.info(f"Loading {config['hf_dataset']}...")
        ds = retry_operation(
            lambda: load_dataset(config["hf_dataset"], split=config["split"]),
            max_retries=args.max_retries,
        )

        if config["text_col"] not in ds.column_names:
            possible_text_cols = [
                "text",
                "question",
                "input",
                "prompt",
                "message",
                "full_text",
            ]
            for col in possible_text_cols:
                if col in ds.column_names:
                    logger.info(f"Auto-detected text column '{col}' for {model_name}")
                    config["text_col"] = col
                    break
            else:
                raise ValueError(
                    f"No suitable text column found in dataset for {model_name}. Columns: {ds.column_names}"
                )

        if config["label_col"] not in ds.column_names:
            possible_label_cols = ["label_id", "label_name", "label_text", "label"]
            for col in possible_label_cols:
                if col in ds.column_names:
                    logger.info(f"Auto-detected label column '{col}' for {model_name}")
                    config["label_col"] = col
                    break
            else:
                raise ValueError(
                    f"No suitable label column found in dataset for {model_name}. Columns: {ds.column_names}"
                )

        if args.language:
            ds = filter_dataset_by_lang(ds, args.language, config["text_col"])

        if args.limit:
            ds = ds.select(range(min(len(ds), args.limit)))

        #                   Renaming columns
        if config["text_col"] in ds.column_names and config["text_col"] != "text":
            if "text" in ds.column_names:
                ds = ds.remove_columns(["text"])
            ds = ds.rename_column(config["text_col"], "text")
        if config["label_col"] in ds.column_names and config["label_col"] != "label":
            if "label" in ds.column_names:
                ds = ds.remove_columns(["label"])
            ds = ds.rename_column(config["label_col"], "label")

        def map_to_int(example):
            example["label"] = classification_label_id(example["label"], config)
            return example

        ds = ds.map(map_to_int)

        logger.info(f"Dataset columns after processing: {ds.column_names}")

        return ds

    except Exception as e:
        logger.error(f"Failed to load dataset for {model_name}: {e}")
        raise


def load_model_and_tokenizer(model_name: str, args):
    """Load the exact registered artifact without replacing its configuration."""
    return load_registered_model(model_registry(args.collection)[model_name], args)


def batch_row_count(batch: dict) -> int:
    """Return the number of rows in a slice of a HuggingFace ``Dataset``.

    Slicing a ``Dataset`` gives back a dict of column name -> list of values,
    not a list of rows, so ``len(batch)`` is the column count. Every column
    carries one entry per row, so any column's length is the row count. This
    also gives the right answer for a short final batch.
    """
    return len(next(iter(batch.values())))


def evaluate_single_model(model_name: str, args) -> tuple[str, dict]:
    """Evaluate a single model with comprehensive error handling."""
    config = model_registry(args.collection)[model_name]
    logger = logging.getLogger("MoMEval")

    try:
        #                           Load data and model
        dataset = load_eval_data(model_name, args)
        model, tokenizer = load_model_and_tokenizer(model_name, args)
        if args.max_length > model.config.max_position_embeddings:
            raise ValueError("--max_length exceeds this artifact's context capacity")

        all_preds, all_truths, lats = [], [], []

        #                          Inference Loop
        for i in tqdm(
            range(0, len(dataset), args.batch_size), desc=f"Eval {model_name}"
        ):
            batch = dataset[i : i + args.batch_size]
            start = time.time()

            try:
                if config["type"] == "text_classification":
                    inputs = tokenize_complete(
                        tokenizer,
                        batch["text"],
                        args.max_length,
                        args.device,
                        return_tensors="pt",
                    )
                    with torch.no_grad():
                        out = model(**inputs)
                        preds = torch.argmax(out.logits, dim=-1).cpu().tolist()
                    all_preds.extend(preds)
                    all_truths.extend(batch["label"])
                else:
                    inputs = tokenize_complete(
                        tokenizer,
                        batch["tokens"],
                        args.max_length,
                        args.device,
                        is_split_into_words=True,
                        return_tensors="pt",
                    )

                    with torch.no_grad():
                        out = model(**inputs)
                        pred_ids = torch.argmax(out.logits, dim=-1).cpu().tolist()

                    for idx, (p_seq, true_labels) in enumerate(
                        zip(pred_ids, batch["labels"], strict=False)
                    ):
                        word_ids = inputs.word_ids(batch_index=idx)
                        p_labels = []

                        for j, word_id in enumerate(word_ids):
                            if word_id is None:
                                continue
                            #          Take prediction only for the **first** subword
                            if j == 0 or word_ids[j - 1] != word_id:
                                label_idx = p_seq[j]
                                if 0 <= label_idx < len(config["labels"]):
                                    p_labels.append(config["labels"][label_idx])
                                else:
                                    p_labels.append("O")

                        if len(p_labels) != len(true_labels):
                            raise ValueError(
                                f"Word-label alignment mismatch sample {i+idx}: "
                                f"pred={len(p_labels)}, gold={len(true_labels)}"
                            )
                        all_preds.append(p_labels)
                        all_truths.append(true_labels)

                batch_time = (time.time() - start) * 1000
                rows = batch_row_count(batch)
                lats.extend([batch_time / rows] * rows)

            except torch.cuda.OutOfMemoryError:
                logger.error(
                    f"OOM at batch {i}. Reduce --batch_size; this run is incomplete."
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise
            except Exception as e:
                raise RuntimeError(f"Evaluation failed at batch {i}") from e

        # Validate we got predictions
        if not all_preds or not all_truths:
            raise ValueError(
                f"No valid predictions for {model_name}. Check dataset and model compatibility."
            )

        # Metrics Calculation
        stats = {
            "artifact": model.evaluation_identity,
            "collection": args.collection,
            "max_length": args.max_length,
            "inference_policy": "native-full-input-argmax",
            "dataset_scope": (
                "custom" if args.custom_dataset else "historical-source-diagnostic"
            ),
            "latency": {
                "avg_ms": float(np.mean(lats)),
                "p50_ms": float(np.percentile(lats, 50)),
                "p99_ms": float(np.percentile(lats, 99)),
            },
        }

        if config["type"] == "text_classification":
            unique_labels = list(range(len(config["labels"])))
            stats["label_support"] = {
                label: all_truths.count(index)
                for index, label in enumerate(config["labels"])
            }
            stats["unsupported_labels"] = [
                label for label, count in stats["label_support"].items() if count == 0
            ]
            stats["macro_f1"] = float(
                precision_recall_fscore_support(
                    all_truths,
                    all_preds,
                    labels=unique_labels,
                    average="macro",
                    zero_division=0,
                )[2]
            )
            stats["accuracy"] = float(accuracy_score(all_truths, all_preds))
            p, r, f1, _sup = precision_recall_fscore_support(
                all_truths,
                all_preds,
                labels=unique_labels,
                average="weighted",
                zero_division=0,
            )
            stats.update({"precision": float(p), "recall": float(r), "f1": float(f1)})
            stats["cm"] = (
                confusion_matrix(
                    all_truths, all_preds, labels=range(len(config["labels"]))
                )
                .astype(int)
                .tolist()
            )
            stats["report"] = classification_report(
                all_truths,
                all_preds,
                target_names=config["labels"],
                labels=range(len(config["labels"])),
                output_dict=True,
                zero_division=0,
            )
        else:
            # Align sequence lengths
            aligned_t, aligned_p = [], []
            for t, p in zip(all_truths, all_preds, strict=False):
                m_len = min(len(t), len(p))
                aligned_t.append(t[:m_len])
                aligned_p.append(p[:m_len])

            stats["accuracy"] = float(seqe_accuracy_score(aligned_t, aligned_p))
            stats["f1"] = float(seq_f1_score(aligned_t, aligned_p))
            stats["report"] = seq_classification_report(
                aligned_t, aligned_p, output_dict=True, zero_division=0
            )

            # Per-class metrics
            per_class = seq_classification_report(
                aligned_t, aligned_p, output_dict=True, zero_division=0
            )
            stats["per_class"] = {
                entity: {
                    "precision": float(per_class[entity]["precision"]),
                    "recall": float(per_class[entity]["recall"]),
                    "f1": float(per_class[entity]["f1-score"]),
                    "support": int(per_class[entity]["support"]),
                }
                for entity in per_class
                if entity not in ["accuracy", "macro avg", "weighted avg"]
            }

        #                                Save Results
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        def default_converter(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(
                f"Object of type {obj.__class__.__name__} is not JSON serializable"
            )

        with open(out_dir / f"{model_name}_results.json", "w") as f:
            json.dump(stats, f, indent=2, default=default_converter)

        # Save conf mat if available
        if "cm" in stats:
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                stats["cm"],
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=config["labels"],
                yticklabels=config["labels"],
            )
            plt.title(f"Confusion Matrix: {model_name.upper()}")
            plt.tight_layout()
            plt.savefig(out_dir / f"{model_name}_cm.png")
            plt.close()

        #  cleanup the memoryyyy

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"{model_name} evaluation complete")
        return model_name, stats

    except Exception as e:
        logger.error(f"{model_name} evaluation failed: {e}")
        return model_name, {"error": str(e)}


def main():
    args = parse_args()
    logger = setup_logging(Path(args.output_dir))
    if args.parallel and "cuda" in args.device:
        logger.warning(
            "Parallel execution on CUDA is unstable/OOM-prone. Forcing device='cpu' for workers."
        )
        args.device = "cpu"
    logger.info(f"Starting evaluation with args: {vars(args)}")

    models = args.model if isinstance(args.model, list) else [args.model]
    summary = {}

    #                                   || or seq ex
    if args.parallel and len(models) > 1:
        logger.info(f"Launching parallel evaluation for {len(models)} models...")
        with ProcessPoolExecutor(max_workers=min(len(models), os.cpu_count())) as ex:
            futures = [ex.submit(evaluate_single_model, m, args) for m in models]
            for f in as_completed(futures):
                try:
                    name, res = f.result()
                    summary[name] = res
                except Exception as e:
                    logger.error(f"Parallel task failed: {e}")
                    summary[models[futures.index(f)]] = {"error": str(e)}
    else:
        for m in models:
            name, res = evaluate_single_model(m, args)
            summary[name] = res

    #  On console summary
    print("\n" + "=" * 80)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 80)
    print(
        f"{'Model':<15} | {'Accuracy':<10} | {'F1 Score':<10} | {'Latency (p50)':<15}"
    )
    print("-" * 80)

    for name, res in summary.items():
        if "error" in res:
            print(f"{name.upper():<15} | {'ERROR':<10} | {'ERROR':<10} | {'ERROR':<15}")
        else:
            acc = res.get("accuracy", 0)
            f1 = res.get("f1", 0)
            lat = res["latency"]["p50_ms"]
            print(f"{name.upper():<15} | {acc:<10.4f} | {f1:<10.4f} | {lat:<15.2f}ms")

    print("=" * 80 + "\n")
    logger.info("Evaluation pipeline complete!")
    missing = set(models) - set(summary)
    for name in missing:
        summary[name] = {"error": "Evaluation produced no result"}
    Path(args.output_dir, "summary.json").write_text(json.dumps(summary, indent=2))
    if missing or any("error" in result for result in summary.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
