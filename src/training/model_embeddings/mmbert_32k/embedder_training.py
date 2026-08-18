"""Narrow training orchestration for the historical mmBERT embedder recipe."""
# Derived from Model-training@3bc41e1322ee5a53e08d18eb940855dec53c1539.
# Upstream blob: 41e60e17ca960718dd1b71a23a86992128b9ed61.

from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)

from .embedder_data import convert_bge_to_triplets, load_bge_data_directory
from .embedder_evaluation import (
    create_evaluator,
    get_model_info,
    load_evaluation_data,
    test_layer_reduction,
)

logger = logging.getLogger(__name__)


class SelfDistillationLoss(torch.nn.Module):
    """Compatibility shell for the upstream self-distillation loss."""

    def __init__(
        self,
        model: SentenceTransformer,
        base_loss: torch.nn.Module,
        distill_weight: float = 0.5,
        temperature: float = 2.0,
    ):
        super().__init__()
        self.model = model
        self.base_loss = base_loss
        self.distill_weight = distill_weight
        self.temperature = temperature

    def forward(self, sentence_features, labels=None):
        """Delegate to the exact base loss, matching the imported implementation."""
        return self.base_loss(sentence_features, labels)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_output(args) -> None:
    if args.output_dir is None:
        model_short = args.model_name.split("/")[-1]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"./outputs/{model_short}-bge-style-{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Output directory: %s", args.output_dir)
    with open(os.path.join(args.output_dir, "training_args.json"), "w") as handle:
        json.dump(vars(args), handle, indent=2)


def _load_model(args) -> SentenceTransformer:
    logger.info("Loading model: %s", args.model_name)
    model_kwargs = {}
    if args.bf16:
        model_kwargs["dtype"] = torch.bfloat16
        logger.info("Loading model in bfloat16 for Flash Attention compatibility")
    elif args.fp16:
        model_kwargs["dtype"] = torch.float16
        logger.info("Loading model in float16 for Flash Attention compatibility")
    model = SentenceTransformer(
        args.model_name,
        revision=args.model_revision,
        trust_remote_code=True,
        model_kwargs=model_kwargs,
    )
    if args.max_seq_length:
        model.max_seq_length = args.max_seq_length
    info = get_model_info(model)
    logger.info("Model: %s parameters", f"{info['num_params']:,}")
    logger.info("Layers: %s (attr: %s)", info["num_layers"], info["layer_attr"])
    logger.info("Embedding dimension: %s", info["embedding_dim"])
    logger.info("Max sequence length: %s", model.max_seq_length)
    return model


def _load_training_dataset(args):
    datasets = []
    if args.train_data and os.path.isdir(args.train_data):
        logger.info("Loading BGE-format data from: %s", args.train_data)
        grouped = load_bge_data_directory(
            args.train_data,
            max_samples_per_file=args.max_samples_per_file,
            max_total_samples=args.max_samples,
        )
        datasets.append(convert_bge_to_triplets(grouped))
    if args.use_nli:
        logger.info("Loading AllNLI data...")
        nli = load_dataset(
            "sentence-transformers/all-nli",
            "triplet",
            split="train",
            revision=args.nli_revision,
        )
        if args.max_nli_samples and len(nli) > args.max_nli_samples:
            nli = nli.shuffle(seed=args.seed).select(range(args.max_nli_samples))
        datasets.append(nli)
        logger.info("NLI: %s samples", f"{len(nli):,}")
    if not datasets:
        raise ValueError("No training data! Specify --train_data or --use_nli")
    dataset = concatenate_datasets(datasets).shuffle(seed=args.seed)
    logger.info("\n%s", "=" * 60)
    logger.info("Total training samples: %s", f"{len(dataset):,}")
    logger.info("%s\n", "=" * 60)
    return dataset


def _build_loss(args, model):
    base_loss = losses.MultipleNegativesRankingLoss(model=model)
    dimensions = [int(value) for value in args.matryoshka_dims.split(",")]
    if args.use_adaptive_layer and args.use_matryoshka:
        logger.info("Using Matryoshka2dLoss (Adaptive Layers + Matryoshka)")
        logger.info("  - Matryoshka dims: %s", dimensions)
        logger.info("  - KL temperature: 0.0 (disabled for ModernBERT stability)")
        return losses.Matryoshka2dLoss(
            model=model,
            loss=base_loss,
            matryoshka_dims=dimensions,
            kl_temperature=0.0,
        )
    if args.use_adaptive_layer:
        logger.info("Using AdaptiveLayerLoss for layer reduction (KL disabled)")
        return losses.AdaptiveLayerLoss(model=model, loss=base_loss, kl_temperature=0.0)
    if args.use_matryoshka:
        logger.info("Using MatryoshkaLoss with dims: %s", dimensions)
        return losses.MatryoshkaLoss(
            model=model, loss=base_loss, matryoshka_dims=dimensions
        )
    if args.use_2d_matryoshka:
        logger.info("Using Matryoshka2dLoss with dims: %s (KL disabled)", dimensions)
        return losses.Matryoshka2dLoss(
            model=model,
            loss=base_loss,
            matryoshka_dims=dimensions,
            kl_temperature=0.0,
        )
    logger.info("Using MultipleNegativesRankingLoss")
    return base_loss


def _build_training_arguments(args) -> SentenceTransformerTrainingArguments:
    return SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16 and not args.bf16,
        bf16=args.bf16,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="sts-dev_spearman_cosine",
        greater_is_better=True,
        dataloader_num_workers=args.num_workers,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        data_seed=args.seed,
        max_grad_norm=1.0,
    )


def _evaluate_and_save(args, model, test_dataset) -> str:
    logger.info("\nEvaluating on STS test set...")
    results = create_evaluator(test_dataset, "sts-test")(model)
    logger.info("STS Test Results: %s", results)
    final_output_dir = os.path.join(args.output_dir, "final")
    model.save(final_output_dir)
    logger.info("Model saved to: %s", final_output_dir)
    with open(os.path.join(args.output_dir, "results.json"), "w") as handle:
        json.dump({"sts_test": results}, handle, indent=2)
    if args.use_adaptive_layer:
        test_layer_reduction(model)
    return final_output_dir


def _print_usage(final_output_dir: str) -> None:
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nModel saved to: {final_output_dir}")
    print("\nUsage:")
    print(
        f'''\nfrom sentence_transformers import SentenceTransformer

model = SentenceTransformer("{final_output_dir}")
embeddings = model.encode(["Hello world", "你好世界", "Hallo Welt"])

# For faster inference (with adaptive layers):
# model[0].auto_model.layers = model[0].auto_model.layers[:6]
'''
    )


def train(args):
    """Train the embedder with the imported recipe and artifact layout."""
    _seed_everything(args.seed)
    _prepare_output(args)
    model = _load_model(args)
    train_dataset = _load_training_dataset(args)
    train_loss = _build_loss(args, model)
    eval_dataset, test_dataset = load_evaluation_data(args.stsb_revision)
    trainer = SentenceTransformerTrainer(
        model=model,
        args=_build_training_arguments(args),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        evaluator=create_evaluator(eval_dataset, "sts-dev"),
    )
    logger.info("Starting training...")
    trainer.train()
    final_output_dir = _evaluate_and_save(args, model, test_dataset)
    _print_usage(final_output_dir)
    return model
