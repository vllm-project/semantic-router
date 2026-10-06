"""Train the large multimodal model through native or tri-encoder paths."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from .runtime import (
    configure_unbuffered_output,
    is_datacenter_tri_encoder_config,
    load_yaml,
    log_progress,
    require_sentence_transformers_version,
    set_seed,
    write_status,
)


def _log_dataset_info(
    train_info: dict[str, Any], eval_info: dict[str, Any] | None
) -> None:
    log_progress(
        f"[startup] loaded train dataset with {train_info['num_rows']} rows "
        f"and modalities {train_info['modalities']}"
    )
    if eval_info is not None:
        log_progress(
            f"[startup] loaded eval dataset with {eval_info['num_rows']} rows "
            f"and modalities {eval_info['modalities']}"
        )


def _native_training(cfg: dict[str, Any], resume: str | None, version: str) -> None:
    from .native_training import (  # noqa: PLC0415 - optional runtime dependency
        NativeSentenceTransformerTrainer,
        build_data_collator,
        build_evaluator,
        build_loss,
        build_model,
        build_training_args,
        load_sentence_transformers_datasets,
        resolve_training_max_steps,
        validate_modalities,
    )

    log_progress("[startup] loading datasets")
    train_dataset, train_info, eval_dataset, eval_info = (
        load_sentence_transformers_datasets(cfg)
    )
    _log_dataset_info(train_info, eval_info)

    log_progress(f"[startup] loading model {cfg['model']['model_name']}")
    model = build_model(cfg)
    validate_modalities(model, train_info)
    if eval_info is not None:
        validate_modalities(model, eval_info)

    if train_info["num_negatives_present"] and not train_info["has_uniform_negatives"]:
        print(
            "Training manifest has mixed negative availability; dropping negative_0 "
            "and training on pairs only.",
            file=sys.stderr,
        )
    resolved_steps = resolve_training_max_steps(cfg, train_info)
    if resolved_steps is not None:
        cfg.setdefault("training", {})["max_steps"] = resolved_steps
        log_progress(
            f"[startup] resolved max_steps={resolved_steps} for iterable training"
        )

    trainer = NativeSentenceTransformerTrainer(
        model=model,
        args=build_training_args(cfg),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=build_loss(model, cfg),
        evaluator=build_evaluator(eval_dataset, cfg),
        data_collator=build_data_collator(model),
    )
    log_progress("[startup] starting native Sentence Transformers training")
    result = trainer.train(resume_from_checkpoint=resume or None)
    trainer.save_model(os.path.join(cfg["output_dir"], "final"))

    metrics = dict(result.metrics)
    metrics.update(
        sentence_transformers_version=version,
        train_rows=train_info["num_rows"],
        train_modalities=train_info["modalities"],
    )
    if eval_info is not None:
        metrics.update(
            eval_rows=eval_info["num_rows"],
            eval_modalities=eval_info["modalities"],
        )
    write_status(cfg["output_dir"], metrics)
    log_progress(
        f"[done] wrote status to {os.path.join(cfg['output_dir'], 'train_status.json')}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    return parser


def main() -> None:
    configure_unbuffered_output()
    args = build_parser().parse_args()
    log_progress(f"[startup] loading config from {args.config}")
    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))
    version = require_sentence_transformers_version()
    log_progress(f"[startup] sentence-transformers version {version}")

    if is_datacenter_tri_encoder_config(cfg):
        from .tri_encoder_training import (  # noqa: PLC0415 - optional runtime dependency
            run_datacenter_tri_encoder_training,
        )

        log_progress("[startup] detected datacenter tri-encoder config")
        run_datacenter_tri_encoder_training(cfg, args.resume, version)
    else:
        _native_training(cfg, args.resume, version)


if __name__ == "__main__":
    main()
