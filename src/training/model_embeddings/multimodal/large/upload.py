#!/usr/bin/env python3

import argparse
import configparser
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent
HTTP_FORBIDDEN = 403


def read_token(token: str | None, token_name: str) -> str:
    if token:
        return token

    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        return env_token

    token_store = Path.home() / ".cache" / "huggingface" / "stored_tokens"
    parser = configparser.ConfigParser()
    parser.read(token_store)
    if token_name in parser and "hf_token" in parser[token_name]:
        return parser[token_name]["hf_token"].strip()

    raise ValueError(
        "No Hugging Face token found. Pass --token, set HF_TOKEN, or store it in ~/.cache/huggingface/stored_tokens."
    )


def evaluate_final_export(final_dir: str, max_samples: int | None) -> dict[str, float]:
    from .evaluate import (  # noqa: PLC0415 - optional runtime dependency
        build_eval_loader,
        load_config,
        load_final_weights,
    )
    from .runtime import (  # noqa: PLC0415 - optional runtime dependency
        normalize_mixed_precision,
    )
    from .tri_encoder import (  # noqa: PLC0415 - optional runtime dependency
        build_datacenter_tri_encoder_model,
        evaluate_tri_encoder_model,
    )

    config_path = os.path.join(final_dir, "config.json")
    cfg = load_config(config_path)
    model = build_datacenter_tri_encoder_model(cfg)
    load_final_weights(model, final_dir)

    from accelerate import Accelerator  # noqa: PLC0415 - optional runtime dependency

    accelerator = Accelerator(
        mixed_precision=normalize_mixed_precision(
            cfg.get("training", {}).get("mixed_precision", "bf16")
        )
    )
    eval_loader = build_eval_loader(cfg, max_samples=max_samples)
    model, eval_loader = accelerator.prepare(model, eval_loader)

    metrics = evaluate_tri_encoder_model(
        model,
        eval_loader,
        accelerator,
        float(cfg.get("loss", {}).get("scale", 20.0)),
    )
    accelerator.wait_for_everyone()
    return metrics


def render_model_card(
    template_path: str, repo_id: str, cfg: dict[str, Any], metrics: dict[str, float]
) -> str:
    with open(template_path, encoding="utf-8") as handle:
        template = handle.read()

    model_cfg = cfg["model"]
    replacements = {
        "__MODEL_NAME__": repo_id.rsplit("/", maxsplit=1)[-1],
        "__REPO_ID__": repo_id,
        "__TEXT_ENCODER_NAME__": str(model_cfg["text_encoder_name"]),
        "__IMAGE_ENCODER_NAME__": str(model_cfg["image_encoder_name"]),
        "__AUDIO_ENCODER_NAME__": str(model_cfg["audio_encoder_name"]),
        "__EMBEDDING_DIM__": str(model_cfg["embedding_dim"]),
        "__MAX_TEXT_LENGTH__": str(model_cfg["max_text_length"]),
        "__EVAL_LOSS__": f"{metrics.get('eval_loss', float('nan')):.6f}",
        "__EVAL_TOP1__": f"{metrics.get('eval_top1', float('nan')):.6f}",
    }

    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    return rendered


def prepare_upload_bundle(
    final_dir: str, repo_id: str, metrics: dict[str, float], output_dir: str
) -> str:
    from .evaluate import load_config  # noqa: PLC0415 - optional runtime dependency

    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(final_dir, "config.json")
    model_path = os.path.join(final_dir, "model.pt")
    cfg = load_config(config_path)

    shutil.copy2(model_path, os.path.join(output_dir, "model.pt"))
    shutil.copy2(config_path, os.path.join(output_dir, "config.json"))

    template_path = str(PACKAGE_ROOT / "model_card_template.md")
    readme = render_model_card(template_path, repo_id, cfg, metrics)
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as handle:
        handle.write(readme)

    package_dir = Path(output_dir) / "src" / "hf_st_mm"
    if package_dir.parent.exists():
        shutil.rmtree(package_dir.parent)
    package_dir.mkdir(parents=True)
    shutil.copy2(PACKAGE_ROOT / "model.py", package_dir / "model.py")
    shutil.copy2(PACKAGE_ROOT / "artifact_data.py", package_dir / "data.py")
    (package_dir / "__init__.py").write_text(
        '"""Packaged multimodal-large inference source."""\n', encoding="utf-8"
    )

    return output_dir


def upload_folder(
    folder_path: str, repo_id: str, token: str, commit_message: str
) -> None:
    try:
        from huggingface_hub import (  # noqa: PLC0415 - optional upload dependency
            HfApi,
            create_repo,
        )
        from huggingface_hub.errors import (  # noqa: PLC0415 - optional upload dependency
            HfHubHTTPError,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Upload dependencies are missing; install this package's requirements.txt"
        ) from exc

    api = HfApi()
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True, token=token)
    except HfHubHTTPError as exc:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        if status_code != HTTP_FORBIDDEN:
            raise

        try:
            api.repo_info(repo_id=repo_id, repo_type="model", token=token)
        except Exception as repo_exc:
            raise PermissionError(
                f"Token cannot create or access model repo '{repo_id}'. "
                "Pre-create the repo under that namespace or use a token with the required org permissions."
            ) from repo_exc

    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
        token=token,
        commit_message=commit_message,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload the final tri-encoder export to Hugging Face Hub"
    )
    parser.add_argument(
        "--final-dir",
        required=True,
        help="Final export directory containing model.pt and config.json",
    )
    parser.add_argument(
        "--repo-id",
        default="llm-semantic-router/multi-modal-embed-large",
        help="Destination Hugging Face model repository",
    )
    parser.add_argument("--token", default=None, help="Hugging Face token")
    parser.add_argument(
        "--token-name",
        default="model training",
        help="Entry name in ~/.cache/huggingface/stored_tokens when --token is not provided",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optionally evaluate only the first N validation samples",
    )
    parser.add_argument(
        "--min-eval-top1",
        type=float,
        default=None,
        help="Optional minimum eval_top1 required before upload",
    )
    parser.add_argument(
        "--max-eval-loss",
        type=float,
        default=None,
        help="Optional maximum eval_loss allowed before upload",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Upload without running final evaluation",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Prepare the upload bundle but do not upload it",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the prepared upload bundle. Defaults to a temporary directory.",
    )
    return parser


def validate_final_dir(final_dir: str) -> None:
    required = ("model.pt", "config.json")
    missing = [
        name for name in required if not os.path.exists(os.path.join(final_dir, name))
    ]
    if missing:
        raise FileNotFoundError(
            f"Final export is missing {', '.join(missing)} under {final_dir}"
        )


def validate_metrics(metrics: dict[str, float], args: argparse.Namespace) -> None:
    if (
        args.min_eval_top1 is not None
        and metrics.get("eval_top1", float("-inf")) < args.min_eval_top1
    ):
        raise ValueError(
            f"Refusing upload because eval_top1={metrics.get('eval_top1')} "
            f"is below required {args.min_eval_top1}"
        )
    if (
        args.max_eval_loss is not None
        and metrics.get("eval_loss", float("inf")) > args.max_eval_loss
    ):
        raise ValueError(
            f"Refusing upload because eval_loss={metrics.get('eval_loss')} "
            f"exceeds allowed {args.max_eval_loss}"
        )


def print_status(status: str, metrics: dict[str, float], **fields: Any) -> None:
    print(json.dumps({status: True, **fields, **metrics}, indent=2, sort_keys=True))


def main() -> None:
    args = build_parser().parse_args()

    final_dir = os.path.abspath(args.final_dir)
    validate_final_dir(final_dir)

    metrics: dict[str, float] = {}
    if not args.skip_eval:
        metrics = evaluate_final_export(final_dir, args.max_samples)
        validate_metrics(metrics, args)

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.abspath(args.output_dir) if args.output_dir else temp_dir
        bundle_dir = prepare_upload_bundle(final_dir, args.repo_id, metrics, output_dir)
        print_status("prepared", metrics, bundle_dir=bundle_dir)

        if args.skip_upload:
            return

        token = read_token(args.token, args.token_name)
        upload_folder(
            bundle_dir,
            args.repo_id,
            token,
            commit_message=f"Upload {args.repo_id.split('/')[-1]} final model",
        )
        print_status("uploaded", metrics, repo_id=args.repo_id)


if __name__ == "__main__":
    main()
