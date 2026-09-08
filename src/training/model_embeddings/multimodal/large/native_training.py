"""Native Sentence Transformers training path for multimodal-large."""

import math
import os
from contextlib import nullcontext
from io import BytesIO
from typing import Any

import torch
from datasets import IterableDataset as HFIterableDataset
from PIL import Image

from .data import manifest_to_sentence_transformers_dataset
from .runtime import normalize_mixed_precision

try:
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
except ImportError:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.trainer import SentenceTransformerTrainer
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments

try:
    from sentence_transformers.sentence_transformer.evaluation import (
        InformationRetrievalEvaluator,
    )
except ImportError:
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

try:
    from sentence_transformers.sentence_transformer.losses import (
        CachedMultipleNegativesRankingLoss,
        MatryoshkaLoss,
    )
except ImportError:
    from sentence_transformers.losses import (
        CachedMultipleNegativesRankingLoss,
        MatryoshkaLoss,
    )

try:
    from sentence_transformers.sentence_transformer.losses.cached_multiple_negatives_ranking import (
        RandContext,
        _create_minibatch,
    )
except ImportError:
    from sentence_transformers.sentence_transformer.losses.cached_multiple_negatives_ranking import (
        RandContext,
        _create_minibatch,
    )

try:
    from sentence_transformers.sentence_transformer.training_args import BatchSamplers
except ImportError:
    from sentence_transformers.training_args import BatchSamplers

try:
    from sentence_transformers.base.data_collator import BaseDataCollator
except ImportError:
    from sentence_transformers.base.data_collator import BaseDataCollator


class NativeSentenceTransformerTrainer(SentenceTransformerTrainer):
    def add_model_card_callback(self, default_args_dict: dict[str, Any]) -> None:
        # Sentence Transformers eagerly inspects dataset examples for model-card metadata.
        # In this container, that path trips over torchvision JPEG decoding before training starts.
        # Skipping the callback keeps training functional and does not affect optimization.
        return

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError(
                f"Training requires specifying a train_dataset to the {self.__class__.__name__}."
            )

        self.accelerator.even_batches = False
        if isinstance(self.train_dataset, HFIterableDataset):
            self.accelerator.dataloader_config.dispatch_batches = False
            self.accelerator.dataloader_config.split_batches = False

        self._train_dataloader = self.accelerator.prepare(
            self._build_dataloader(
                self.train_dataset, self.args.train_batch_size, dataset_kind="train"
            )
        )
        return self._train_dataloader


class NativeCachedMultipleNegativesRankingLoss(CachedMultipleNegativesRankingLoss):
    @staticmethod
    def _prune_empty_multimodal_tensors(
        sentence_feature_minibatch: dict[str, Any],
    ) -> dict[str, Any]:
        pruned = dict(sentence_feature_minibatch)

        for grid_key, pixel_key, count_key in (
            ("image_grid_thw", "pixel_values", "num_images_per_sample"),
            ("video_grid_thw", "pixel_values_videos", "num_videos_per_sample"),
        ):
            grid = pruned.get(grid_key)
            if getattr(grid, "numel", lambda: 0)() == 0:
                pruned.pop(grid_key, None)
                pruned.pop(pixel_key, None)
                pruned.pop(count_key, None)

        return pruned

    def embed_minibatch(
        self,
        sentence_feature: dict[str, torch.Tensor],
        begin: int,
        end: int,
        with_grad: bool,
        copy_random_state: bool,
        random_state: RandContext | None = None,
    ):
        grad_context = nullcontext if with_grad else torch.no_grad
        random_state_context = nullcontext() if random_state is None else random_state
        sentence_feature_minibatch = _create_minibatch(sentence_feature, begin, end)
        sentence_feature_minibatch = self._prune_empty_multimodal_tensors(
            sentence_feature_minibatch
        )
        with random_state_context, grad_context():
            random_state = (
                RandContext(*sentence_feature_minibatch.values())
                if copy_random_state
                else None
            )
            reps = self.model(sentence_feature_minibatch)["sentence_embedding"]
        return reps, random_state


def build_model(cfg: dict[str, Any]) -> SentenceTransformer:
    model_cfg = cfg["model"]
    model_name = model_cfg.get("model_name")
    if not model_name:
        raise ValueError(
            "Native Sentence Transformers multimodal training requires model.model_name. "
            "The previous text/image/audio encoder triplet config is not compatible with this path."
        )

    model_kwargs = dict(model_cfg.get("model_kwargs", {}))
    processor_kwargs = dict(model_cfg.get("processor_kwargs", {}))
    if bool(model_cfg.get("trust_remote_code", False)):
        model_kwargs.setdefault("trust_remote_code", True)
        processor_kwargs.setdefault("trust_remote_code", True)

    model = SentenceTransformer(
        model_name,
        model_kwargs=model_kwargs or None,
        processor_kwargs=processor_kwargs or None,
        truncate_dim=model_cfg.get("truncate_dim"),
    )

    max_text_length = model_cfg.get("max_text_length")
    if max_text_length:
        model.max_seq_length = int(max_text_length)

    return model


def resolve_supported_modalities(model: SentenceTransformer) -> set[str]:
    modalities = getattr(model, "modalities", None)
    if modalities:
        return {str(item) for item in modalities}

    supported = set()
    supports = getattr(model, "supports", None)
    if callable(supports):
        for modality in ("text", "image", "audio", "video"):
            try:
                if supports(modality):
                    supported.add(modality)
            except Exception:
                continue
    return supported


def build_loss(model: SentenceTransformer, cfg: dict[str, Any]):
    loss_cfg = cfg.get("loss", {})
    loss_type = loss_cfg.get("type", "cached_mnrl")
    if loss_type != "cached_mnrl":
        raise ValueError(f"Unsupported loss type for native trainer path: {loss_type}")

    loss = NativeCachedMultipleNegativesRankingLoss(
        model,
        scale=float(loss_cfg.get("scale", 20.0)),
        mini_batch_size=int(loss_cfg.get("mini_batch_size", 1)),
    )

    matryoshka_dims = loss_cfg.get("matryoshka_dims") or []
    if matryoshka_dims:
        loss = MatryoshkaLoss(
            model, loss, matryoshka_dims=[int(dim) for dim in matryoshka_dims]
        )
    return loss


def build_evaluator(
    eval_dataset, cfg: dict[str, Any]
) -> InformationRetrievalEvaluator | None:
    if eval_dataset is None:
        return None

    eval_rows = list(eval_dataset)
    if not eval_rows:
        return None

    queries = {
        idx: normalize_st_input(row["query"]) for idx, row in enumerate(eval_rows)
    }
    corpus = {
        idx: normalize_st_input(row["positive"]) for idx, row in enumerate(eval_rows)
    }
    relevant_docs = {idx: [idx] for idx in range(len(eval_rows))}

    if "negative_0" in eval_dataset.column_names:
        offset = len(eval_rows)
        for idx, row in enumerate(eval_rows):
            corpus[offset + idx] = normalize_st_input(row["negative_0"])

    validation_cfg = cfg.get("validation", {})
    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        batch_size=int(validation_cfg.get("batch_size", 1)),
        show_progress_bar=True,
        name=validation_cfg.get("name", "multimodal-ir"),
    )


def build_training_args(cfg: dict[str, Any]) -> SentenceTransformerTrainingArguments:
    training_cfg = cfg["training"]
    use_iterable_dataset = bool(cfg.get("data", {}).get("use_iterable_dataset", False))
    mixed_precision = normalize_mixed_precision(
        training_cfg.get("mixed_precision", "bf16")
    )
    eval_enabled = bool(cfg.get("validation", {}).get("manifest_path"))
    max_steps = training_cfg.get("max_steps")
    warmup_steps = training_cfg.get("warmup_steps")
    if warmup_steps is None:
        warmup_steps = float(training_cfg.get("warmup_ratio", 0.0))

    args = SentenceTransformerTrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=float(training_cfg["epochs"]),
        max_steps=int(max_steps) if max_steps is not None else -1,
        per_device_train_batch_size=int(training_cfg["batch_size"]),
        per_device_eval_batch_size=int(
            training_cfg.get("eval_batch_size", training_cfg["batch_size"])
        ),
        gradient_accumulation_steps=int(training_cfg.get("grad_accum_steps", 1)),
        learning_rate=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
        warmup_steps=float(warmup_steps),
        max_grad_norm=float(training_cfg.get("max_grad_norm", 1.0)),
        dataloader_num_workers=int(training_cfg.get("num_workers", 4)),
        dataloader_drop_last=bool(training_cfg.get("drop_last", True)),
        batch_sampler=(
            BatchSamplers.BATCH_SAMPLER
            if use_iterable_dataset
            else BatchSamplers.NO_DUPLICATES
        ),
        accelerator_config=(
            {"dispatch_batches": False} if use_iterable_dataset else None
        ),
        save_strategy="steps",
        save_steps=int(training_cfg.get("save_every", 1000)),
        save_total_limit=int(training_cfg.get("save_total_limit", 2)),
        logging_strategy="steps",
        logging_steps=int(training_cfg.get("log_every", 10)),
        logging_first_step=True,
        eval_strategy="steps" if eval_enabled else "no",
        eval_steps=(
            int(training_cfg.get("eval_every", training_cfg.get("save_every", 1000)))
            if eval_enabled
            else None
        ),
        fp16=mixed_precision == "fp16",
        bf16=mixed_precision == "bf16",
        seed=int(cfg.get("seed", 42)),
        remove_unused_columns=False,
        run_name=training_cfg.get("run_name"),
        disable_tqdm=False,
    )
    return args


def resolve_training_max_steps(
    cfg: dict[str, Any], train_info: dict[str, Any]
) -> int | None:
    training_cfg = cfg["training"]
    explicit_max_steps = training_cfg.get("max_steps")
    if explicit_max_steps is not None:
        return int(explicit_max_steps)

    if not bool(cfg.get("data", {}).get("use_iterable_dataset", False)):
        return None

    world_size = int(
        os.environ.get("WORLD_SIZE") or os.environ.get("ACCELERATE_NUM_PROCESSES") or 1
    )
    world_size = max(world_size, 1)
    per_device_batch_size = int(training_cfg["batch_size"])
    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    drop_last = bool(training_cfg.get("drop_last", True))
    global_micro_batch = max(per_device_batch_size * world_size, 1)
    train_rows = int(train_info["num_rows"])

    if drop_last:
        micro_steps_per_epoch = train_rows // global_micro_batch
    else:
        micro_steps_per_epoch = math.ceil(train_rows / global_micro_batch)

    if micro_steps_per_epoch <= 0:
        raise ValueError(
            "Computed zero training steps per epoch; dataset is too small for the configured global batch size."
        )

    optimizer_steps_per_epoch = max(
        1, math.ceil(micro_steps_per_epoch / grad_accum_steps)
    )
    num_epochs = float(training_cfg["epochs"])
    return max(1, math.ceil(num_epochs * optimizer_steps_per_epoch))


def deserialize_image_payload(payload: Any) -> Any:
    if payload is None or isinstance(payload, Image.Image):
        return payload
    if isinstance(payload, (list, tuple)):
        return [deserialize_image_payload(item) for item in payload]
    if isinstance(payload, str):
        with Image.open(payload) as image:
            return image.convert("RGB").copy()
    if isinstance(payload, dict):
        if payload.get("bytes") is not None:
            with Image.open(BytesIO(payload["bytes"])) as image:
                return image.convert("RGB").copy()
        if payload.get("path"):
            with Image.open(payload["path"]) as image:
                return image.convert("RGB").copy()
    return payload


def normalize_st_input(value: Any) -> Any:
    if not isinstance(value, dict):
        return value

    normalized: dict[str, Any] = {}
    if value.get("text") is not None:
        normalized["text"] = value["text"]
    if value.get("image") is not None:
        normalized["image"] = deserialize_image_payload(value["image"])
    if value.get("audio") is not None:
        normalized["audio"] = value["audio"]
    if value.get("video") is not None:
        normalized["video"] = value["video"]

    if normalized:
        return normalized
    return value


def build_data_collator(model: SentenceTransformer) -> BaseDataCollator:
    def preprocess_fn(inputs, prompt=None, task=None):
        normalized_inputs = [normalize_st_input(item) for item in inputs]
        preprocessed = model.preprocess(normalized_inputs, prompt=prompt, task=task)

        num_images_per_sample = []
        num_videos_per_sample = []
        has_images = False
        has_videos = False
        for item in normalized_inputs:
            if isinstance(item, dict):
                image_value = item.get("image")
                video_value = item.get("video")
            else:
                image_value = None
                video_value = None

            image_count = (
                0
                if image_value is None
                else (len(image_value) if isinstance(image_value, list) else 1)
            )
            video_count = (
                0
                if video_value is None
                else (len(video_value) if isinstance(video_value, list) else 1)
            )
            num_images_per_sample.append(image_count)
            num_videos_per_sample.append(video_count)
            has_images = has_images or image_count > 0
            has_videos = has_videos or video_count > 0

        if has_images:
            preprocessed["num_images_per_sample"] = torch.tensor(
                num_images_per_sample, dtype=torch.long
            )
        if has_videos:
            preprocessed["num_videos_per_sample"] = torch.tensor(
                num_videos_per_sample, dtype=torch.long
            )

        return preprocessed

    return BaseDataCollator(preprocess_fn=preprocess_fn)


def load_sentence_transformers_datasets(cfg: dict[str, Any]):
    data_cfg = cfg["data"]
    allowed_modalities = data_cfg.get("allowed_modalities")
    train_use_iterable_dataset = bool(data_cfg.get("use_iterable_dataset", False))
    if data_cfg.get("cache_dir"):
        raise ValueError(
            "cache_dir is not supported by the native Sentence Transformers trainer path. "
            "The existing cache shards contain encoder-specific tensors from preprocess_manifest_cache.py and cannot be reused."
        )

    train_dataset, train_info = manifest_to_sentence_transformers_dataset(
        manifest_path=data_cfg["manifest_path"],
        image_root=data_cfg.get("image_root"),
        audio_root=data_cfg.get("audio_root"),
        allow_missing_negative=bool(data_cfg.get("allow_missing_negative", True)),
        allowed_modalities=allowed_modalities,
        query_modalities=data_cfg.get("query_modalities"),
        positive_modalities=data_cfg.get("positive_modalities"),
        negative_modalities=data_cfg.get("negative_modalities"),
        as_iterable=train_use_iterable_dataset,
    )

    validation_cfg = cfg.get("validation", {})
    eval_use_iterable_dataset = bool(validation_cfg.get("use_iterable_dataset", False))
    eval_dataset = None
    eval_info = None
    if validation_cfg.get("cache_dir"):
        raise ValueError(
            "validation.cache_dir is not supported by the native Sentence Transformers trainer path. "
            "Use validation.manifest_path with raw file paths instead."
        )
    if validation_cfg.get("manifest_path"):
        eval_dataset, eval_info = manifest_to_sentence_transformers_dataset(
            manifest_path=validation_cfg["manifest_path"],
            image_root=data_cfg.get("image_root"),
            audio_root=data_cfg.get("audio_root"),
            allow_missing_negative=True,
            allowed_modalities=allowed_modalities,
            query_modalities=data_cfg.get("query_modalities"),
            positive_modalities=data_cfg.get("positive_modalities"),
            negative_modalities=data_cfg.get("negative_modalities"),
            as_iterable=eval_use_iterable_dataset,
            max_records=validation_cfg.get("max_rows"),
        )

    return train_dataset, train_info, eval_dataset, eval_info


def validate_modalities(
    model: SentenceTransformer, dataset_info: dict[str, Any]
) -> None:
    supported_modalities = resolve_supported_modalities(model)
    if not supported_modalities:
        return

    unsupported = [
        modality
        for modality in dataset_info["modalities"]
        if modality not in supported_modalities
    ]
    if unsupported:
        supported_csv = ", ".join(sorted(supported_modalities))
        unsupported_csv = ", ".join(sorted(unsupported))
        raise ValueError(
            f"Model does not support dataset modalities: {unsupported_csv}. Supported modalities: {supported_csv}."
        )
