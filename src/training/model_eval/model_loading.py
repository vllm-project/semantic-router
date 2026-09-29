"""Strict native artifact loading shared by classifier evaluation entrypoints."""

from pathlib import Path


def validate_label_contract(config, labels):
    """Refuse wrong heads or permuted logits before evaluating any rows."""
    expected = dict(enumerate(labels))
    actual = {int(index): label for index, label in config.id2label.items()}
    inverse = {label: index for index, label in expected.items()}
    if actual != expected or config.label2id != inverse:
        raise ValueError(
            f"Artifact label order disagrees with the evaluation contract: {actual}"
        )
    if config.num_labels != len(labels):
        raise ValueError("Artifact classifier size disagrees with its label contract")


def validate_loaded_weights(info, saved_modules=()):
    """No newly randomized parameter may silently enter an evaluation."""
    missing = [
        key
        for key in info.get("missing_keys", [])
        if not any(key == name or key.startswith(name + ".") for name in saved_modules)
    ]
    if missing or info.get("mismatched_keys"):
        raise ValueError(f"Artifact leaves uninitialized model parameters: {missing}")


def validate_saved_modules(base_state, adapter_state, saved_modules):
    """Check actual adapter tensors, not just its modules_to_save declaration."""
    for module in saved_modules:
        expected = {
            name: tensor
            for name, tensor in base_state.items()
            if name == module or name.startswith(module + ".")
        }
        if not expected:
            raise ValueError(f"Adapter declares unknown saved module: {module}")
        for name, tensor in expected.items():
            matches = [
                value
                for key, value in adapter_state.items()
                if key in (name, "base_model.model." + name)
            ]
            if len(matches) != 1 or matches[0].shape != tensor.shape:
                raise ValueError(f"Adapter lacks an exact saved task tensor: {name}")
            if not matches[0].isfinite().all().item():
                raise ValueError(f"Adapter saved task tensor is not finite: {name}")


def load_registered_model(entry, args):
    """Preserve serialized pooling, context, task head, tokenizer and precision.

    Vela adapters have repository-specific reproduction contracts; this generic
    entrypoint intentionally supports their merged native artifacts only.
    """
    import torch  # noqa: PLC0415 - keep registry checks dependency-light
    from transformers import (  # noqa: PLC0415
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoTokenizer,
    )

    use_lora = args.use_lora
    if use_lora and "lora_id" not in entry:
        raise ValueError(
            "This served Vela entry supports merged evaluation only; use its "
            "published lora/ reproduction instructions, or --collection legacy-mom"
        )
    override = args.model_id
    model_id = override or entry["lora_id" if use_lora else "id"]
    revision = getattr(args, "revision", None)
    if revision is None and not override and not use_lora:
        revision = entry.get("revision")
    dtype_name = getattr(args, "dtype", "float32")
    dtype = getattr(torch, dtype_name)
    model_class = (
        AutoModelForTokenClassification
        if entry["type"] == "token_classification"
        else AutoModelForSequenceClassification
    )
    metadata = {
        "model_id": str(model_id),
        "requested_revision": revision,
        "dtype": dtype_name,
        "device": args.device,
        "use_lora": use_lora,
    }
    if use_lora:
        from peft import PeftConfig, PeftModel  # noqa: PLC0415
        from peft.utils.save_and_load import load_peft_weights  # noqa: PLC0415

        adapter = PeftConfig.from_pretrained(model_id, revision=revision)
        base_id = adapter.base_model_name_or_path
        if not base_id:
            raise ValueError("Adapter does not declare its base model")
        if base_id.startswith(("/", ".")) and not Path(base_id).is_dir():
            raise ValueError(f"Adapter declares an unavailable local base: {base_id}")
        # The companion merged artifact supplies the task's full head/pooling
        # contract; never rebuild that config from an unrelated 8K base.
        config = AutoConfig.from_pretrained(entry["id"])
        validate_label_contract(config, entry["labels"])
        base_revision = getattr(adapter, "revision", None)
        base, info = model_class.from_pretrained(
            base_id,
            revision=base_revision,
            config=config,
            torch_dtype=dtype,
            output_loading_info=True,
        )
        validate_loaded_weights(info, adapter.modules_to_save or ())
        adapter_state = load_peft_weights(model_id, device="cpu", revision=revision)
        validate_saved_modules(
            base.state_dict(), adapter_state, adapter.modules_to_save or ()
        )
        model = PeftModel.from_pretrained(base, model_id, revision=revision)
        tokenizer = AutoTokenizer.from_pretrained(entry["id"])
        metadata.update(
            base_id=base_id,
            base_revision=base_revision,
            task_config_revision=getattr(config, "_commit_hash", None),
        )
    else:
        config = AutoConfig.from_pretrained(model_id, revision=revision)
        validate_label_contract(config, entry["labels"])
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model, info = model_class.from_pretrained(
            model_id,
            revision=revision,
            config=config,
            torch_dtype=dtype,
            output_loading_info=True,
        )
        validate_loaded_weights(info)
    validate_label_contract(model.config, entry["labels"])
    model.to(args.device).eval()
    metadata.update(
        # A PEFT model retains its companion/base config's hash; that is not
        # evidence of the adapter repository's revision.
        resolved_revision=(
            None if use_lora else getattr(model.config, "_commit_hash", None)
        ),
        classifier_pooling=getattr(model.config, "classifier_pooling", None),
        max_position_embeddings=model.config.max_position_embeddings,
        labels=list(entry["labels"]),
    )
    model.evaluation_identity = metadata
    return model, tokenizer


def tokenize_complete(tokenizer, texts, max_length, device, **kwargs):
    """Never silently turn a full-input evaluation into a prefix evaluation."""
    encoded = tokenizer(texts, padding=True, truncation=False, **kwargs)
    lengths = encoded["attention_mask"].sum(dim=-1).tolist()
    if any(length > max_length for length in lengths):
        raise ValueError(f"Input exceeds --max_length={max_length}: {lengths}")
    return encoded.to(device)
