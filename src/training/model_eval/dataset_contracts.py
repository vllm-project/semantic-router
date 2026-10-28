"""Dataset compatibility checks shared by classifier evaluation entrypoints."""

from numbers import Integral


def require_default_dataset(entry):
    """Do not silently reuse a predecessor's incompatible task annotations."""
    if not entry.get("hf_dataset"):
        raise ValueError(
            f"{entry['id']} has no compatible default evaluation dataset; "
            "use --custom_dataset with independently reviewed instruction-attack "
            "labels (benign/jailbreak). Toxicity and safe/unsafe labels are not "
            "instruction-attack annotations."
        )


def classification_label_id(label, entry):
    """Map only declared classes or integral class IDs, without dropping rows."""
    labels = entry["labels"]
    if isinstance(label, str):
        label = entry.get("dataset_label_aliases", {}).get(label, label)
        if label not in labels:
            raise ValueError(f"Unrecognized dataset label: {label}")
        return labels.index(label)
    if isinstance(label, bool) or not isinstance(label, Integral):
        raise ValueError(f"Dataset label must be a class name or integer: {label}")
    if label not in range(len(labels)):
        raise ValueError(f"Dataset label outside artifact contract: {label}")
    return int(label)
