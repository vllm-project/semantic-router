"""Vela's applicability class extends, but does not rename, legacy feedback IDs."""

if __package__:
    from .data_contract import ID2LABEL
else:
    from data_contract import ID2LABEL

VELA_ID2LABEL = {**ID2LABEL, 4: "NO_FEEDBACK"}
VELA_LABEL2ID = {label: index for index, label in VELA_ID2LABEL.items()}


def checkpoint_labels(id2label, label2id):
    labels = {int(index): label for index, label in id2label.items()}
    if labels not in (ID2LABEL, VELA_ID2LABEL):
        raise ValueError("Unsupported feedback checkpoint label contract")
    if label2id != {label: index for index, label in labels.items()}:
        raise ValueError("Feedback label mappings must be exact inverses")
    return labels
