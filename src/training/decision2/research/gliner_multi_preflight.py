"""One gold-free native smoke for a pinned multilingual GLiNER boundary model.

Run only on an authorized experiment host. This establishes API compatibility;
the invented sentence is not a benchmark or a model-selection example.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import torch
    from gliner2.classification.engine import Classifier
    from gliner2.classification.schema import ClassificationSchema
    from inference.gliner25 import score_question

    native = (
        Classifier.from_pretrained(
            str(args.model_path), device=args.device, dtype=torch.float32
        )
        .to(args.device)
        .eval()
    )
    schema = ClassificationSchema().single(
        "decision",
        {"book": "预订航班", "cancel": "取消航班", "status": "查询航班状态"},
        instruction="选择最合适的操作。",
    )
    compiled = native.compile_schema(schema)
    result = native.score("请帮我取消明天的航班。", compiled)
    labels = result.tasks["decision"]
    probabilities = {
        label: float(result.probability("decision", label)) for label in labels
    }
    projected = score_question(
        native,
        "请帮我取消明天的航班。",
        {
            "type": "choice",
            "instructions": "选择最合适的操作。",
            "criteria": {
                "book": "预订航班",
                "cancel": "取消航班",
                "status": "查询航班状态",
            },
        },
    )
    long_text = "请帮我取消明天的航班。" * 100
    long_tokens = len(
        native.model.processor.transform_record(long_text, compiled.build()).input_ids
    )
    try:
        native.score(long_text, compiled)
        long_native_error = None
    except (RuntimeError, ValueError) as exc:
        long_native_error = type(exc).__name__
    print(
        json.dumps(
            {
                "model_class": type(native.model).__name__,
                "labels": sorted(labels),
                "probabilities": probabilities,
                "projected_choice": projected["choice"],
                "native_encoder_positions": native.model.encoder.config.max_position_embeddings,
                "native_config_max_len": getattr(native.model.config, "max_len", None),
                "long_input_tokens": long_tokens,
                "long_native_error": long_native_error,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
