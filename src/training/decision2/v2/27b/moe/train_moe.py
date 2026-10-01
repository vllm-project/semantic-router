"""MoE entry point for the unchanged ordinary trainer ``training.model.train``.

``train.py`` is byte-frozen (the archived trainer that also trained A20r). This wrapper runs
its ``main`` unchanged with the two things only an official MoE base needs:

- ``--experts-implementation`` (taken out of argv) goes to every ``DecisionModel.from_base``
  call that does not name one; a resume reads the pin from its checkpoint's metadata;
- the prompt encoder and the contract's prompt version follow the base (Gemma trains
  behind its BOS token; Qwen3.5-MoE keeps the shared prompt).

A receipt (this file's SHA-256, the pin, the prompt version) is written beside ``--output``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def resolve(argv: list[str]) -> tuple[str, str, list[str]]:
    """(experts implementation, prompt version, remaining trainer argv)."""
    from training.model.decision_model import MOE_PROMPT_VERSIONS

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--experts-implementation", choices=("eager", "grouped_mm"), required=True
    )
    known, rest = parser.parse_known_args(argv)

    def value(flag: str) -> str | None:
        return rest[rest.index(flag) + 1] if flag in rest else None

    if value("--resume"):
        metadata = json.loads(
            (Path(value("--resume")) / "decision_config.json").read_text(
                encoding="utf-8"
            )
        )
        version = metadata["prompt_version"]
        if metadata.get("experts_implementation") != known.experts_implementation:
            raise SystemExit(
                "resume checkpoint pins a different experts implementation"
            )
    else:
        config = json.loads(
            (Path(value("--model-path")) / "config.json").read_text(encoding="utf-8")
        )
        if config.get("model_type") not in MOE_PROMPT_VERSIONS:
            raise SystemExit("train_moe is for official MoE bases only")
        version = MOE_PROMPT_VERSIONS[config["model_type"]]
    return known.experts_implementation, version, rest


def main(argv: list[str] | None = None) -> None:
    experts, version, rest = resolve(list(sys.argv[1:] if argv is None else argv))
    from training.model import decision_model, train

    original = decision_model.DecisionModel.from_base.__func__

    def from_base(cls, *args, experts_implementation=None, **kwargs):
        return original(
            cls,
            *args,
            experts_implementation=experts_implementation or experts,
            **kwargs,
        )

    decision_model.DecisionModel.from_base = classmethod(from_base)
    train.PROMPT_VERSION = version
    train.encode = (
        decision_model.encode_bos
        if version == decision_model.BOS_PROMPT_VERSION
        else decision_model.encode
    )
    output = Path(rest[rest.index("--output") + 1])
    receipt = output.with_name(
        f"{output.name}.moe-trainer-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(receipt, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(
            {
                "schema_version": "decision2-27b-moe-trainer/1",
                "wrapper_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "trainer_sha256": hashlib.sha256(
                    Path(train.__file__).read_bytes()
                ).hexdigest(),
                "experts_implementation": experts,
                "prompt_version": version,
                "argv": rest,
            },
            stream,
            indent=1,
            sort_keys=True,
        )
        stream.write("\n")
    sys.argv = [train.__file__, *rest]
    train.main()


if __name__ == "__main__":
    main()
