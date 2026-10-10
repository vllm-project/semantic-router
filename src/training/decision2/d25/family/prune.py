"""Drop decoder layers of a Qwen3.5 code-readout export (vision tower, embeddings and readout unchanged).

    python -m d25.family.prune <src export> <dst> <kept layer indices, comma separated>

Kept layers are renumbered in order; ``layer_types`` / ``num_hidden_layers`` (and ``full_attention_interval`` when the
kept pattern is regular) follow. Writes PRUNE.json with the kept indices and parameter counts.
"""

import json
import re
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

LAYER = re.compile(r"^(language_model\.layers\.)(\d+)(\..+)$")


def main() -> None:
    src, dst, keep = (
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        [int(x) for x in sys.argv[3].split(",")],
    )
    if dst.exists():
        raise SystemExit(f"{dst} exists")
    tmp = dst.with_name(dst.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for f in src.iterdir():
        if f.is_file() and f.name not in (
            "model.safetensors",
            "config.json",
            "parity_rows.jsonl",
        ):
            shutil.copy2(f, tmp / f.name)
    config = json.loads((src / "config.json").read_text())
    text = config["text_config"]
    types = [text["layer_types"][i] for i in keep]
    text["layer_types"], text["num_hidden_layers"] = types, len(keep)
    full = [i for i, t in enumerate(types) if t == "full_attention"]
    gaps = {b - a for a, b in zip([-1] + full, full)}
    if len(gaps) == 1 and full[-1] == len(types) - 1:
        text["full_attention_interval"] = gaps.pop()
    (tmp / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    renumber = {old: new for new, old in enumerate(keep)}
    tensors, counts = {}, {"kept": 0}
    with safe_open(str(src / "model.safetensors"), framework="pt") as handle:
        metadata = handle.metadata()
        for name in handle.keys():
            m = LAYER.match(name)
            if m:
                old = int(m.group(2))
                if old not in renumber:
                    continue
                name_out = f"{m.group(1)}{renumber[old]}{m.group(3)}"
            else:
                name_out = name
            tensor = handle.get_tensor(name)
            tensors[name_out] = tensor
            counts["kept"] += tensor.numel()
    save_file(tensors, str(tmp / "model.safetensors"), metadata=metadata)
    (tmp / "PRUNE.json").write_text(
        json.dumps(
            {
                "source": src.name,
                "kept_layers": keep,
                "layer_types": types,
                "backbone_parameters": counts["kept"],
            },
            indent=1,
        )
        + "\n"
    )
    tmp.rename(dst)
    print(
        json.dumps(
            {
                "dst": str(dst),
                "layers": len(keep),
                "backbone_parameters": counts["kept"],
            }
        )
    )


if __name__ == "__main__":
    main()
