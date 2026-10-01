#!/usr/bin/env bash
# CPU spot check of one built package on node E with a standard CPU PyTorch (the image's ROCm build has no CPU
# LAPACK, which the Qwen3.5 reference gated-delta path needs): native (no flash-linear-attention on the path) vs
# AutoModel with flash-linear-attention on the path (its GPU-only kernels bound at import, so the remote code's
# per-layer reference forwards run): the examples bit-identical, then the first N typed-final prompts compared
# prompt by prompt (vs the GPU-scored predictions for information only).
# Usage: cpu_spot.sh <mirror decision2 dir> <package dir> <work dir> <scored predictions dir> [N]
set -euo pipefail
S="$1" PKG="$2" W="$3" P="$4" N="${5:-200}"
CPU_TORCH=/data/dev2/tools/cpu-torch212
G=/data/dev2/private/panels/goldfree
DOCK="$(dirname "$0")/dock.sh"
mkdir -p "$W"
ex() {
  bash "$DOCK" cpu --mount "$S" --mount "$(dirname "$PKG")" --mount "$CPU_TORCH" --mount "$G" --mount "$P" \
    --mount-rw "$W" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 -- python3 -I -B "$S/v2/release/examples.py" "$@"
}
common=(--package "$PKG" --device cpu --threads 32)
panel=(--panel "typed-final:$G/typed-final.prompts.jsonl:$P/typed-final.predictions.jsonl:$N")
ex run "${common[@]}" --site "$CPU_TORCH" --output "$W/native-cpu.json" > "$W/native-cpu.log" 2>&1
ex automap "${common[@]}" --site "$CPU_TORCH" --site /opt/decision-fla --reference "$W/native-cpu.json" \
  --output "$W/automap-cpu.json" > "$W/automap-cpu.log" 2>&1 || true
ex parity "${common[@]}" --site "$CPU_TORCH" "${panel[@]}" --tolerance 1 --answers "$W/native-cpu.jsonl" \
  --output "$W/parity-cpu.json" > "$W/parity-cpu.log" 2>&1 || true
ex automap-parity "${common[@]}" --site "$CPU_TORCH" --site /opt/decision-fla "${panel[@]}" --tolerance 1 \
  --answers "$W/automap-cpu.jsonl" --output "$W/automap-parity-cpu.json" > "$W/automap-parity-cpu.log" 2>&1 || true
python3 "$S/v2/release/examples.py" compare-answers "$W/native-cpu.jsonl" "$W/automap-cpu.jsonl" --tolerance 0 \
  --output "$W/cpu-compare.json" || true
python3 - "$W" <<'PY'
import json, sys
from pathlib import Path
w = Path(sys.argv[1])
def load(name):
    p = w / name
    return json.loads(p.read_text()) if p.is_file() else {}
automap, compare, parity = load("automap-cpu.json"), load("cpu-compare.json"), load("parity-cpu.json")
print(json.dumps({
    "automap_examples_passed": automap.get("passed"),
    "cpu_reference_layers": ((automap.get("checks") or {}).get("model") or {}).get("cpu_reference_layers"),
    "torch": (automap.get("runtime") or {}).get("torch"),
    "native_vs_automap": {"passed": compare.get("passed"), "max_abs_drift": compare.get("max_abs_drift"),
                          "panels": compare.get("panels")},
    "native_cpu_vs_gpu_scored": (parity.get("panels") or {}).get("typed-final"),
}, indent=1))
PY
