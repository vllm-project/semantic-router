# 9B operator isolation: CPU smoke setup amendment

The [signed prospective diagnostic](qwen35-9b-rocm-operator-isolation-prereg-2026-09-27.md)
required a CPU-only 64-token forward/backward smoke before GPU admission. Its
first execution matched the frozen runner and official config hashes, imported
the pinned model code, then stopped inside CPU `torch.linalg.solve_triangular`:
the pinned ROCm PyTorch build lacks CPU BLAS for `triangular_solve`. Exit code
was 1 **before any GPU device was mounted or used**. This is a CPU setup
**INDETERMINATE**, not an observed 9B ROCm result. The failed command/output
remain in the private diagnostic receipt.

This amendment changes **only the CPU admission check**. The signed runner
SHA-256 stays `c00c8ada89122acea19b035f25383e9bc3f9e2dc8504a80651c111918a5c7134`.
The GPU image, source/config hashes, synthetic inputs, gated-delta then SDPA
order, 4,096/6,144 schedule, one-container rule, 300-second/0.0834 GPU-hour
cap and stop/interpretation rules are unchanged. No optimizer, model weights,
task data or labels are introduced.

Run the following import/source/shape-only check in the **same pinned image**
with read-only `/probe.py` and `/config.json` mounts, no GPU device and no
network. It does not call `torch.linalg.solve_triangular` or execute backward:

```python
import runpy
from pathlib import Path

import torch
import torch.nn.functional as F

assert torch.cuda.device_count() == 0
scope = runpy.run_path("/probe.py", run_name="operator_isolation_cpu_preflight")
assert tuple(scope["LENGTHS"]) == (4096, 6144, 6144, 6144)
assert scope["SEED"] == 20260927
assert callable(scope["source_preflight"](Path("/config.json")))
assert callable(F.scaled_dot_product_attention)
print("PASS: CPU import, source hashes, official shapes and frozen schedule")
```

This check establishes that the exact imports, code/config hashes and official
operator shapes are available. It **does not** prove either GPU backward path
works. Require its exit 0, a fresh GPU occupancy/ownership check, hash checks
on both mounts and an independent review of this amendment before the one
frozen GPU cell. A failed check again remains INDETERMINATE; do not improvise
another setup correction or start GPU work without a new prospective amendment.
