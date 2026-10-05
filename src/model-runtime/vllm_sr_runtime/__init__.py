"""Built-in model runtime for vLLM Semantic Router.

A standalone process that serves decision, classification, embedding and
reranking models behind one HTTP contract (``api/openapi.yaml``). Model
families, execution engines, accelerators and numerics profiles are plugins
discovered through Python entry points; see ``docs/design.md``.
"""

import os

# libgomp reads this when PyTorch loads it, so it must be set before any
# submodule imports torch. A short spin keeps PyTorch's idle OpenMP threads off
# the cores an ONNX Runtime run takes right after a native forward.
os.environ.setdefault("GOMP_SPINCOUNT", "10000")
# oneDNN reads this when it creates its first primitive. Packed CPU linears see a
# new row count for almost every coalesced batch; at the default 1,024 entries
# they recompiled kernels on most calls. A cached primitive costs about 26 KB.
os.environ.setdefault("ONEDNN_PRIMITIVE_CACHE_CAPACITY", "8192")
# MIOpen reads these at its first convolution. Its default find mode times the
# candidate solvers of each new shape and keeps the fastest, so cold processes
# started together pick different solvers and answer differently; FAST takes
# the find database or the heuristic and never times. It then warns once per
# new shape, so only errors are logged.
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
os.environ.setdefault("MIOPEN_LOG_LEVEL", "3")

__version__ = "0.2.0"
