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

__version__ = "0.2.0"
