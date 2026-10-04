"""Built-in model runtime for vLLM Semantic Router.

A standalone process that serves typed decision models behind one HTTP
contract (``api/openapi.yaml``). Model families, execution engines,
accelerators and numerics profiles are plugins discovered through Python
entry points; see ``docs/design.md``.
"""

__version__ = "0.1.0"
