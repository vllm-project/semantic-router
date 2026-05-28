"""Anthropic-shape shim in front of llama.cpp's llama-server.

Bridges three gaps in llama-server's Messages API support so the e2e
suite can drive realistic Anthropic-shape upstream behaviour without
forking llama.cpp.
"""

__version__ = "0.1.0"
