"""
LLM Katan - Lightweight LLM Server for Testing

A lightweight LLM serving package using FastAPI and HuggingFace transformers,
designed for testing and development with real tiny models.
Katan (קטן) means "small" in Hebrew.

Signed-off-by: Yossi Ovadia <yovadia@redhat.com>
"""

__version__ = "0.1.4"
__author__ = "Yossi Ovadia"
__email__ = "yovadia@redhat.com"

from .cli import main
from .model import ModelBackend
from .server import create_app

__all__ = ["create_app", "ModelBackend", "main"]
