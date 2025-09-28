# Copyright 2025 The vLLM Semantic Router Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
