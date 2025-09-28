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
vLLM Semantic Router Benchmark Suite

A comprehensive benchmark suite for evaluating vLLM semantic router performance
against direct vLLM across multiple reasoning datasets.

Supported Datasets:
- MMLU-Pro: Academic knowledge across 57 subjects
- ARC: AI2 Reasoning Challenge for scientific reasoning
- GPQA: Graduate-level Google-proof Q&A
- TruthfulQA: Truthful response evaluation
- CommonsenseQA: Commonsense reasoning evaluation
- HellaSwag: Commonsense natural language inference

Key Features:
- Dataset-agnostic architecture with factory pattern
- Router vs direct vLLM comparison
- Multiple evaluation modes (NR, XC, NR_REASONING)
- Comprehensive plotting and analysis tools
- Research-ready CSV output
- Configurable token limits per dataset
"""

__version__ = "1.0.0"
__author__ = "vLLM Semantic Router Team"

from .dataset_factory import DatasetFactory, list_available_datasets
from .dataset_interface import DatasetInfo, DatasetInterface, PromptFormatter, Question

# Make key classes available at package level
__all__ = [
    "DatasetInterface",
    "Question",
    "DatasetInfo",
    "PromptFormatter",
    "DatasetFactory",
    "list_available_datasets",
    "__version__",
]
