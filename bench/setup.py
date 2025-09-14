#!/usr/bin/env python3
"""Setup script for vllm-semantic-router-bench package."""

import os

from setuptools import find_packages, setup


# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "A comprehensive benchmark suite for vLLM Semantic Router vs direct vLLM evaluation"


# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as f:
            return [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]
    return []


setup(
    name="vllm-semantic-router-bench",
    version="1.0.0",
    author="vLLM Semantic Router Team",
    description="Comprehensive benchmark suite for vLLM Semantic Router vs direct vLLM evaluation across multiple reasoning datasets",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/vllm-project/semantic-router",
    project_urls={
        "Bug Tracker": "https://github.com/vllm-project/semantic-router/issues",
        "Documentation": "https://vllm-semantic-router.com",
        "Source": "https://github.com/vllm-project/semantic-router",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Testing",
        "Topic :: System :: Benchmark",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
            "pre-commit>=2.15.0",
        ],
        "plotting": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vllm-semantic-router-bench=vllm_semantic_router_bench.cli:main",
            "router-bench=vllm_semantic_router_bench.router_reason_bench_multi_dataset:main",
            "bench-plot=vllm_semantic_router_bench.bench_plot:main",
        ],
    },
    include_package_data=True,
    package_data={
        "vllm_semantic_router_bench": [
            "*.md",
            "dataset_implementations/*.py",
        ],
    },
    keywords=[
        "vllm-semantic-router",
        "benchmark",
        "vllm",
        "llm",
        "evaluation",
        "reasoning",
        "multiple-choice",
        "mmlu",
        "arc",
        "gpqa",
        "commonsense",
        "hellaswag",
        "truthfulqa",
    ],
    zip_safe=False,
)
