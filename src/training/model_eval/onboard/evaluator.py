import random
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .arc_eval import ArcEvalMixin
from .mmlu_eval import MmluEvalMixin
from .report import ReportMixin
from .system_eval import SystemEvalMixin
from .types import ModelConfig, TestResult


class OnboardEvaluate(SystemEvalMixin, ArcEvalMixin, MmluEvalMixin, ReportMixin):
    """Model onboarding evaluation class (system-level eval via /api/v1/eval)"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize OnboardEvaluate

        Args:
            config_path: Path to config file
        """
        self.config_path = config_path
        self.model_config: Optional[ModelConfig] = None
        self.test_results: List[TestResult] = []
        self.arc_results: Optional[pd.DataFrame] = None
        self.mmlu_results: Optional[pd.DataFrame] = None
        self.system_eval_responses: List[Dict[str, Any]] = []
        self.system_eval_summary: List[Dict[str, Any]] = []

    def parse(self, config: Dict[str, Any]) -> ModelConfig:
        """
        Parse model configuration

        Args:
            config: Config dict including model name, endpoint, etc.

        Returns:
            ModelConfig: Parsed model config
        """
        self.model_config = ModelConfig(
            model_name=config.get("model_name", ""),
            endpoint=config.get("endpoint", ""),
            api_key=config.get("api_key", ""),
            max_tokens=config.get("max_tokens", 512),
            temperature=config.get("temperature", 0.0),
            use_cot=config.get("use_cot", False),
            seed=config.get("seed", 42),
            extra_params=config.get("extra_params", {}),
        )

        random.seed(self.model_config.seed)
        np.random.seed(self.model_config.seed)

        return self.model_config

    def run_performance_test(self, test_name: str, **kwargs) -> TestResult:
        """
        Run model performance test

        Args:
            test_name: Test name ("arc_challenge", "mmlu_pro", or "system_eval")
            **kwargs: Test parameters
                - samples: number of samples (arc_challenge)
                - samples_per_category: samples per category (mmlu_pro)
                - categories: MMLU-Pro categories
                - texts: list of texts to evaluate (system_eval)
                - options: intent options dict (system_eval)
                - input_path: optional text/jsonl file with one prompt per line (system_eval)
                - datasets: list of dataset IDs for system eval (system_eval)
                - max_samples: max samples per dataset (system_eval)

        Returns:
            TestResult: Test result
        """
        if not self.model_config:
            raise ValueError("Please call parse() first to load model config")

        if test_name == "arc_challenge":
            return self._run_arc_challenge(**kwargs)
        if test_name == "mmlu_pro":
            return self._run_mmlu_pro(**kwargs)
        if test_name == "system_eval":
            return self._run_system_eval(**kwargs)
        raise ValueError(f"Unsupported test type: {test_name}")
