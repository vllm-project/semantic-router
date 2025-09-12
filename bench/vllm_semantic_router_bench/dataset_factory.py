"""
Dataset factory for loading different evaluation datasets.

This module provides a factory pattern for instantiating different dataset
implementations in a unified way.
"""

from typing import Dict, List, Optional, Type

from .dataset_implementations.arc_dataset import (
    ARCChallengeDataset,
    ARCDataset,
    ARCEasyDataset,
)
from .dataset_implementations.commonsenseqa_dataset import CommonsenseQADataset
from .dataset_implementations.gpqa_dataset import (
    GPQADataset,
    GPQADiamondDataset,
    GPQAExtendedDataset,
    GPQAMainDataset,
)
from .dataset_implementations.hellaswag_dataset import HellaSwagDataset
from .dataset_implementations.mmlu_dataset import MMLUDataset
from .dataset_implementations.truthfulqa_dataset import TruthfulQADataset
from .dataset_interface import DatasetInterface


class DatasetFactory:
    """Factory for creating dataset instances."""

    _registered_datasets: Dict[str, Type[DatasetInterface]] = {}

    @classmethod
    def register_dataset(cls, name: str, dataset_class: Type[DatasetInterface]) -> None:
        """Register a new dataset class.

        Args:
            name: Name to register the dataset under
            dataset_class: Class implementing DatasetInterface
        """
        cls._registered_datasets[name.lower()] = dataset_class

    @classmethod
    def get_available_datasets(cls) -> List[str]:
        """Get list of all registered dataset names."""
        return list(cls._registered_datasets.keys())

    @classmethod
    def create_dataset(cls, name: str) -> DatasetInterface:
        """Create a dataset instance by name.

        Args:
            name: Name of the dataset to create

        Returns:
            Dataset instance implementing DatasetInterface

        Raises:
            ValueError: If dataset name is not registered
        """
        name_lower = name.lower()
        if name_lower not in cls._registered_datasets:
            available = ", ".join(cls.get_available_datasets())
            raise ValueError(
                f"Unknown dataset: {name}. Available datasets: {available}"
            )

        dataset_class = cls._registered_datasets[name_lower]
        return dataset_class()

    @classmethod
    def get_dataset_info(cls, name: str) -> Dict[str, str]:
        """Get basic info about a dataset without loading it.

        Args:
            name: Name of the dataset

        Returns:
            Dictionary with dataset information
        """
        dataset = cls.create_dataset(name)
        return {
            "name": dataset.dataset_name,
            "supports_cot": str(dataset.supports_cot),
            "categories_count": str(len(dataset.get_available_categories())),
        }


# Register built-in datasets
DatasetFactory.register_dataset("mmlu", MMLUDataset)
DatasetFactory.register_dataset("mmlu-pro", MMLUDataset)

# Register ARC datasets
DatasetFactory.register_dataset("arc", ARCDataset)
DatasetFactory.register_dataset("arc-easy", ARCEasyDataset)
DatasetFactory.register_dataset("arc-challenge", ARCChallengeDataset)

# Register GPQA datasets
DatasetFactory.register_dataset("gpqa", GPQAMainDataset)
DatasetFactory.register_dataset("gpqa-main", GPQAMainDataset)
DatasetFactory.register_dataset("gpqa-extended", GPQAExtendedDataset)
DatasetFactory.register_dataset("gpqa-diamond", GPQADiamondDataset)

# Register hard reasoning datasets
DatasetFactory.register_dataset("truthfulqa", TruthfulQADataset)
DatasetFactory.register_dataset("commonsenseqa", CommonsenseQADataset)
DatasetFactory.register_dataset("hellaswag", HellaSwagDataset)


def list_available_datasets() -> None:
    """Print information about all available datasets."""
    print("Available datasets:")
    print("-" * 50)

    for name in DatasetFactory.get_available_datasets():
        try:
            info = DatasetFactory.get_dataset_info(name)
            print(f"• {name}")
            print(f"  Name: {info['name']}")
            print(f"  Supports CoT: {info['supports_cot']}")
            print(f"  Categories: {info['categories_count']}")
            print()
        except Exception as e:
            print(f"• {name} (error loading info: {e})")
            print()


def create_dataset(name: str) -> DatasetInterface:
    """Convenience function to create a dataset instance.

    Args:
        name: Name of the dataset to create

    Returns:
        Dataset instance
    """
    return DatasetFactory.create_dataset(name)
