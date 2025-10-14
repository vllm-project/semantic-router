"""
Common LoRA Training Utilities
=============================

Shared utilities for LoRA training across different tasks (intent classification, PII detection, security detection).
This module provides common functions to avoid code duplication and ensure consistency.
"""

import gc
import logging
import os
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def get_target_modules_for_model(model_name: str) -> List[str]:
    """
    Get appropriate target_modules for LoRA based on model architecture.

    Args:
        model_name: Name of the model (e.g., "modernbert-base", "bert-base-uncased")

    Returns:
        List of module names to apply LoRA to

    Raises:
        ValueError: If model architecture is not supported
    """

    if model_name == "modernbert-base" or model_name == "answerdotai/ModernBERT-base":
        # ModernBERT architecture
        return [
            "attn.Wqkv",  # Combined query, key, value projection
            "attn.Wo",  # Attention output projection
            "mlp.Wi",  # MLP input projection (feed-forward)
            "mlp.Wo",  # MLP output projection
        ]
    elif model_name == "bert-base-uncased":
        # Standard BERT architecture - Enhanced for better performance
        return [
            "attention.self.query",
            "attention.self.key",  # Added key projection for better attention learning
            "attention.self.value",
            "attention.output.dense",
            "intermediate.dense",
            "output.dense",
        ]
    elif model_name == "roberta-base":
        # RoBERTa architecture - Enhanced for better performance
        return [
            "attention.self.query",
            "attention.self.key",  # Added key projection for better attention learning
            "attention.self.value",
            "attention.output.dense",
            "intermediate.dense",
            "output.dense",
        ]
    else:
        # Only these 3 models are supported for LoRA training
        supported_models = [
            "bert-base-uncased",
            "roberta-base",
            "modernbert-base",
            "answerdotai/ModernBERT-base",
        ]
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Only these models are supported: {supported_models}"
        )


def validate_lora_config(lora_config: Dict) -> Dict:
    """
    Validate and normalize LoRA configuration parameters.

    Args:
        lora_config: Dictionary containing LoRA parameters

    Returns:
        Validated and normalized configuration

    Raises:
        ValueError: If configuration is invalid
    """
    validated_config = lora_config.copy()

    # Validate rank
    rank = validated_config.get("rank", 8)
    if not isinstance(rank, int) or rank <= 0:
        raise ValueError(f"LoRA rank must be a positive integer, got: {rank}")
    if rank > 256:
        logger.warning(
            f"LoRA rank {rank} is very large, consider using smaller values (8-64)"
        )

    # Validate alpha
    alpha = validated_config.get("alpha", 16)
    if not isinstance(alpha, (int, float)) or alpha <= 0:
        raise ValueError(f"LoRA alpha must be a positive number, got: {alpha}")

    # Validate dropout
    dropout = validated_config.get("dropout", 0.1)
    if not isinstance(dropout, (int, float)) or not (0 <= dropout <= 1):
        raise ValueError(f"LoRA dropout must be between 0 and 1, got: {dropout}")

    # Validate target_modules
    target_modules = validated_config.get("target_modules", [])
    if not isinstance(target_modules, list) or len(target_modules) == 0:
        raise ValueError("target_modules must be a non-empty list")

    # Log configuration
    logger.info(f"LoRA Configuration validated:")
    logger.info(f"  Rank: {rank}")
    logger.info(f"  Alpha: {alpha}")
    logger.info(f"  Dropout: {dropout}")
    logger.info(f"  Target modules: {target_modules}")

    return validated_config


def get_all_gpu_info() -> List[Dict]:
    """
    Get information about all available GPUs.

    Returns:
        List of dictionaries with GPU information
    """
    if not torch.cuda.is_available():
        return []

    gpu_info = []
    num_gpus = torch.cuda.device_count()

    for gpu_id in range(num_gpus):
        try:
            props = torch.cuda.get_device_properties(gpu_id)
            total_memory = props.total_memory / 1024**3  # Convert to GB

            # Get current memory usage
            torch.cuda.set_device(gpu_id)
            allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
            reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
            free_memory = total_memory - reserved

            gpu_info.append(
                {
                    "id": gpu_id,
                    "name": torch.cuda.get_device_name(gpu_id),
                    "total_memory_gb": total_memory,
                    "allocated_memory_gb": allocated,
                    "reserved_memory_gb": reserved,
                    "free_memory_gb": free_memory,
                    "utilization_percent": (reserved / total_memory) * 100,
                }
            )
        except Exception as e:
            logger.warning(f"Could not get info for GPU {gpu_id}: {e}")
            continue

    return gpu_info


def find_free_gpu(min_free_memory_gb: float = 2.0) -> Optional[int]:
    """
    Find the GPU with the most free memory.

    Args:
        min_free_memory_gb: Minimum free memory required (in GB)

    Returns:
        GPU ID with most free memory, or None if no suitable GPU found
    """
    gpu_info = get_all_gpu_info()

    if not gpu_info:
        logger.warning("No GPUs available")
        return None

    # Sort by free memory (descending)
    gpu_info.sort(key=lambda x: x["free_memory_gb"], reverse=True)

    # Log all GPUs
    logger.info(f"Found {len(gpu_info)} GPU(s):")
    for gpu in gpu_info:
        logger.info(
            f"  GPU {gpu['id']}: {gpu['name']} - "
            f"{gpu['free_memory_gb']:.2f}GB free / {gpu['total_memory_gb']:.2f}GB total "
            f"({gpu['utilization_percent']:.1f}% utilized)"
        )

    # Find best GPU
    best_gpu = gpu_info[0]

    if best_gpu["free_memory_gb"] < min_free_memory_gb:
        logger.warning(
            f"Best GPU {best_gpu['id']} only has {best_gpu['free_memory_gb']:.2f}GB free, "
            f"but {min_free_memory_gb}GB required"
        )
        return None

    logger.info(
        f"Selected GPU {best_gpu['id']} with {best_gpu['free_memory_gb']:.2f}GB free"
    )
    return best_gpu["id"]


def set_gpu_device(
    gpu_id: Optional[int] = None, auto_select: bool = True
) -> Tuple[str, int]:
    """
    Set the GPU device to use for training.

    Args:
        gpu_id: Specific GPU ID to use (0-based), or None for auto-selection
        auto_select: If True, automatically select best GPU when gpu_id is None

    Returns:
        Tuple of (device_string, gpu_id)
    """
    if not torch.cuda.is_available():
        logger.warning("No CUDA available, using CPU")
        return "cpu", -1

    if gpu_id is not None:
        # Use specified GPU
        if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
            raise ValueError(
                f"Invalid GPU ID {gpu_id}. Available GPUs: 0-{torch.cuda.device_count()-1}"
            )

        torch.cuda.set_device(gpu_id)
        logger.info(
            f"Using specified GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}"
        )

        # Log memory info
        props = torch.cuda.get_device_properties(gpu_id)
        total_gb = props.total_memory / 1024**3
        free_gb = (props.total_memory - torch.cuda.memory_reserved(gpu_id)) / 1024**3
        logger.info(
            f"GPU {gpu_id} memory: {free_gb:.2f}GB free / {total_gb:.2f}GB total"
        )

        return f"cuda:{gpu_id}", gpu_id

    elif auto_select:
        # Auto-select best GPU
        best_gpu_id = find_free_gpu()

        if best_gpu_id is None:
            logger.warning("No suitable GPU found, using CPU")
            return "cpu", -1

        torch.cuda.set_device(best_gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu_id)

        return f"cuda:{best_gpu_id}", best_gpu_id

    else:
        # Use default GPU 0
        torch.cuda.set_device(0)
        logger.info(f"Using default GPU 0: {torch.cuda.get_device_name(0)}")
        return "cuda:0", 0


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPU memory cache cleared")


def get_memory_usage() -> Dict:
    """
    Get current memory usage information.

    Returns:
        Dictionary with memory usage statistics
    """
    memory_info = {}

    if torch.cuda.is_available():
        memory_info = {
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }
    else:
        # For CPU, we can use psutil if available, otherwise return empty
        try:
            import psutil

            memory_info = {
                "system_memory_gb": psutil.virtual_memory().total / 1024**3,
                "available_memory_gb": psutil.virtual_memory().available / 1024**3,
                "used_memory_gb": psutil.virtual_memory().used / 1024**3,
            }
        except ImportError:
            memory_info = {"note": "Install psutil for CPU memory monitoring"}

    return memory_info


def log_memory_usage(stage: str = ""):
    """Log current memory usage."""
    memory_info = get_memory_usage()
    if memory_info:
        stage_prefix = f"[{stage}] " if stage else ""
        if torch.cuda.is_available():
            logger.info(
                f"{stage_prefix}GPU Memory - Allocated: {memory_info['allocated_gb']:.2f}GB, "
                f"Reserved: {memory_info['reserved_gb']:.2f}GB"
            )
        else:
            if "system_memory_gb" in memory_info:
                logger.info(
                    f"{stage_prefix}System Memory - Used: {memory_info['used_memory_gb']:.2f}GB, "
                    f"Available: {memory_info['available_memory_gb']:.2f}GB"
                )


def create_lora_config(
    model_name: str, rank: int = 8, alpha: int = 16, dropout: float = 0.1
) -> Dict:
    """
    Create a complete LoRA configuration for a given model.

    Args:
        model_name: Name of the base model
        rank: LoRA rank (default: 8)
        alpha: LoRA alpha (default: 16)
        dropout: LoRA dropout (default: 0.1)

    Returns:
        Complete LoRA configuration dictionary
    """
    target_modules = get_target_modules_for_model(model_name)

    lora_config = {
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "target_modules": target_modules,
    }

    # Validate the configuration
    validated_config = validate_lora_config(lora_config)

    logger.info(f"Created LoRA config for {model_name}")
    logger.info(f"Target modules: {target_modules}")

    return validated_config


def get_model_mapping() -> Dict[str, str]:
    """
    Get mapping from short model names to full HuggingFace model paths.

    Returns:
        Dictionary mapping short names to full model paths
    """
    return {
        "modernbert-base": "answerdotai/ModernBERT-base",
        "modernbert-large": "answerdotai/ModernBERT-large",
        "bert-base-uncased": "bert-base-uncased",
        "bert-large-uncased": "bert-large-uncased",
        "roberta-base": "roberta-base",
        "roberta-large": "roberta-large",
        "deberta-v3-base": "microsoft/deberta-v3-base",
        "deberta-v3-large": "microsoft/deberta-v3-large",
        "distilbert-base-uncased": "distilbert-base-uncased",
    }


def resolve_model_path(model_name: str) -> str:
    """
    Resolve short model name to full HuggingFace path.

    Args:
        model_name: Short model name or full path

    Returns:
        Full model path for HuggingFace
    """
    model_mapping = get_model_mapping()
    resolved_path = model_mapping.get(model_name, model_name)

    if resolved_path != model_name:
        logger.info(f"Resolved model: {model_name} -> {resolved_path}")

    return resolved_path


def verify_target_modules(model, target_modules: List[str]) -> bool:
    """
    Verify that target_modules exist in the model architecture.

    Args:
        model: The model to check
        target_modules: List of target module names

    Returns:
        True if all target modules are found, False otherwise
    """
    model_module_names = set()
    for name, _ in model.named_modules():
        # Extract module pattern (remove layer numbers)
        if "encoder.layer" in name:
            # Convert encoder.layer.0.attention.self.query -> attention.self.query
            parts = name.split(".")
            if len(parts) >= 4 and parts[2].isdigit():
                pattern = ".".join(parts[3:])
                model_module_names.add(pattern)
        elif "layers." in name:  # ModernBERT style
            # Convert layers.0.attn.Wqkv -> attn.Wqkv
            parts = name.split(".")
            if len(parts) >= 3 and parts[1].isdigit():
                pattern = ".".join(parts[2:])
                model_module_names.add(pattern)

    missing_modules = []
    for target in target_modules:
        if target not in model_module_names:
            missing_modules.append(target)

    if missing_modules:
        logger.warning(f"Missing target modules in model: {missing_modules}")
        logger.warning(f"Available modules: {sorted(model_module_names)}")
        return False

    logger.info(f"All target modules verified: {target_modules}")
    return True


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration for LoRA training.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(__name__)
    return logger
