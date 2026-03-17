"""Container image resolution helpers for vLLM Semantic Router."""

import os
import sys

from cli.consts import (
    DEFAULT_IMAGE_PULL_POLICY,
    IMAGE_PULL_POLICY_ALWAYS,
    IMAGE_PULL_POLICY_IF_NOT_PRESENT,
    IMAGE_PULL_POLICY_NEVER,
    PLATFORM_AMD,
    VLLM_SR_DOCKER_IMAGE_DEFAULT,
    VLLM_SR_DOCKER_IMAGE_ROCM,
)
from cli.docker_runtime import (
    docker_image_exists,
    docker_pull_image,
    get_container_runtime,
)
from cli.utils import get_logger

log = get_logger(__name__)


def _normalize_platform(platform):
    """Normalize platform input for comparisons."""
    if platform is None:
        return ""
    return str(platform).strip().lower()


def _is_rocm_image(image_name):
    """Return True when the image name appears to be a ROCm image variant."""
    if not image_name:
        return False
    return "rocm" in image_name.lower()


def _derive_rocm_variant(image_name):
    """Return a ROCm variant for official vllm-sr image references."""
    if not image_name:
        return ""

    image_name = image_name.strip()
    default_repo = VLLM_SR_DOCKER_IMAGE_DEFAULT.rsplit(":", 1)[0]
    if not image_name.startswith(default_repo):
        return ""

    if image_name == default_repo:
        return f"{default_repo}-rocm:latest"
    if image_name.startswith(f"{default_repo}:"):
        tag = image_name.split(":")[-1]
        return f"{default_repo}-rocm:{tag}"
    return ""


def _resolve_platform_hint(platform):
    return _normalize_platform(platform) or _normalize_platform(
        os.getenv("VLLM_SR_PLATFORM")
    )


def _select_image_source(image, normalized_platform):
    if image:
        log.info(f"Using specified image: {image}")
        return image
    env_image = os.getenv("VLLM_SR_IMAGE")
    if env_image:
        log.info(f"Using image from VLLM_SR_IMAGE: {env_image}")
        return env_image
    if normalized_platform == PLATFORM_AMD:
        amd_image = os.getenv("VLLM_SR_IMAGE_AMD", VLLM_SR_DOCKER_IMAGE_ROCM).strip()
        selected_image = amd_image or VLLM_SR_DOCKER_IMAGE_ROCM
        log.info(
            f"Platform '{normalized_platform}' detected, using AMD ROCm default image: "
            f"{selected_image}"
        )
        return selected_image
    log.info(f"Using default image: {VLLM_SR_DOCKER_IMAGE_DEFAULT}")
    return VLLM_SR_DOCKER_IMAGE_DEFAULT


def _maybe_upgrade_to_rocm_image(selected_image, normalized_platform, source_name):
    if normalized_platform != PLATFORM_AMD or _is_rocm_image(selected_image):
        return selected_image

    rocm_variant = _derive_rocm_variant(selected_image)
    if rocm_variant:
        log.warning(
            f"Platform 'amd' selected with non-ROCm official {source_name}. "
            f"Switching to ROCm image: {rocm_variant}"
        )
        return rocm_variant

    log.warning(
        f"Platform 'amd' selected but {source_name} does not look like a ROCm image. "
        "GPU acceleration may not be enabled. Prefer a '*-rocm' image."
    )
    return selected_image


def _resolve_selected_image(image, normalized_platform):
    if image:
        return _maybe_upgrade_to_rocm_image(
            _select_image_source(image, normalized_platform),
            normalized_platform,
            "vllm-sr image",
        )

    env_image = os.getenv("VLLM_SR_IMAGE")
    if env_image:
        return _maybe_upgrade_to_rocm_image(
            _select_image_source(None, normalized_platform),
            normalized_platform,
            "vllm-sr image in VLLM_SR_IMAGE",
        )

    return _select_image_source(None, normalized_platform)


def _ensure_image_available(selected_image, pull_policy):
    image_exists = docker_image_exists(selected_image)
    if pull_policy == IMAGE_PULL_POLICY_ALWAYS:
        _pull_or_exit(selected_image)
        return
    if pull_policy == IMAGE_PULL_POLICY_IF_NOT_PRESENT:
        if image_exists:
            log.info(f"Image exists locally: {selected_image}")
            return
        log.info("Image not found locally, pulling...")
        _pull_or_exit(selected_image, show_not_found=True)
        return
    if pull_policy == IMAGE_PULL_POLICY_NEVER and not image_exists:
        log.error(f"Image not found locally: {selected_image}")
        log.error("Pull policy is 'never', cannot pull image")
        _show_image_not_found_error(selected_image)
        sys.exit(1)
    if image_exists:
        log.info(f"Image exists locally: {selected_image}")


def _pull_or_exit(selected_image, show_not_found=False):
    if docker_pull_image(selected_image):
        return
    log.error(f"Failed to pull image: {selected_image}")
    if show_not_found:
        _show_image_not_found_error(selected_image)
    sys.exit(1)


def get_docker_image(image=None, pull_policy=None, platform=None):
    """
    Determine which Docker image to use and handle pulling if needed.

    Priority:
    1. Explicit image parameter (--image)
    2. VLLM_SR_IMAGE environment variable
    3. Platform-specific default image (e.g. AMD -> ROCm image)
    4. Default image

    Args:
        image: Explicit image name (optional)
        pull_policy: Image pull policy - 'always', 'ifnotpresent', 'never'
        platform: Platform hint from CLI/environment (e.g. 'amd')

    Returns:
        Docker image name
    """
    if pull_policy is None:
        pull_policy = DEFAULT_IMAGE_PULL_POLICY
    normalized_platform = _resolve_platform_hint(platform)
    selected_image = _resolve_selected_image(image, normalized_platform)
    _ensure_image_available(selected_image, pull_policy)
    return selected_image


def _show_image_not_found_error(image_name):
    """Show helpful error message when image is not found."""
    runtime = get_container_runtime()
    log.error("=" * 70)
    log.error("Container image not found!")
    log.error("=" * 70)
    log.error("")
    log.error(f"Image: {image_name}")
    log.error("")
    log.error("Options:")
    log.error("")
    log.error("  1. Pull the image:")
    log.error(f"     {runtime} pull {image_name}")
    log.error("")
    log.error("  2. Use custom image:")
    log.error("     vllm-sr serve config.yaml --image your-image:tag")
    log.error("")
    log.error("  3. Change pull policy to always:")
    log.error("     vllm-sr serve config.yaml --image-pull-policy always")
    log.error("")
    log.error("=" * 70)
