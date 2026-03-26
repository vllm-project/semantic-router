"""Container image resolution helpers for vLLM Semantic Router."""

import os
import sys

from cli.consts import (
    DEFAULT_IMAGE_PULL_POLICY,
    IMAGE_PULL_POLICY_ALWAYS,
    IMAGE_PULL_POLICY_IF_NOT_PRESENT,
    IMAGE_PULL_POLICY_NEVER,
    PLATFORM_AMD,
    VLLM_SR_DASHBOARD_DOCKER_IMAGE_DEFAULT,
    VLLM_SR_DOCKER_IMAGE_DEFAULT,
    VLLM_SR_DOCKER_IMAGE_ROCM,
    VLLM_SR_ENVOY_DOCKER_IMAGE_DEFAULT,
    VLLM_SR_ROUTER_DOCKER_IMAGE_DEFAULT,
    VLLM_SR_ROUTER_DOCKER_IMAGE_ROCM,
    VLLM_SR_SIM_DOCKER_IMAGE_DEFAULT,
)
from cli.docker_runtime import (
    docker_image_exists,
    docker_pull_image,
    get_container_runtime,
)
from cli.utils import get_logger

log = get_logger(__name__)

RUNTIME_SERVICE_IMAGE_ENV_VARS = {
    "router": "VLLM_SR_ROUTER_IMAGE",
    "envoy": "VLLM_SR_ENVOY_IMAGE",
    "dashboard": "VLLM_SR_DASHBOARD_IMAGE",
}


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
    supported_pairs = (
        (VLLM_SR_DOCKER_IMAGE_DEFAULT, VLLM_SR_DOCKER_IMAGE_ROCM),
        (VLLM_SR_ROUTER_DOCKER_IMAGE_DEFAULT, VLLM_SR_ROUTER_DOCKER_IMAGE_ROCM),
    )
    for default_image, rocm_image in supported_pairs:
        default_repo = default_image.rsplit(":", 1)[0]
        rocm_repo = rocm_image.rsplit(":", 1)[0]
        if image_name == default_repo:
            return f"{rocm_repo}:latest"
        if image_name.startswith(f"{default_repo}:"):
            tag = image_name.split(":")[-1]
            return f"{rocm_repo}:{tag}"
    return ""


def _derive_envoy_variant(image_name):
    """Return the upstream Envoy image when the base image uses official vllm-sr tags."""
    if _is_official_vllm_sr_image(image_name):
        return VLLM_SR_ENVOY_DOCKER_IMAGE_DEFAULT
    return ""


def _derive_dashboard_variant(image_name):
    """Return a dashboard image variant when the base image uses official vllm-sr tags."""
    return _derive_official_variant(image_name, VLLM_SR_DASHBOARD_DOCKER_IMAGE_DEFAULT)


def _derive_official_variant(image_name, default_variant_image):
    """Return a role image variant when the base image uses official vllm-sr tags."""
    if not image_name:
        return ""

    image_name = image_name.strip()
    default_repo = VLLM_SR_DOCKER_IMAGE_DEFAULT.rsplit(":", 1)[0]
    rocm_repo = VLLM_SR_DOCKER_IMAGE_ROCM.rsplit(":", 1)[0]
    variant_repo = default_variant_image.rsplit(":", 1)[0]

    for candidate_repo in (default_repo, rocm_repo):
        if image_name == candidate_repo:
            return f"{variant_repo}:latest"
        prefix = f"{candidate_repo}:"
        if image_name.startswith(prefix):
            return f"{variant_repo}:{image_name[len(prefix):]}"

    return ""


def _is_official_vllm_sr_image(image_name):
    """Return True when the image is an official vllm-sr CPU or ROCm reference."""
    if not image_name:
        return False

    image_name = image_name.strip()
    for candidate in (VLLM_SR_DOCKER_IMAGE_DEFAULT, VLLM_SR_DOCKER_IMAGE_ROCM):
        candidate_repo = candidate.rsplit(":", 1)[0]
        if image_name == candidate_repo or image_name.startswith(f"{candidate_repo}:"):
            return True
    return False


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


def _resolve_runtime_service_image(
    service_name,
    *,
    explicit_image,
    base_image_is_explicit,
    base_image,
    normalized_platform,
):
    requested_image = (explicit_image or "").strip()
    if requested_image:
        selected_image = requested_image
        source_name = f"{service_name} image"
        log.info(f"Using {service_name} image from explicit override: {selected_image}")
    elif base_image_is_explicit:
        selected_image = base_image
        source_name = "explicit vllm-sr image"
        log.info(
            f"Using {service_name} image from explicit --image override: {selected_image}"
        )
    else:
        service_env_var = RUNTIME_SERVICE_IMAGE_ENV_VARS[service_name]
        env_image = os.getenv(service_env_var, "").strip()
        if env_image:
            selected_image = env_image
            source_name = f"{service_name} image in {service_env_var}"
            log.info(
                f"Using {service_name} image from {service_env_var}: {selected_image}"
            )
        else:
            derived_image = ""
            if service_name == "envoy":
                derived_image = _derive_envoy_variant(base_image)
            elif service_name == "dashboard":
                derived_image = _derive_dashboard_variant(base_image)
            selected_image = derived_image or base_image
            source_name = (
                f"derived official {service_name} image"
                if derived_image
                else "base vllm-sr image"
            )
            if derived_image:
                log.info(
                    f"Using {service_name} image derived from base runtime tag: {selected_image}"
                )
            else:
                log.info(
                    f"Using {service_name} image from base runtime image: {selected_image}"
                )

    if service_name == "router":
        return _maybe_upgrade_to_rocm_image(
            selected_image,
            normalized_platform,
            source_name,
        )
    return selected_image


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


def get_runtime_images(
    *,
    image=None,
    router_image=None,
    envoy_image=None,
    dashboard_image=None,
    pull_policy=None,
    platform=None,
):
    """Resolve role-specific runtime images with backward-compatible fallback."""
    if pull_policy is None:
        pull_policy = DEFAULT_IMAGE_PULL_POLICY

    normalized_platform = _resolve_platform_hint(platform)
    base_image_is_explicit = bool((image or "").strip())
    base_image = _resolve_selected_image(image, normalized_platform)
    selected_images = {
        "router": _resolve_runtime_service_image(
            "router",
            explicit_image=router_image,
            base_image_is_explicit=base_image_is_explicit,
            base_image=base_image,
            normalized_platform=normalized_platform,
        ),
        "envoy": _resolve_runtime_service_image(
            "envoy",
            explicit_image=envoy_image,
            base_image_is_explicit=base_image_is_explicit,
            base_image=base_image,
            normalized_platform=normalized_platform,
        ),
        "dashboard": _resolve_runtime_service_image(
            "dashboard",
            explicit_image=dashboard_image,
            base_image_is_explicit=base_image_is_explicit,
            base_image=base_image,
            normalized_platform=normalized_platform,
        ),
    }

    for unique_image in dict.fromkeys(selected_images.values()):
        _ensure_image_available(unique_image, pull_policy)

    return selected_images


def get_fleet_sim_docker_image(image=None, pull_policy=None):
    """Resolve the simulator image and ensure it is available locally."""
    if pull_policy is None:
        pull_policy = DEFAULT_IMAGE_PULL_POLICY

    if image:
        selected_image = image
        log.info(f"Using specified simulator image: {selected_image}")
    else:
        selected_image = os.getenv(
            "VLLM_SR_SIM_IMAGE", VLLM_SR_SIM_DOCKER_IMAGE_DEFAULT
        ).strip()
        log.info(f"Using simulator image: {selected_image}")

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
