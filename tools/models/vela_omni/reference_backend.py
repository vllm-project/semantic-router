"""Independent, exact CPU reference kernel selection for long GQA sequences."""

from contextlib import contextmanager

from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.integrations import sdpa_attention


@contextmanager
def cpu_flash_reference():
    """Keep published HF forward; select explicit KV repeat + Torch CPU flash.

    Transformers 4.57.6 chooses enable_gqa on Torch 2.8, which selects CPU's
    quadratic math kernel. Disabling that optimization makes its unchanged SDPA
    integration repeat KV heads explicitly. Torch's independent CPU flash kernel
    then evaluates the same complete causal attention in bounded workspace.
    This is only a qualification reference; it never changes an exported graph.
    """

    original = sdpa_attention.use_gqa_in_sdpa
    sdpa_attention.use_gqa_in_sdpa = lambda _mask, _key: False
    try:
        # Fail rather than silently selecting the quadratic math backend.
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            yield
    finally:
        sdpa_attention.use_gqa_in_sdpa = original
