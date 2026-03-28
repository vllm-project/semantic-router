"""Unit tests for fleet_sim.models — ModelSpec and catalog."""

import pytest
from fleet_sim.models import (
    DEEPSEEK_V3,
    LLAMA_3_1_8B,
    LLAMA_3_1_70B,
    LLAMA_3_1_405B,
    QWEN3_30B_A3B,
    QWEN3_32B,
    QWEN3_235B_A22B,
    ModelSpec,
    get_model,
    list_models,
)


class TestModelSpec:
    def test_head_dim_llama_70b(self):
        # hidden=8192, n_heads=64 → head_dim=128
        assert LLAMA_3_1_70B.head_dim == 128

    def test_head_dim_llama_8b(self):
        # hidden=4096, n_heads=32 → head_dim=128
        assert LLAMA_3_1_8B.head_dim == 128

    def test_kv_bytes_per_token_fp16(self):
        # Llama-3.1-70B: 2×80 layers×8 KV heads×128 head_dim×2 bytes = 327,680
        expected = 2 * 80 * 8 * 128 * 2
        assert LLAMA_3_1_70B.kv_bytes_per_token == expected

    def test_kv_bytes_fp8_is_half_fp16(self):
        kv_fp16 = LLAMA_3_1_70B.kv_bytes_per_token_dtype(dtype_bytes=2)
        kv_fp8 = LLAMA_3_1_70B.kv_bytes_per_token_dtype(dtype_bytes=1)
        assert kv_fp8 == kv_fp16 // 2

    def test_param_count_llama_70b_approx(self):
        # Should be roughly 70B ± 5%
        params = LLAMA_3_1_70B.param_count()
        assert 66e9 <= params <= 74e9

    def test_param_count_llama_8b_approx(self):
        params = LLAMA_3_1_8B.param_count()
        assert 7e9 <= params <= 9e9

    def test_param_count_llama_405b_approx(self):
        params = LLAMA_3_1_405B.param_count()
        assert 390e9 <= params <= 420e9

    def test_param_bytes_fp16_is_2x_param_count(self):
        assert (
            LLAMA_3_1_70B.param_bytes(dtype_bytes=2) == LLAMA_3_1_70B.param_count() * 2
        )

    def test_param_bytes_per_gpu_scales_with_tp(self):
        tp1 = LLAMA_3_1_70B.param_bytes_per_gpu(tp=1)
        tp8 = LLAMA_3_1_70B.param_bytes_per_gpu(tp=8)
        assert tp1 / tp8 == pytest.approx(8.0, rel=0.01)

    def test_moe_flag_dense_models(self):
        for m in [LLAMA_3_1_8B, LLAMA_3_1_70B, QWEN3_32B]:
            assert not m.is_moe

    def test_moe_flag_moe_models(self):
        for m in [QWEN3_235B_A22B, QWEN3_30B_A3B, DEEPSEEK_V3]:
            assert m.is_moe

    def test_deepseek_v3_expert_config(self):
        assert DEEPSEEK_V3.n_experts == 256
        assert DEEPSEEK_V3.n_experts_topk == 8

    def test_qwen3_235b_expert_config(self):
        assert QWEN3_235B_A22B.n_experts == 128
        assert QWEN3_235B_A22B.n_experts_topk == 8

    def test_moe_active_params_less_than_total(self):
        # Active params (topk experts) << total params
        assert DEEPSEEK_V3.active_param_count() < DEEPSEEK_V3.param_count()
        ratio = DEEPSEEK_V3.active_param_count() / DEEPSEEK_V3.param_count()
        # DeepSeek-V3: 37B active / 671B total ≈ 5.5%
        assert 0.04 < ratio < 0.12

    def test_moe_param_count_deepseek_v3_approx(self):
        # DeepSeek-V3: 256 experts × 3 projections × 2048 × 7168 × 61 layers ≈ 688B FFN
        # + attention + embeddings ≈ 700B total (reported as ~671B active parameters).
        params = DEEPSEEK_V3.param_count()
        assert 650e9 <= params <= 750e9

    def test_moe_ffn_has_three_projections(self):
        """Each MoE expert has gate + up + down projections (SwiGLU); total FFN
        params must equal n_experts × 3 × moe_intermediate × hidden × n_layers."""
        m = DEEPSEEK_V3
        expected_ffn = (
            m.n_experts * 3 * m.moe_intermediate_size * m.hidden_size * m.n_layers
        )
        # shared experts (if any) use intermediate_size
        expected_ffn += (
            m.n_shared_experts * 3 * m.intermediate_size * m.hidden_size * m.n_layers
        )
        # Total params also include attention and embeddings; FFN should dominate
        actual_params = m.param_count()
        assert actual_params >= expected_ffn  # FFN is a lower bound on total

    def test_int4_kv_bytes_half_of_fp8(self):
        """int4 (dtype=0.5) KV bytes should be half of fp8 (dtype=1.0)."""
        kv_fp8 = LLAMA_3_1_70B.kv_bytes_per_token_dtype(dtype_bytes=1.0)
        kv_int4 = LLAMA_3_1_70B.kv_bytes_per_token_dtype(dtype_bytes=0.5)
        assert kv_int4 == pytest.approx(kv_fp8 / 2, rel=1e-9)
        assert kv_int4 > 0

    def test_int4_param_bytes_nonzero(self):
        """int4 (dtype=0.5) should give non-zero param bytes, not truncate to 0."""
        pb_int4 = LLAMA_3_1_70B.param_bytes(dtype_bytes=0.5)
        pb_fp16 = LLAMA_3_1_70B.param_bytes(dtype_bytes=2.0)
        assert pb_int4 > 0
        assert pb_int4 == pytest.approx(pb_fp16 / 4, rel=1e-6)

    def test_frozen_immutable(self):
        with pytest.raises((AttributeError, TypeError)):
            LLAMA_3_1_70B.n_layers = 99

    def test_display_name_humanizes_catalog_slug(self):
        assert LLAMA_3_1_70B.display_name == "Llama-3.1-70B"
        assert QWEN3_235B_A22B.display_name == "Qwen3-235B-A22B"


class TestModelCatalog:
    def test_list_models_nonempty(self):
        names = list_models()
        assert len(names) >= 9

    def test_get_by_short_name(self):
        m = get_model("llama-3.1-70b")
        assert m is LLAMA_3_1_70B

    def test_get_deepseek_v3(self):
        m = get_model("deepseek-v3")
        assert m is DEEPSEEK_V3

    def test_get_qwen3_moe(self):
        m = get_model("qwen3-235b")
        assert m is QWEN3_235B_A22B

    def test_get_unknown_raises_keyerror(self):
        with pytest.raises(KeyError):
            get_model("gpt-4")

    def test_all_catalog_names_resolvable(self):
        for name in list_models():
            m = get_model(name)
            assert isinstance(m, ModelSpec)
