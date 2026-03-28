"""Tests for ModelSpec.from_hf_config() and ModelSpec.from_hf_repo().

The from_hf_config() tests use local config dicts (no network).
The from_hf_repo() tests are skipped unless --hf-online is passed,
since they require network access.
"""

import json
import urllib.error

import pytest
from fleet_sim.models.spec import ModelSpec

# ── Sample config dicts (matching real HF config.json structure) ──────────────

_LLAMA_CONFIG = {
    "_name_or_path": "meta-llama/Meta-Llama-3.1-8B",
    "model_type": "llama",
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "vocab_size": 128256,
    "max_position_embeddings": 131072,
}

_MISTRAL_CONFIG = {
    "_name_or_path": "mistralai/Mistral-7B-v0.3",
    "model_type": "mistral",
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "vocab_size": 32000,
    "max_position_embeddings": 32768,
}

_QWEN3_MOE_CONFIG = {
    "_name_or_path": "Qwen/Qwen3-235B-A22B",
    "model_type": "qwen3_moe",
    "num_hidden_layers": 94,
    "num_attention_heads": 64,
    "num_key_value_heads": 4,
    "hidden_size": 4096,
    "intermediate_size": 0,
    "moe_intermediate_size": 1536,
    "vocab_size": 151936,
    "max_position_embeddings": 40960,
    "num_experts": 128,
    "num_experts_per_tok": 8,
}

_MIXTRAL_CONFIG = {
    "_name_or_path": "mistralai/Mixtral-8x7B-v0.1",
    "model_type": "mixtral",
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "vocab_size": 32000,
    "max_position_embeddings": 32768,
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
}

_DEEPSEEK_CONFIG = {
    "_name_or_path": "deepseek-ai/DeepSeek-V3",
    "model_type": "deepseek_v3",
    "num_hidden_layers": 61,
    "num_attention_heads": 128,
    "num_key_value_heads": 128,
    "hidden_size": 7168,
    "intermediate_size": 0,
    "moe_intermediate_size": 2048,
    "vocab_size": 129280,
    "max_position_embeddings": 163840,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "num_shared_experts": 1,
}


# ── from_hf_config(dict) ──────────────────────────────────────────────────────


class TestFromHfConfigDict:
    def test_dense_basic_fields(self):
        spec = ModelSpec.from_hf_config(_LLAMA_CONFIG)
        assert spec.n_layers == 32
        assert spec.n_heads == 32
        assert spec.n_kv_heads == 8
        assert spec.hidden_size == 4096
        assert spec.intermediate_size == 14336
        assert spec.vocab_size == 128256
        assert spec.max_position == 131072
        assert not spec.is_moe

    def test_name_from_config(self):
        spec = ModelSpec.from_hf_config(_LLAMA_CONFIG)
        assert spec.name == "meta-llama/Meta-Llama-3.1-8B"

    def test_display_name_from_hf_config(self):
        spec = ModelSpec.from_hf_config(_LLAMA_CONFIG)
        assert spec.display_name == "Meta-Llama-3.1-8B"

    def test_name_override(self):
        spec = ModelSpec.from_hf_config(_LLAMA_CONFIG, name="my-llama")
        assert spec.name == "my-llama"

    def test_mha_fallback_for_missing_kv_heads(self):
        cfg = dict(_MISTRAL_CONFIG)
        del cfg["num_key_value_heads"]
        spec = ModelSpec.from_hf_config(cfg)
        assert spec.n_kv_heads == spec.n_heads

    def test_mixtral_is_moe(self):
        spec = ModelSpec.from_hf_config(_MIXTRAL_CONFIG)
        assert spec.is_moe
        assert spec.n_experts == 8
        assert spec.n_experts_topk == 2

    def test_qwen3_moe_config(self):
        spec = ModelSpec.from_hf_config(_QWEN3_MOE_CONFIG)
        assert spec.is_moe
        assert spec.n_experts == 128
        assert spec.n_experts_topk == 8
        assert spec.moe_intermediate_size == 1536

    def test_deepseek_v3_config(self):
        spec = ModelSpec.from_hf_config(_DEEPSEEK_CONFIG)
        assert spec.is_moe
        assert spec.n_experts == 256
        assert spec.n_experts_topk == 8
        assert spec.moe_intermediate_size == 2048
        assert spec.n_shared_experts == 1

    def test_returns_modelspec_instance(self):
        spec = ModelSpec.from_hf_config(_LLAMA_CONFIG)
        assert isinstance(spec, ModelSpec)

    def test_frozen_result(self):
        spec = ModelSpec.from_hf_config(_LLAMA_CONFIG)
        with pytest.raises((AttributeError, TypeError)):
            spec.n_layers = 99

    def test_head_dim_correct(self):
        spec = ModelSpec.from_hf_config(_LLAMA_CONFIG)
        assert spec.head_dim == 4096 // 32  # 128

    def test_param_count_reasonable(self):
        spec = ModelSpec.from_hf_config(_LLAMA_CONFIG)
        assert 7e9 <= spec.param_count() <= 9e9

    def test_kv_bytes_per_token_positive(self):
        spec = ModelSpec.from_hf_config(_LLAMA_CONFIG)
        assert spec.kv_bytes_per_token > 0


# ── from_hf_config(path) ──────────────────────────────────────────────────────


class TestFromHfConfigPath:
    def test_load_from_file_path(self, tmp_path):
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(_LLAMA_CONFIG))
        spec = ModelSpec.from_hf_config(cfg_path)
        assert spec.n_layers == 32

    def test_load_from_string_path(self, tmp_path):
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(_LLAMA_CONFIG))
        spec = ModelSpec.from_hf_config(str(cfg_path))
        assert spec.n_layers == 32

    def test_name_from_config_when_loaded_from_file(self, tmp_path):
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(_LLAMA_CONFIG))
        spec = ModelSpec.from_hf_config(cfg_path)
        assert spec.name == "meta-llama/Meta-Llama-3.1-8B"

    def test_name_override_when_loaded_from_file(self, tmp_path):
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(_LLAMA_CONFIG))
        spec = ModelSpec.from_hf_config(cfg_path, name="local-model")
        assert spec.name == "local-model"

    def test_moe_from_file(self, tmp_path):
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(_DEEPSEEK_CONFIG))
        spec = ModelSpec.from_hf_config(cfg_path)
        assert spec.is_moe
        assert spec.n_experts == 256


# ── from_hf_config with model_id string (mocked, no real network) ─────────────


class TestFromHfConfigModelId:
    def test_nonexistent_path_triggers_download_attempt(self, monkeypatch):
        from fleet_sim.models import spec as spec_module

        fetched = {}

        def mock_fetch(model_id, token=None):
            fetched["model_id"] = model_id
            return dict(_LLAMA_CONFIG, **{"_name_or_path": model_id})

        monkeypatch.setattr(spec_module, "_fetch_hf_config", mock_fetch)
        result = ModelSpec.from_hf_config("meta-llama/Meta-Llama-3.1-8B")
        assert fetched["model_id"] == "meta-llama/Meta-Llama-3.1-8B"
        assert result.n_layers == 32


# ── from_hf_repo (mocked network) ────────────────────────────────────────────


class TestFromHfRepo:
    def test_downloads_and_parses_config(self, monkeypatch):
        from fleet_sim.models import spec as spec_module

        def mock_fetch(model_id, token=None):
            assert model_id == "Qwen/Qwen3-8B"
            return {
                "_name_or_path": "Qwen/Qwen3-8B",
                "model_type": "qwen3",
                "num_hidden_layers": 36,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "hidden_size": 4096,
                "intermediate_size": 12288,
                "vocab_size": 151936,
                "max_position_embeddings": 40960,
            }

        monkeypatch.setattr(spec_module, "_fetch_hf_config", mock_fetch)
        spec = ModelSpec.from_hf_repo("Qwen/Qwen3-8B")
        assert spec.n_layers == 36
        assert spec.name == "Qwen/Qwen3-8B"

    def test_passes_token_to_fetch(self, monkeypatch):
        from fleet_sim.models import spec as spec_module

        received_token = {}

        def mock_fetch(model_id, token=None):
            received_token["token"] = token
            return dict(_LLAMA_CONFIG)

        monkeypatch.setattr(spec_module, "_fetch_hf_config", mock_fetch)
        ModelSpec.from_hf_repo("meta-llama/Meta-Llama-3.1-8B", token="hf_abc123")
        assert received_token["token"] == "hf_abc123"

    def test_name_override(self, monkeypatch):
        from fleet_sim.models import spec as spec_module

        monkeypatch.setattr(
            spec_module,
            "_fetch_hf_config",
            lambda model_id, token=None: dict(_LLAMA_CONFIG),
        )
        spec = ModelSpec.from_hf_repo("meta-llama/Meta-Llama-3.1-8B", name="llama8b")
        assert spec.name == "llama8b"

    def test_http_error_propagates(self, monkeypatch):
        from fleet_sim.models import spec as spec_module

        def raise_404(model_id, token=None):
            raise urllib.error.HTTPError("url", 404, "Not Found", {}, None)

        monkeypatch.setattr(spec_module, "_fetch_hf_config", raise_404)
        with pytest.raises(urllib.error.HTTPError):
            ModelSpec.from_hf_repo("nonexistent/model")


# ── Compatibility with ProfileBuilder ────────────────────────────────────────


class TestHfSpecWithProfileBuilder:
    def test_profile_builder_accepts_hf_spec(self):
        from fleet_sim import H100_SXM
        from fleet_sim.gpu_profiles import ProfileBuilder, ServingConfig

        spec = ModelSpec.from_hf_config(_LLAMA_CONFIG)
        profile = ProfileBuilder().build(
            H100_SXM, spec, ServingConfig(tp=4, dtype_bytes=2, mean_ctx_tokens=2048)
        )
        assert profile.W > 0
        assert profile.H > 0
        assert profile.total_kv_blks > 0

    def test_moe_profile_builder_accepts_hf_spec(self):
        from fleet_sim import H100_SXM
        from fleet_sim.gpu_profiles import ProfileBuilder, ServingConfig

        spec = ModelSpec.from_hf_config(_MIXTRAL_CONFIG)
        profile = ProfileBuilder().build(
            H100_SXM, spec, ServingConfig(tp=4, dtype_bytes=2, mean_ctx_tokens=2048)
        )
        assert profile.W > 0
        assert spec.is_moe
