"""Top-level config key policy for the transitional CLI authoring path."""

EXPLICIT_TYPED_TOP_LEVEL_KEYS = {
    "auto_model_name",
    "authz",
    "bert_model",
    "classifier",
    "clear_route_cache",
    "feedback_detector",
    "hallucination_mitigation",
    "image_gen_backends",
    "include_config_models_in_list",
    "looper",
    "max_streamed_body_bytes",
    "modality_detector",
    "observability",
    "prompt_guard",
    "provider_profiles",
    "ratelimit",
    "semantic_cache",
    "response_api",
    "router_replay",
    "streamed_body_mode",
    "streamed_body_timeout_sec",
    "tools",
}


LEGACY_RUNTIME_TOP_LEVEL_COMPATIBILITY_KEYS = {
    "api",
    "categories",
    "complexity_rules",
    "config_source",
    "context_rules",
    "default_model",
    "default_reasoning_effort",
    "embedding_rules",
    "fact_check_rules",
    "jailbreak",
    "keyword_rules",
    "language_rules",
    "modality_rules",
    "model_config",
    "model_selection",
    "mom_registry",
    "pii",
    "preference_rules",
    "reasoning_families",
    "role_bindings",
    "strategy",
    "user_feedback_rules",
    "vector_store",
    "vllm_endpoints",
}

ALLOWED_USER_CONFIG_TOP_LEVEL_KEYS = (
    {
        "decisions",
        "embedding_models",
        "listeners",
        "memory",
        "providers",
        "signals",
        "version",
    }
    | EXPLICIT_TYPED_TOP_LEVEL_KEYS
    | LEGACY_RUNTIME_TOP_LEVEL_COMPATIBILITY_KEYS
)


def validate_user_config_top_level_keys(data: dict[str, object]) -> None:
    """Reject unknown top-level blocks while C004 migrates explicit schema support."""

    unknown_keys = sorted(set(data) - ALLOWED_USER_CONFIG_TOP_LEVEL_KEYS)
    if unknown_keys:
        raise ValueError(
            "unsupported top-level config keys: "
            + ", ".join(unknown_keys)
            + ". Add explicit schema support or use a named compatibility block."
        )
