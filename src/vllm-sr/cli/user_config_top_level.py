"""Top-level config key policy for the transitional CLI authoring path."""

EXPLICIT_TYPED_TOP_LEVEL_KEYS = {
    "api",
    "auto_model_name",
    "authz",
    "bert_model",
    "classifier",
    "config_source",
    "clear_route_cache",
    "feedback_detector",
    "hallucination_mitigation",
    "image_gen_backends",
    "include_config_models_in_list",
    "looper",
    "max_streamed_body_bytes",
    "modality_detector",
    "mom_registry",
    "observability",
    "prompt_guard",
    "provider_profiles",
    "default_model",
    "default_reasoning_effort",
    "ratelimit",
    "reasoning_families",
    "semantic_cache",
    "response_api",
    "router_replay",
    "strategy",
    "streamed_body_mode",
    "streamed_body_timeout_sec",
    "tools",
    "vector_store",
}


LEGACY_RUNTIME_TOP_LEVEL_COMPATIBILITY_KEYS = {
    "categories",
    "complexity_rules",
    "context_rules",
    "embedding_rules",
    "fact_check_rules",
    "jailbreak",
    "keyword_rules",
    "language_rules",
    "modality_rules",
    "model_config",
    "model_selection",
    "pii",
    "preference_rules",
    "role_bindings",
    "user_feedback_rules",
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
