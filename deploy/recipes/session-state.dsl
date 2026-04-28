# =============================================================================
# SESSION STATE SCHEMA RECIPE
# =============================================================================

SESSION_STATE session_routing {
  # Conversation turn counter — incremented by the router each turn.
  turn_number: int

  # Name of the model that served the most recent assistant turn.
  current_model: string

  # Accumulated spend for this session in USD, updated after each turn.
  cumulative_cost_usd: float

  # Exponential moving average of per-turn cost; tracks spend momentum.
  retry_count_ema: float

  # Exponential moving average of quality scores from the eval pipeline.
  quality_score_ema: float

  # Fraction of the KV cache that was warm on the most recent turn (0–1).
  kv_cache_warm: float
}
