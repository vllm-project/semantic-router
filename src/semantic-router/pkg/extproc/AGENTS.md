# ExtProc runtime phases

- Keep request extraction, decision evaluation, route construction, streaming,
  replay/cache persistence, and response shaping as separate runtime phases.
- `processor_res_usage.go`, `processor_res_cache.go`, and
  `processor_res_memory.go` own their response-time effects.
- `res_filter_hallucination.go` and `res_filter_jailbreak.go` own response safety
  warnings; transport translation does not.
- Response API/provider translation stays at the transport edge. It must not
  acquire routing-decision or classifier policy.
- Add model selection after a decision in algorithm code, not in a signal or
  plugin branch.
- Target the affected flow first; the full package can require optional local
  model artifacts.
