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
- Each phase's reply is built once in `processor_phase.go` and shared by the
  ext_proc stream and routing sessions (`routing_session.go`); keep send-side
  transport logic out of those builders.
- `routing_codec.go` decodes replies into `pkg/routing` effects and fails closed.
  Extend the routing contract before the pipeline uses a new ext_proc field.
- The parity goldens under `testdata/parity` pin the exact ext_proc messages;
  a pure refactor keeps them unchanged.
