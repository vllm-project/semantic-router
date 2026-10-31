# Architecture risks

Prefer GitHub issues for owned work, milestones, and discussion. This compact
list exists only for durable repository-specific gaps that must stay visible
while editing adjacent code. Remove an entry once an issue becomes its clear
owner or the stated proof lands.

| ID | Risk | Proof needed to retire it |
| --- | --- | --- |
| AR020 | Classification construction, discovery, and request-time dispatch can reconverge in central orchestrators. | A new classifier family can land through a family-owned adapter and focused tests without unrelated constructor edits. |
| AR044 | Flow tool state has memory/file/Redis backends but lacks Redis expiry and consume-once integration evidence. | Redis integration and E2E resume tests plus deployment guidance. |
| AR045 | Repository content moderation has no reviewed implementation; the unsafe secret-script workflow was removed. | A least-privilege reviewed app or workflow with adversarial tests. |
| AR048 | Online evaluation lacks a runtime-owned assignment/exposure ledger with behavior propensity. | Versioned, idempotent assignment and outcome contracts with estimator fixtures and rollback gates. |
| AR050 | Benchmark normalization proves schema/provenance, not execution by the pinned native adapter or grader. | Sealed adapter execution and native-metric parity on maintained golden subsets. |
| AR051 | Evaluation workers share the Dashboard service identity despite syscall and filesystem hardening. | A distinct identity/service, sealed mounts, explicit network policy, and adversarial isolation tests. |
| AR054 | Model eligibility enforces context capacity but not typed tool, image, or structured-output requirements. | One provider-neutral eligibility function used before every selection path. |
| [AR055](https://github.com/vllm-project/semantic-router/issues/2338) | Session switching cannot yet distinguish model failure from tool/provider/infrastructure failure. | A calibrated recent-window evidence gate with provenance-aware evaluation. |
| AR057 | CLI, Helm, Operator, and Dashboard do not preserve and reject backend-target shapes consistently. | Producer-parity tests and resolution of [#2469](https://github.com/vllm-project/semantic-router/issues/2469), [#2355](https://github.com/vllm-project/semantic-router/issues/2355), and [#2332](https://github.com/vllm-project/semantic-router/issues/2332). |
| AR058 | The request pipeline behind `pkg/routing` still lives in `pkg/extproc` and builds ext_proc messages internally, decoded at one boundary. | Pipeline phases return `routing.Effect` from a package outside `pkg/extproc`, the ext_proc adapter only encodes, and the parity goldens stay unchanged. |
| AR059 | Provider prompt-cache accounting has core protocol and settlement coverage, but metric-family ownership and operational consumer coverage remain incomplete. | Focused metric registration and recording modules, dashboard and documentation coverage, and maintained buffered/streaming E2E verification distinguish provider cache usage from Router response-cache usage. |
