# Cross-model KV handoff planning

`PlanHandoff` checks a source cache candidate against an exact mapper and target
identity. `Coordinator` adds an enabled directional pair policy, an inclusive
turn bound, and authenticated namespace lookup through `SourceLookup`.

Initialize policies before serving requests. The zero value disables transfer.
Turn zero is the initial prompt. Registry errors, expired records, unknown
pairs, and unsupported serving configurations return no hint and use normal
prefill. A hint is a candidate, not a cache-hit result.

The registry adapter must supply the actual source endpoint, exact model
identity, cache ID, session scope, and expiry. A configured service address or
routing alias cannot substitute for those facts. The connector verifies the
actual token prefix before loading KV.

Run `go test ./pkg/kvtransfer` and `go vet ./pkg/kvtransfer` from the Router
module.
