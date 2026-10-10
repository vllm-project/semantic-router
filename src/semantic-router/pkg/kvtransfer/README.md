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

## Configuration

`global.integrations.kv_transfer` declares enabled directional mapper pairs and
concrete backend capabilities. Each backend has a routing alias, backend name,
and immutable serving identity. Each pair names its mapper, source and target
identities, and inclusive `max_transfer_turn`. The integration and each pair
have explicit enable switches. Weight and tokenizer revisions are pinned commit
hashes. The current connector uses bf16 with TP=1.
