package systemonecodec

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemoneprotocol"

// StateTextDeviation sends an object or array state as its canonical text,
// exactly Python's json.dumps(state, ensure_ascii=False): ", " and ": ", the
// members in their order, no \u escaping of non-ASCII and no HTML escaping.
// It is for a systemone/0.2 target that refuses any state but a string. A
// family that renders an object state as that text reads the two the same.
type StateTextDeviation struct{}

func (StateTextDeviation) ID() string { return DeviationStateText }

func (StateTextDeviation) AppliesTo(schema SchemaID) bool { return schema == SchemaSystemOne02 }

func (StateTextDeviation) RewriteRequest(call *WireCall) (Diagnostics, error) {
	state, ok := call.Body.Get("state")
	if !ok || (state.Kind() != systemoneprotocol.KindObject && state.Kind() != systemoneprotocol.KindArray) {
		return nil, nil
	}
	call.Body = call.Body.With("state", systemoneprotocol.String(systemoneprotocol.Canonical(state)))
	return Diagnostics{{Field: "state", Action: DiagnosticRewritten, Reason: "state_text: the state goes as its canonical text"}}, nil
}

func (StateTextDeviation) RewriteReply(*WireReply) error { return nil }
