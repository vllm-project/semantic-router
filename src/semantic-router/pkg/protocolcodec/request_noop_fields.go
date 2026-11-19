package protocolcodec

import (
	"encoding/json"
	"slices"
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// noOpRequestValues are the values of request fields the Router cannot carry
// that ask for what a backend does without the field. A Chat or Responses
// request that sends one reaches the backend without it and reports it as
// dropped; any other value of the field is still refused.
var noOpRequestValues = map[string]struct {
	noOp   func(json.RawMessage) bool
	reason string
}{
	"service_tier": {
		func(value json.RawMessage) bool { return jsonStringIn(value, "auto", "default") },
		"the auto and default service tiers ask for the backend's standard processing",
	},
	"modalities": {
		func(value json.RawMessage) bool {
			var modalities []string
			return json.Unmarshal(value, &modalities) == nil && len(modalities) == 1 && modalities[0] == "text"
		},
		"text is the only output a backend returns without modalities",
	},
	"safety_identifier": {
		func(value json.RawMessage) bool { return jsonStringIn(value) },
		"an end-user identifier for the provider's abuse monitoring is not model input",
	},
}

// dropNoOpRequestFields clears each field whose value is a no-op and returns
// their names, sorted.
func dropNoOpRequestFields(fields map[string]*json.RawMessage) []string {
	var dropped []string
	for name, value := range fields {
		if rule, ok := noOpRequestValues[name]; ok && len(*value) > 0 && rule.noOp(*value) {
			*value = nil
			dropped = append(dropped, name)
		}
	}
	sort.Strings(dropped)
	return dropped
}

func appendNoOpFieldOmissions(diagnostics *llmprotocol.Diagnostics, policy llmprotocol.Policy, source llmprotocol.WireFormat, dropped []string) {
	for _, name := range dropped {
		appendProviderFieldOmission(diagnostics, policy, source, name, noOpRequestValues[name].reason)
	}
}

// jsonStringIn reports whether value is a JSON string, one of allowed when
// any are given.
func jsonStringIn(value json.RawMessage, allowed ...string) bool {
	var text string
	return json.Unmarshal(value, &text) == nil && (len(allowed) == 0 || slices.Contains(allowed, text))
}
