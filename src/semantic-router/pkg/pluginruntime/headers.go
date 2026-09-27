package pluginruntime

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

// HeaderMutation preserves the exact order and append semantics emitted to
// Envoy. A preview returns this plan, not an approximation of Envoy's output.
type HeaderMutation struct {
	Name      string `json:"name"`
	Value     string `json:"value,omitempty"`
	Operation string `json:"operation"`
}

func HeaderMutations(policy *config.HeaderMutationPluginConfig) []HeaderMutation {
	result := []HeaderMutation{}
	if policy == nil {
		return result
	}
	for _, header := range policy.Add {
		result = append(result, HeaderMutation{Name: header.Name, Value: header.Value, Operation: "append"})
	}
	for _, header := range policy.Update {
		result = append(result, HeaderMutation{Name: header.Name, Value: header.Value, Operation: "set"})
	}
	for _, name := range policy.Delete {
		result = append(result, HeaderMutation{Name: name, Operation: "remove"})
	}
	return result
}
