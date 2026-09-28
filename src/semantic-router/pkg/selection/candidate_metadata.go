package selection

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

// CandidateModelParams resolves only configured identity: the model key, an
// explicitly assigned adapter, or a unique exact external ID. It never guesses
// an unregistered model's capabilities from a name or another provider.
func CandidateModelParams(inventory map[string]config.ModelParams, refs []config.ModelRef, model string) (config.ModelParams, bool) {
	if params, ok := inventory[model]; ok {
		return params, true
	}
	for _, ref := range refs {
		if ref.LoRAName == model && ref.LoRAName != "" {
			params, ok := inventory[ref.Model]
			return params, ok
		}
	}
	var found config.ModelParams
	matches := 0
	for _, params := range inventory {
		for _, externalID := range params.ExternalModelIDs {
			if externalID == model {
				found = params
				matches++
				break
			}
		}
	}
	return found, matches == 1
}
