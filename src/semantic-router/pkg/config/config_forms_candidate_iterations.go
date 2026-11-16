package config

// candidateIterationForms are the candidate iteration models an earlier
// release loaded and this one refuses, as it refuses them in modelRefs: a
// model providers.models does not declare, and a lora_name the model's
// providers.models entry does not declare. Candidate iterations do not choose
// the model a request is sent to, so migrate drops the model, or the name.
var candidateIterationForms = []ConfigForm{
	{
		Code:    CodeInvalidValue,
		Paths:   RoutingScopes("decisions[].candidateIterations[].models[]"),
		MatchAt: unknownIterationModel,
		What:    "a candidate iteration model names a model providers.models does not declare",
		Becomes: "the iteration without the model, and with no model left, the decision without the iteration; declare the model in providers.models to keep it",
		Migrate: func(m *Migration, at FormMatch) error {
			removeItem(at.Parent, at.Node)
			if iterations := at.Above(3); len(at.Parent.Content) == 0 {
				removeItem(iterations, at.Above(2))
				if len(iterations.Content) == 0 {
					m.Delete(at.Above(4), "candidateIterations")
				}
			}
			return nil
		},
	},
	{
		Code:    CodeInvalidValue,
		Paths:   RoutingScopes("decisions[].candidateIterations[].models[]"),
		MatchAt: undeclaredIterationLoRA,
		What:    "a candidate iteration model names a lora_name its model does not declare",
		Becomes: "the model without lora_name; declare the LoRA in the model's loras to keep it",
		Migrate: func(m *Migration, at FormMatch) error {
			m.Delete(at.Node, "lora_name")
			return nil
		},
	},
}

// unknownIterationModel reports a candidate iteration model that names neither
// a providers.models entry nor a LoRA one declares. As in modelRefs, the load
// checks the name only once providers.models declares a model.
func unknownIterationModel(at FormMatch) bool {
	name := scalarAt(at.Node, "model")
	models := sequenceAt(at.Root, "providers", "models")
	if name == "" || len(models) == 0 {
		return false
	}
	for _, model := range models {
		model = resolveAlias(model)
		if scalarAt(model, "name") == name {
			return false
		}
		for _, lora := range sequenceAt(model, "loras") {
			if scalarAt(lora, "name") == name {
				return false
			}
		}
	}
	return true
}

// undeclaredIterationLoRA reports a candidate iteration model whose lora_name
// is not among the loras of its custom providers.models entry. A catalog card
// declares its own LoRAs, which the load checks.
func undeclaredIterationLoRA(at FormMatch) bool {
	lora := scalarAt(at.Node, "lora_name")
	model := providerModelsByName(at.Root)[scalarAt(at.Node, "model")]
	if lora == "" || model == nil || scalarAt(model, "catalog") != "" {
		return false
	}
	for _, declared := range sequenceAt(model, "loras") {
		if scalarAt(declared, "name") == lora {
			return false
		}
	}
	return true
}
