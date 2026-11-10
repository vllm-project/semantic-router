package config

// candidateIterationForms are the candidate iteration models an earlier
// release loaded and this one refuses: a lora_name the model's
// providers.models entry does not declare, as it is refused in modelRefs.
// Candidate iterations do not choose the model a request is sent to, so
// migrate drops the name.
var candidateIterationForms = []ConfigForm{
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
