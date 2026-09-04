package catalog

func cloneEffectiveCard(value EffectiveModelCard) EffectiveModelCard {
	value.Card = cloneModel(value.Card)
	value.LoRAs = append([]LoRAAdapter(nil), value.LoRAs...)
	value.Evaluations = cloneUserEvaluations(value.Evaluations)
	value.Provenance = cloneMap(value.Provenance)
	return value
}

func cloneUserEvaluations(values []UserEvaluation) []UserEvaluation {
	if len(values) == 0 {
		return nil
	}
	result := make([]UserEvaluation, len(values))
	for index, value := range values {
		result[index] = value
		result[index].Metrics = cloneMap(value.Metrics)
		result[index].Metadata = cloneMap(value.Metadata)
	}
	return result
}

func (registry *EffectiveRegistry) Defaults() Defaults { return registry.defaults }
func (registry *EffectiveRegistry) Digest() string     { return registry.digest }

func (registry *EffectiveRegistry) Provider(name string) (EffectiveProvider, bool) {
	value, ok := registry.providers[name]
	if !ok {
		return EffectiveProvider{}, false
	}
	value.Instance.Endpoints = append([]Endpoint(nil), value.Instance.Endpoints...)
	value.Instance.Headers = cloneMap(value.Instance.Headers)
	value.Definition = cloneProvider(value.Definition)
	return value, true
}

func (registry *EffectiveRegistry) Model(name string) (EffectiveModel, bool) {
	value, ok := registry.models[name]
	if !ok {
		return EffectiveModel{}, false
	}
	value.Card = cloneEffectiveCard(value.Card)
	value.Providers = cloneEffectiveModelProviders(value.Providers)
	value.Indices = cloneIndexResults(value.Indices)
	value.BindingDefaults.ExternalModelIDs = cloneMap(value.BindingDefaults.ExternalModelIDs)
	return value, true
}

func cloneEffectiveModelProviders(values []EffectiveModelProvider) []EffectiveModelProvider {
	if len(values) == 0 {
		return nil
	}
	result := make([]EffectiveModelProvider, len(values))
	for index, value := range values {
		value.Binding.ExternalModelIDs = cloneMap(value.Binding.ExternalModelIDs)
		value.Provider.Instance.Endpoints = append([]Endpoint(nil), value.Provider.Instance.Endpoints...)
		value.Provider.Instance.Headers = cloneMap(value.Provider.Instance.Headers)
		value.Provider.Definition = cloneProvider(value.Provider.Definition)
		value.Offering = cloneOfferingPointer(value.Offering)
		result[index] = value
	}
	return result
}

func cloneOfferingPointer(value *OfferingDefinition) *OfferingDefinition {
	if value == nil {
		return nil
	}
	result := *value
	result.Protocols = append([]string(nil), value.Protocols...)
	result.Restrictions = cloneMap(value.Restrictions)
	return &result
}

func cloneIndexResults(values map[string]IndexResult) map[string]IndexResult {
	if values == nil {
		return nil
	}
	result := make(map[string]IndexResult, len(values))
	for id, value := range values {
		result[id] = cloneIndexResult(value)
	}
	return result
}

func (registry *EffectiveRegistry) ModelNames() []string { return sortedKeys(registry.models) }

func (registry *EffectiveRegistry) ProviderNames() []string { return sortedKeys(registry.providers) }

func (registry *EffectiveRegistry) Providers() []EffectiveProvider {
	names := registry.ProviderNames()
	result := make([]EffectiveProvider, 0, len(names))
	for _, name := range names {
		provider, _ := registry.Provider(name)
		result = append(result, provider)
	}
	return result
}

func (registry *EffectiveRegistry) Models() []EffectiveModel {
	names := registry.ModelNames()
	result := make([]EffectiveModel, 0, len(names))
	for _, name := range names {
		model, _ := registry.Model(name)
		result = append(result, model)
	}
	return result
}

func (registry *EffectiveRegistry) ReasoningFamilies() []ReasoningFamilyDefinition {
	ids := sortedKeys(registry.reasoningFamilies)
	result := make([]ReasoningFamilyDefinition, 0, len(ids))
	for _, id := range ids {
		result = append(result, cloneReasoningFamily(registry.reasoningFamilies[id]))
	}
	return result
}
