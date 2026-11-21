package catalog

// ModelLabel is how a person reads a configured model: the display name its
// card gives, the catalog's or the operator's, else the display name of the
// card a backend's provider maps its provider model id to, else that
// provider model id. A name the configuration does not hold reads as itself.
func (registry *EffectiveRegistry) ModelLabel(name string) string {
	model, ok := registry.models[name]
	if !ok {
		return name
	}
	if _, named := model.Card.Provenance["display_name"]; named && model.Card.Card.DisplayName != "" {
		return model.Card.Card.DisplayName
	}
	for _, provider := range model.Providers {
		if card, ok := registry.cardServedAs(provider.Provider.Definition, provider.Binding.ModelID); ok {
			return card.DisplayName
		}
	}
	for _, provider := range model.Providers {
		if provider.Binding.ModelID != "" {
			return provider.Binding.ModelID
		}
	}
	if model.BindingDefaults.ModelID != "" {
		return model.BindingDefaults.ModelID
	}
	return name
}

// ModelLabels is the label of each configured model, by name. Models whose
// labels match, such as several on one provider model id, read by their
// configured names instead, so no two models read alike.
func (registry *EffectiveRegistry) ModelLabels() map[string]string {
	labels := make(map[string]string, len(registry.models))
	shared := make(map[string]int)
	for name := range registry.models {
		labels[name] = registry.ModelLabel(name)
		shared[labels[name]]++
	}
	for name, label := range labels {
		if shared[label] > 1 {
			labels[name] = name
		}
	}
	return labels
}

// cardServedAs is the card a provider's catalog maps a native model id to.
func (registry *EffectiveRegistry) cardServedAs(provider ProviderDefinition, modelID string) (ModelCard, bool) {
	for _, binding := range provider.Models {
		if modelID == "" || binding.ID != modelID {
			continue
		}
		if card, ok := registry.cards[binding.Catalog]; ok && card.Card.DisplayName != "" {
			return card.Card, true
		}
	}
	return ModelCard{}, false
}
