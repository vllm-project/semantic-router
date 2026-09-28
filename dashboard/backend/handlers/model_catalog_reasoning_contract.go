package handlers

import (
	"strings"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
)

func validCatalogEffortFlags(family modelcatalog.ReasoningFamilyDefinition) bool {
	if len(family.EffortFlags) == 0 {
		return true
	}
	if family.Type != "reasoning_effort" || family.ActivationParameter == "" {
		return false
	}
	activeLevels := len(family.Levels)
	if family.Disabled != "" && catalogContains(family.Levels, family.Disabled) {
		activeLevels--
	}
	if activeLevels-len(family.EffortFlags) > 1 {
		return false
	}
	seen := make(map[string]struct{}, len(family.EffortFlags))
	for effort, parameter := range family.EffortFlags {
		if !catalogContains(family.Levels, effort) || effort == family.Disabled || parameter == "" ||
			parameter == family.Parameter || parameter == family.ActivationParameter {
			return false
		}
		if _, duplicate := seen[parameter]; duplicate {
			return false
		}
		seen[parameter] = struct{}{}
	}
	return true
}

func validCatalogReasoningModes(family modelcatalog.ReasoningFamilyDefinition) bool {
	if len(family.Modes) == 0 || family.DefaultMode == "" ||
		!catalogContains(family.Modes, family.DefaultMode) {
		return false
	}
	return validCatalogStringSet(
		family.Modes,
		func(value string) bool { return oneOf(value, "enabled", "disabled", "adaptive") },
	)
}

func validCatalogReasoningBindingContract(
	provider modelcatalog.ProviderDefinition,
	binding modelcatalog.CatalogModelBinding,
	models map[string]modelcatalog.ModelCard,
	reasoning map[string]modelcatalog.ReasoningFamilyDefinition,
) bool {
	model, modelExists := models[binding.Catalog]
	if !modelExists {
		// The caller emits the canonical unknown-model diagnostic.
		return true
	}
	if model.ReasoningFamily == "" {
		return len(binding.ReasoningModes) == 0 && len(binding.ReasoningEfforts) == 0 &&
			len(binding.ReasoningEffortsByProtocol) == 0
	}
	family, exists := reasoning[model.ReasoningFamily]
	if !exists {
		// Model validation owns the unknown-family diagnostic.
		return true
	}
	transport := binding.ReasoningTransport
	if transport == "" {
		transport = provider.ReasoningTransport
	}
	if transport == "" {
		transport = modelcatalog.ReasoningTransportChatTemplate
	}
	if !transport.SupportsFamilyType(family.Type) {
		return false
	}
	if len(binding.ReasoningModes) > 0 &&
		(!catalogStringSubset(binding.ReasoningModes, family.Modes) ||
			!catalogContains(binding.ReasoningModes, family.DefaultMode)) {
		return false
	}
	if len(binding.ReasoningEfforts) > 0 &&
		(!catalogStringSubset(binding.ReasoningEfforts, family.Levels) ||
			(family.Default != "" && !catalogContains(binding.ReasoningEfforts, family.Default))) {
		return false
	}
	for _, efforts := range binding.ReasoningEffortsByProtocol {
		if !catalogStringSubset(efforts, family.Levels) ||
			(family.Default != "" && !catalogContains(efforts, family.Default)) {
			return false
		}
	}
	return true
}

func catalogStringSubset(values, allowed []string) bool {
	for _, value := range values {
		if !catalogContains(allowed, value) {
			return false
		}
	}
	return true
}

func validCatalogReasoningBindingValues(binding modelcatalog.CatalogModelBinding) bool {
	if len(binding.ReasoningModes) > 0 && !validCatalogStringSet(
		binding.ReasoningModes,
		func(value string) bool { return oneOf(value, "enabled", "disabled", "adaptive") },
	) {
		return false
	}
	if len(binding.ReasoningEfforts) > 0 && !validCatalogStringSet(
		binding.ReasoningEfforts,
		func(value string) bool { return strings.TrimSpace(value) != "" },
	) {
		return false
	}
	if binding.ReasoningEffortsByProtocol == nil {
		return true
	}
	if len(binding.ReasoningEffortsByProtocol) == 0 || len(binding.ReasoningEfforts) == 0 {
		return false
	}
	for protocol, efforts := range binding.ReasoningEffortsByProtocol {
		if !catalogContains(binding.Protocols, protocol) || len(efforts) == 0 ||
			!validCatalogStringSet(efforts, func(value string) bool {
				return strings.TrimSpace(value) != "" && catalogContains(binding.ReasoningEfforts, value)
			}) {
			return false
		}
	}
	return true
}

func validCatalogStringSet(values []string, allowed func(string) bool) bool {
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		if !allowed(value) {
			return false
		}
		if _, duplicate := seen[value]; duplicate {
			return false
		}
		seen[value] = struct{}{}
	}
	return true
}
