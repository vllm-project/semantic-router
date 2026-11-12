package dsl

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

// mergeModelFacts makes the facts of the base providers.models[] entries the
// DSL's: an entry's facts are those of the MODEL of its name, none without
// one; a MODEL no entry names becomes a metadata-only entry. Each entry's
// transport stays the base's.
func mergeModelFacts(base, replacement map[string]interface{}) {
	declared := map[string]map[string]interface{}{}
	var order []string
	if providers, ok := replacement["providers"].(map[string]interface{}); ok {
		models, _ := providers["models"].([]interface{})
		for _, raw := range models {
			if model, ok := raw.(map[string]interface{}); ok {
				name, _ := model["name"].(string)
				declared[name] = model
				order = append(order, name)
			}
		}
	}
	providers, _ := base["providers"].(map[string]interface{})
	if providers == nil {
		if len(order) == 0 {
			return
		}
		providers = map[string]interface{}{}
		base["providers"] = providers
	}
	models, _ := providers["models"].([]interface{})
	merged := map[string]bool{}
	for _, raw := range models {
		model, ok := raw.(map[string]interface{})
		if !ok {
			continue
		}
		for _, field := range config.ModelFactFields() {
			delete(model, field)
		}
		name, _ := model["name"].(string)
		for field, value := range declared[name] {
			model[field] = value
		}
		merged[name] = declared[name] != nil
	}
	for _, name := range order {
		if !merged[name] {
			models = append(models, declared[name])
		}
	}
	if len(models) > 0 {
		providers["models"] = models
	}
}

// BaseDecisionFields are the decision fields the DSL cannot express. A DSL
// recipe replacement keeps each from the base decision of its recipe and name;
// the Dashboard deploy keeps the same list.
func BaseDecisionFields() []string {
	return []string{"adaptations", "reliability", "fallback", "output_contract", "output_contract_spec", "annotations"}
}

// preserveBaseDecisionField carries YAML-only decision policy through a DSL
// recipe replacement. The DSL owns the executable routing surface, but the
// BaseDecisionFields are configured only in the base document.
func preserveBaseDecisionField(compiled, base interface{}, field string) {
	compiledRouting, ok := compiled.(map[string]interface{})
	if !ok {
		return
	}
	baseRouting, ok := base.(map[string]interface{})
	if !ok {
		return
	}
	compiledDecisions, ok := compiledRouting["decisions"].([]interface{})
	if !ok {
		return
	}
	baseDecisions, ok := baseRouting["decisions"].([]interface{})
	if !ok {
		return
	}

	baseByName := make(map[string]map[string]interface{}, len(baseDecisions))
	for _, rawDecision := range baseDecisions {
		decision, ok := rawDecision.(map[string]interface{})
		if !ok {
			continue
		}
		name, _ := decision["name"].(string)
		if name != "" {
			baseByName[name] = decision
		}
	}
	for _, rawDecision := range compiledDecisions {
		decision, ok := rawDecision.(map[string]interface{})
		if !ok {
			continue
		}
		if _, exists := decision[field]; exists {
			continue
		}
		name, _ := decision["name"].(string)
		baseDecision := baseByName[name]
		value, exists := baseDecision[field]
		if exists {
			decision[field] = value
		}
	}
}

// preserveBaseRecipeDecisionField matches the recipe before matching a decision.
// Identically named decisions in different recipes never share base policy.
func preserveBaseRecipeDecisionField(compiled, base interface{}, field string) {
	compiledRecipes, ok := compiled.([]interface{})
	if !ok {
		return
	}
	baseRecipes, ok := base.([]interface{})
	if !ok {
		return
	}
	baseByName := make(map[string]map[string]interface{}, len(baseRecipes))
	for _, raw := range baseRecipes {
		recipe, ok := raw.(map[string]interface{})
		if !ok {
			continue
		}
		if name, _ := recipe["name"].(string); name != "" {
			baseByName[name] = recipe
		}
	}
	for _, raw := range compiledRecipes {
		recipe, ok := raw.(map[string]interface{})
		if !ok {
			continue
		}
		name, _ := recipe["name"].(string)
		preserveBaseDecisionField(recipe, baseByName[name], field)
	}
}
