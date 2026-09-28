package dsl

// preserveBaseDecisionField carries YAML-only decision policy through a DSL
// routing replacement. The DSL owns the executable routing surface, but fields
// such as adaptations are intentionally configured only in the base document.
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
		preserveBaseDecisionField(recipe["routing"], baseByName[name]["routing"], field)
	}
}
