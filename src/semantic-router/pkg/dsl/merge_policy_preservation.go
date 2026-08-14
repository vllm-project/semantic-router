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
