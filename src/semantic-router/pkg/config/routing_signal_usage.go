package config

import "strings"

// HasSignalType returns true if the decision's rules tree references at least
// one signal of the given type (e.g., "jailbreak", "pii").
func (d *Decision) HasSignalType(signalType string) bool {
	return len(collectSignalNames(&d.Rules, signalType)) > 0
}

// UsesSignalTypeInRouting returns true when any routing decision uses the given
// signal type directly or indirectly via projection outputs.
func (c *RouterConfig) UsesSignalTypeInRouting(signalType string) bool {
	if c == nil {
		return false
	}

	normalizedType := strings.ToLower(strings.TrimSpace(signalType))
	if normalizedType == "" {
		return false
	}

	projectionOutputs := make(map[string]struct{})
	for i := range c.Decisions {
		decision := &c.Decisions[i]
		if decision.HasSignalType(normalizedType) {
			return true
		}
		for _, name := range collectSignalNames(&decision.Rules, SignalTypeProjection) {
			normalizedName := strings.ToLower(strings.TrimSpace(name))
			if normalizedName != "" {
				projectionOutputs[normalizedName] = struct{}{}
			}
		}
	}

	return projectionsReferenceSignalType(c.Projections, projectionOutputs, normalizedType)
}

// NeedsCategoryMappingForRouting returns true when routing actually depends on
// the local file-backed category classifier assets.
func (c *RouterConfig) NeedsCategoryMappingForRouting() bool {
	return c != nil && c.IsCategoryClassifierEnabled() && c.UsesSignalTypeInRouting(SignalTypeDomain)
}

// NeedsPIIMappingForRouting returns true when routing actually depends on the
// local file-backed PII classifier assets.
func (c *RouterConfig) NeedsPIIMappingForRouting() bool {
	return c != nil && c.IsPIIClassifierEnabled() && c.UsesSignalTypeInRouting(SignalTypePII)
}

// NeedsJailbreakMappingForRouting returns true when routing actually depends on
// the local file-backed jailbreak classifier assets.
func (c *RouterConfig) NeedsJailbreakMappingForRouting() bool {
	return c != nil && c.IsPromptGuardEnabled() && c.UsesSignalTypeInRouting(SignalTypeJailbreak)
}

func projectionsReferenceSignalType(projections Projections, outputs map[string]struct{}, signalType string) bool {
	if len(outputs) == 0 || len(projections.Scores) == 0 || len(projections.Mappings) == 0 {
		return false
	}

	scoreByName := projectionScoresByName(projections.Scores)
	sourceByOutput := projectionSourcesByOutput(projections.Mappings)
	for outputName := range outputs {
		if projectionOutputReferencesSignalType(outputName, sourceByOutput, scoreByName, signalType) {
			return true
		}
	}

	return false
}

func projectionScoresByName(scores []ProjectionScore) map[string]ProjectionScore {
	scoreByName := make(map[string]ProjectionScore, len(scores))
	for _, score := range scores {
		scoreByName[strings.ToLower(strings.TrimSpace(score.Name))] = score
	}
	return scoreByName
}

func projectionSourcesByOutput(mappings []ProjectionMapping) map[string]string {
	sourceByOutput := make(map[string]string)
	for _, mapping := range mappings {
		normalizedSource := strings.ToLower(strings.TrimSpace(mapping.Source))
		for _, output := range mapping.Outputs {
			normalizedOutput := strings.ToLower(strings.TrimSpace(output.Name))
			if normalizedOutput != "" {
				sourceByOutput[normalizedOutput] = normalizedSource
			}
		}
	}
	return sourceByOutput
}

func projectionOutputReferencesSignalType(outputName string, sourceByOutput map[string]string, scoreByName map[string]ProjectionScore, signalType string) bool {
	scoreName, ok := sourceByOutput[outputName]
	if !ok {
		return false
	}
	score, ok := scoreByName[scoreName]
	if !ok {
		return false
	}
	return projectionScoreReferencesSignalType(score, signalType)
}

func projectionScoreReferencesSignalType(score ProjectionScore, signalType string) bool {
	for _, input := range score.Inputs {
		if strings.EqualFold(input.Type, ProjectionInputKBMetric) {
			continue
		}
		if strings.EqualFold(strings.TrimSpace(input.Type), signalType) {
			return true
		}
	}
	return false
}

// collectSignalNames traverses a RuleNode tree and returns all leaf signal names
// of the given signal type.
func collectSignalNames(node *RuleNode, signalType string) []string {
	if node == nil {
		return nil
	}
	if node.Type == signalType && node.Name != "" {
		return []string{node.Name}
	}
	var names []string
	for i := range node.Conditions {
		names = append(names, collectSignalNames(&node.Conditions[i], signalType)...)
	}
	return names
}
