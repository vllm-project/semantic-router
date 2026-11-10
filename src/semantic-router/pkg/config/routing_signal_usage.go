package config

import "strings"

// HasSignalType returns true if the decision's rules tree references at least
// one signal of the given type (e.g., "jailbreak", "pii").
func (d *Decision) HasSignalType(signalType string) bool {
	normalizedType := strings.ToLower(strings.TrimSpace(signalType))
	if normalizedType == "" {
		return false
	}
	return len(collectSignalNames(&d.Rules, normalizedType)) > 0
}

// UsesSignalTypeInRouting returns true when any routing decision in any
// recipe uses the given signal type directly or indirectly via projection
// outputs.
func (c *RouterConfig) UsesSignalTypeInRouting(signalType string) bool {
	if c == nil {
		return false
	}

	normalizedType := strings.ToLower(strings.TrimSpace(signalType))
	if normalizedType == "" {
		return false
	}

	if len(c.Recipes) > 0 {
		for i := range c.Recipes {
			if c.ConfigForRecipe(&c.Recipes[i]).profileUsesSignalType(normalizedType) {
				return true
			}
		}
		return false
	}

	return c.profileUsesSignalType(normalizedType)
}

// UsesSignalTypeInReachableRouting reports whether a request-reachable routing
// profile depends on the signal type. Immutable recipe-scoped configs evaluate
// their own profile directly; root configs ignore named recipes that have no
// entrypoint.
func (c *RouterConfig) UsesSignalTypeInReachableRouting(signalType string) bool {
	if c == nil {
		return false
	}

	normalizedType := strings.ToLower(strings.TrimSpace(signalType))
	if normalizedType == "" {
		return false
	}
	if c.RoutingScope != "" {
		return c.profileUsesSignalType(normalizedType)
	}
	if len(c.Recipes) == 0 {
		if !c.IsRecipeReachableForRouting(DefaultRecipeName) {
			return false
		}
		return c.profileUsesSignalType(normalizedType)
	}
	for _, recipe := range c.ReachableRoutingRecipes() {
		if c.ConfigForRecipe(recipe).profileUsesSignalType(normalizedType) {
			return true
		}
	}
	return false
}

func decisionsUseSignalType(decisions []Decision, projections Projections, signalType string) bool {
	projectionOutputs := make(map[string]struct{})
	for i := range decisions {
		decision := &decisions[i]
		if decision.HasSignalType(signalType) {
			return true
		}
		for _, name := range collectSignalNames(&decision.Rules, SignalTypeProjection) {
			normalizedName := strings.ToLower(strings.TrimSpace(name))
			if normalizedName != "" {
				projectionOutputs[normalizedName] = struct{}{}
			}
		}
	}

	return projectionsReferenceSignalType(projections, projectionOutputs, signalType)
}

func projectionsReferenceSignalType(projections Projections, outputs map[string]struct{}, signalType string) bool {
	if len(outputs) == 0 || len(projections.Scores) == 0 || len(projections.Mappings) == 0 {
		return false
	}

	scoreByName := projectionScoresByName(projections.Scores)
	sourceByOutput := projectionSourcesByOutput(projections.Mappings)
	state := make(map[string]uint8, len(scoreByName))
	matches := make(map[string]bool, len(scoreByName))
	for outputName := range outputs {
		if projectionOutputReferencesSignalType(outputName, sourceByOutput, scoreByName, signalType, state, matches) {
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

func projectionOutputReferencesSignalType(
	outputName string,
	sourceByOutput map[string]string,
	scoreByName map[string]ProjectionScore,
	signalType string,
	state map[string]uint8,
	matches map[string]bool,
) bool {
	scoreName, ok := sourceByOutput[strings.ToLower(strings.TrimSpace(outputName))]
	if !ok {
		return false
	}
	return projectionScoreReferencesSignalType(scoreName, sourceByOutput, scoreByName, signalType, state, matches)
}

func projectionScoreReferencesSignalType(
	scoreName string,
	sourceByOutput map[string]string,
	scoreByName map[string]ProjectionScore,
	signalType string,
	state map[string]uint8,
	matches map[string]bool,
) bool {
	normalizedScoreName := strings.ToLower(strings.TrimSpace(scoreName))
	switch state[normalizedScoreName] {
	case 1:
		// Invalid programmatic configs can bypass projection validation. Treat a
		// back edge as non-matching so startup discovery terminates safely.
		return false
	case 2:
		return matches[normalizedScoreName]
	}

	score, ok := scoreByName[normalizedScoreName]
	if !ok {
		return false
	}
	state[normalizedScoreName] = 1
	for _, input := range score.Inputs {
		inputType := strings.ToLower(strings.TrimSpace(input.Type))
		if inputType == signalType {
			matches[normalizedScoreName] = true
			break
		}
		if inputType != SignalTypeProjection {
			continue
		}

		dependency := strings.ToLower(strings.TrimSpace(input.Name))
		if strings.EqualFold(strings.TrimSpace(input.ValueSource), ProjectionValueSourceConfidence) {
			dependency = sourceByOutput[dependency]
		}
		if projectionScoreReferencesSignalType(dependency, sourceByOutput, scoreByName, signalType, state, matches) {
			matches[normalizedScoreName] = true
			break
		}
	}
	state[normalizedScoreName] = 2
	return matches[normalizedScoreName]
}

// collectSignalNames traverses a RuleNode tree and returns all leaf signal names
// of the given signal type.
func collectSignalNames(node *RuleNode, signalType string) []string {
	if node == nil {
		return nil
	}
	if strings.ToLower(strings.TrimSpace(node.Type)) == signalType && node.Name != "" {
		return []string{node.Name}
	}
	var names []string
	for i := range node.Conditions {
		names = append(names, collectSignalNames(&node.Conditions[i], signalType)...)
	}
	return names
}

// profileUsesSignalType includes Replay's evidence dependency in addition to
// decision and projection inputs.
func (c *RouterConfig) profileUsesSignalType(signalType string) bool {
	if signalType == SignalTypePII && c.ReplayNeedsPIIEvidence() {
		return true
	}
	return decisionsUseSignalType(c.Decisions, c.Projections, signalType)
}
