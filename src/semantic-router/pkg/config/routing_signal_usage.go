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
			if recipeUsesSignalType(&c.Recipes[i], normalizedType) {
				return true
			}
		}
		return false
	}

	return decisionsUseSignalType(c.Decisions, c.Projections, normalizedType)
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
		return decisionsUseSignalType(c.Decisions, c.Projections, normalizedType)
	}
	if len(c.Recipes) == 0 {
		if !c.IsRecipeReachableForRouting(DefaultRecipeName) {
			return false
		}
		return decisionsUseSignalType(c.Decisions, c.Projections, normalizedType)
	}
	for _, recipe := range c.ReachableRoutingRecipes() {
		if recipeUsesSignalType(recipe, normalizedType) {
			return true
		}
	}
	return false
}

func recipeUsesSignalType(recipe *RoutingRecipe, signalType string) bool {
	if recipe == nil {
		return false
	}
	return decisionsUseSignalType(recipe.Profile.Decisions, recipe.Profile.Projections, signalType)
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

// NeedsCategoryMappingForRouting returns true when routing actually depends on
// the local file-backed category classifier assets.
func (c *RouterConfig) NeedsCategoryMappingForRouting() bool {
	return c != nil && c.IsCategoryClassifierEnabled() && c.UsesSignalTypeInReachableRouting(SignalTypeDomain)
}

// NeedsPIIMappingForRouting returns true when routing actually depends on the
// local file-backed PII classifier assets.
func (c *RouterConfig) NeedsPIIMappingForRouting() bool {
	return c != nil && c.IsPIIClassifierEnabled() && c.UsesSignalTypeInReachableRouting(SignalTypePII)
}

// NeedsJailbreakMappingForRouting returns true when routing actually depends on
// the local file-backed jailbreak classifier assets.
func (c *RouterConfig) NeedsJailbreakMappingForRouting() bool {
	return c != nil && c.IsPromptGuardEnabled() && c.UsesJailbreakClassifierInReachableRouting()
}

// UsesJailbreakClassifierInReachableRouting reports whether a request-reachable
// routing profile depends on the jailbreak classifier. It has three consumers:
// a decision rule that reads a request-direction jailbreak rule, a
// response-direction rule that the selected decision's response_jailbreak
// plugin consumes, and a response_jailbreak plugin that classifies the
// response itself because no rule is declared. Decision rules never name a
// response-direction rule, so the signal-type walk alone misses both
// response-stage consumers; the mapping then stays unloaded, the plugin never
// runs, and an http_classify prompt_guard refuses to build at startup.
func (c *RouterConfig) UsesJailbreakClassifierInReachableRouting() bool {
	if c == nil {
		return false
	}
	if c.UsesSignalTypeInReachableRouting(SignalTypeJailbreak) {
		return true
	}
	for _, signals := range c.reachableRoutingSignals() {
		for _, rule := range signals.JailbreakRules {
			if rule.Stage() == SignalStageResponse {
				return true
			}
		}
	}
	decisions := c.routingConsumerDecisions()
	for i := range decisions {
		plugin := decisions[i].GetResponseJailbreakConfig()
		if plugin != nil && plugin.Enabled {
			return true
		}
	}
	return false
}

// reachableRoutingSignals is the signal-declaration counterpart of
// routingConsumerDecisions: the signals of every request-reachable routing
// profile.
func (c *RouterConfig) reachableRoutingSignals() []Signals {
	if c == nil {
		return nil
	}
	if c.RoutingScope != "" {
		return []Signals{c.Signals}
	}
	if len(c.Recipes) == 0 {
		if !c.IsRecipeReachableForRouting(DefaultRecipeName) {
			return nil
		}
		return []Signals{c.Signals}
	}

	signals := make([]Signals, 0, len(c.Recipes))
	for _, recipe := range c.ReachableRoutingRecipes() {
		if recipe != nil {
			signals = append(signals, recipe.Profile.Signals)
		}
	}
	return signals
}

// NeedsFactCheckModelForAPI reports whether the default public fact-check API
// is configured. Declared default-profile rules and the legacy global enable
// flag both opt the API into the bound model, even when no decision references
// those rules.
func (c *RouterConfig) NeedsFactCheckModelForAPI() bool {
	if !c.ownsDefaultAPIConsumer() ||
		c.HallucinationMitigation.FactCheckModel.ModelID == "" {
		return false
	}
	return c.HallucinationMitigation.Enabled ||
		len(c.RoutingProfileSignals().FactCheckRules) > 0
}

// NeedsFeedbackModelForAPI reports whether the default public feedback API is
// configured independently of recipe routing.
func (c *RouterConfig) NeedsFeedbackModelForAPI() bool {
	return c.ownsDefaultAPIConsumer() &&
		c.FeedbackDetector.Enabled &&
		c.FeedbackDetector.ModelID != "" &&
		len(c.RoutingProfileSignals().UserFeedbackRules) > 0
}

// NeedsHallucinationDetectorForDefaultRuntime preserves the legacy global
// hallucination-module consumer. For a local backend this detector also owns
// the object through which the public NLI API is served.
func (c *RouterConfig) NeedsHallucinationDetectorForDefaultRuntime() bool {
	return c.ownsDefaultAPIConsumer() &&
		c.HallucinationMitigation.Enabled &&
		c.HallucinationMitigation.HallucinationModel.ModelID != ""
}

// NeedsLocalHallucinationNLIForAPI reports whether the default public NLI API
// has a local detector and explainer configured. Endpoint-backed hallucination
// detection does not implement the local NLI classification API.
func (c *RouterConfig) NeedsLocalHallucinationNLIForAPI() bool {
	return c.ownsDefaultAPIConsumer() &&
		c.NeedsHallucinationDetectorForDefaultRuntime() &&
		c.HallucinationMitigation.HallucinationModel.NormalizedBackend() == HallucinationBackendCandle &&
		c.HallucinationMitigation.NLIModel.ModelID != ""
}

func (c *RouterConfig) ownsDefaultAPIConsumer() bool {
	return c != nil &&
		(c.RoutingScope == "" || c.RoutingScope == DefaultRecipeName)
}

// NeedsFactCheckModelForRouting reports whether an active routing path depends
// on the configured fact-check classifier. Signal declarations alone do not
// require model provisioning; a decision must reference the signal directly or
// through a projection.
func (c *RouterConfig) NeedsFactCheckModelForRouting() bool {
	return c != nil &&
		c.HallucinationMitigation.FactCheckModel.ModelID != "" &&
		c.UsesSignalTypeInReachableRouting(SignalTypeFactCheck)
}

// NeedsFeedbackModelForRouting reports whether an active routing path depends
// on the configured feedback classifier.
func (c *RouterConfig) NeedsFeedbackModelForRouting() bool {
	return c != nil &&
		c.FeedbackDetector.Enabled &&
		c.FeedbackDetector.ModelID != "" &&
		c.UsesSignalTypeInReachableRouting(SignalTypeUserFeedback)
}

// NeedsHallucinationDetectorForRouting reports whether any decision in any
// recipe enables the response-side hallucination plugin.
func (c *RouterConfig) NeedsHallucinationDetectorForRouting() bool {
	if c == nil || c.HallucinationMitigation.HallucinationModel.ModelID == "" {
		return false
	}
	decisions := c.routingConsumerDecisions()
	for i := range decisions {
		plugin := decisions[i].GetHallucinationConfig()
		if plugin != nil && plugin.Enabled {
			return true
		}
	}
	return false
}

// NeedsLocalHallucinationModelsForRouting reports whether routing requires the
// in-process hallucination detector and optional NLI model. Endpoint-backed
// detectors own their model lifecycle outside the router.
func (c *RouterConfig) NeedsLocalHallucinationModelsForRouting() bool {
	return c != nil &&
		c.NeedsHallucinationDetectorForRouting() &&
		c.HallucinationMitigation.HallucinationModel.NormalizedBackend() == HallucinationBackendCandle
}

// NeedsLocalHallucinationNLIForRouting reports whether an enabled local
// hallucination plugin requests NLI explanations.
func (c *RouterConfig) NeedsLocalHallucinationNLIForRouting() bool {
	if c == nil ||
		!c.NeedsLocalHallucinationModelsForRouting() ||
		c.HallucinationMitigation.NLIModel.ModelID == "" {
		return false
	}
	decisions := c.routingConsumerDecisions()
	for i := range decisions {
		plugin := decisions[i].GetHallucinationConfig()
		if plugin != nil && plugin.Enabled && plugin.UseNLI {
			return true
		}
	}
	return false
}

// NeedsLocalNLIForSemanticCache reports whether the enabled semantic cache runs
// the NLI polarity tier, which binds the hallucination explainer model. The
// native binding holds a single NLI model shared by every consumer, so this is
// the only model the tier can use.
func (c *RouterConfig) NeedsLocalNLIForSemanticCache() bool {
	return c != nil &&
		c.SemanticCache.Enabled &&
		c.SemanticCache.PolarityGuard.UsesNLI() &&
		c.HallucinationMitigation.NLIModel.ModelID != ""
}

func (c *RouterConfig) routingConsumerDecisions() []Decision {
	if c == nil {
		return nil
	}
	if c.RoutingScope != "" {
		return c.Decisions
	}
	if len(c.Recipes) == 0 {
		if !c.IsRecipeReachableForRouting(DefaultRecipeName) {
			return nil
		}
		return c.Decisions
	}

	decisions := make([]Decision, 0)
	for _, recipe := range c.ReachableRoutingRecipes() {
		if recipe != nil {
			decisions = append(decisions, recipe.Profile.Decisions...)
		}
	}
	return decisions
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
		if inputType == ProjectionInputKBMetric {
			continue
		}
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
