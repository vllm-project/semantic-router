package config

import (
	"fmt"
	"strings"
	"testing"
)

// canonicalRecipeFixture builds the smallest complete standalone v0.4
// document around one Recipe document. Tests pass only the Recipe-owned
// routing value and sparse global override they need to exercise.
func canonicalRecipeFixture(documentYAML, globalYAML string) []byte {
	indent := func(value, prefix string) string {
		value = strings.TrimSpace(value)
		if value == "" {
			return ""
		}
		return prefix + strings.ReplaceAll(value, "\n", "\n"+prefix)
	}

	global := ""
	if strings.TrimSpace(globalYAML) != "" {
		global = "\nglobal:\n" + indent(globalYAML, "  ")
	}
	return []byte(fmt.Sprintf(`
version: v0.4
listeners:
  - name: http
    address: 0.0.0.0
    port: 8899
models:
  - name: cheap
    card: {}
    connections:
      - provider: private-test
        endpoint: http://127.0.0.1:8000
        model: cheap
recipes:
  - name: default
    document:
%s
entrypoints:
  - name: vllm-sr/test
    recipe: default
    assignments:
      route:
        models:
          - model: cheap%s
`, indent(documentYAML, "      "), global))
}

func canonicalLearningFixture(decisionYAML, routerYAML string) []byte {
	decision := `
decisions:
  - name: route
    priority: 1
    rules:
      operator: AND
      conditions: []`
	if strings.TrimSpace(decisionYAML) != "" {
		decision += "\n    " + strings.ReplaceAll(
			strings.TrimSpace(decisionYAML),
			"\n",
			"\n    ",
		)
	}

	global := ""
	if strings.TrimSpace(routerYAML) != "" {
		global = "router:\n  " + strings.ReplaceAll(strings.TrimSpace(routerYAML), "\n", "\n  ")
	}
	return canonicalRecipeFixture(decision, global)
}

func canonicalFixtureDecision(t *testing.T, cfg *RouterConfig) Decision {
	t.Helper()
	recipe := cfg.DefaultRecipe()
	if recipe == nil || len(recipe.Profile.Decisions) != 1 {
		t.Fatalf("fixture default Recipe decisions = %#v", recipe)
	}
	return recipe.Profile.Decisions[0]
}

func TestParseYAMLBytesAcceptsLearningAdaptationAndProtection(t *testing.T) {
	cfg, err := testAuthoringParser(t).ParseYAMLBytes(canonicalLearningFixture(`
adaptations:
  adaptation:
    mode: observe
    candidate_set: global
  protection:
    mode: apply
    stability_weight: 1.5
    switch_margin: 0.11
`, `
learning:
  enabled: true
  adaptation:
    enabled: true
    candidate_set: tier
    strategy: routing_sampling
  protection:
    enabled: true
    scope: session
    identity:
      headers:
        session: x-session-id
        conversation: x-conversation-id
    tuning:
      idle_timeout_seconds: 300
      min_turns_before_switch: 1
      switch_margin: 0.05
      stability_weight: 1.0
  state_store:
    backend: redis
    ttl_seconds: 86400
    timeout_ms: 50
    redis:
      address: redis:6379
      database: 2
      key_prefix: "vsr:router-session:v1:"
`))
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned error: %v", err)
	}
	assertLearningConfig(t, cfg)
	assertDecisionAdaptations(t, canonicalFixtureDecision(t, cfg).Adaptations, cfg.RouterLearning.Adaptation.EffectiveCandidateSet())
}

func TestValidateDecisionLearningCoversRecipeOwnedDecisions(t *testing.T) {
	cfg := &RouterConfig{Recipes: []RoutingRecipe{{
		Name: "private",
		Profile: RoutingProfile{Decisions: []Decision{{
			Name: "private-route",
			Adaptations: DecisionAdaptationsConfig{
				Adaptation: &DecisionLearningAdaptationConfig{
					CandidateSet: "invalid",
				},
			},
		}}},
	}}}

	if err := validateDecisionRouterLearningConfig(cfg); err == nil {
		t.Fatal("invalid recipe-owned learning config must be rejected")
	}
}

func assertLearningConfig(t *testing.T, cfg *RouterConfig) {
	t.Helper()
	if cfg.RouterLearning.Adaptation.EffectiveCandidateSet() != RouterLearningCandidateSetTier {
		t.Fatalf("expected tier candidate set, got %q", cfg.RouterLearning.Adaptation.EffectiveCandidateSet())
	}
	if cfg.RouterLearning.Adaptation.EffectiveStrategy() != RouterLearningStrategyRoutingSampling {
		t.Fatalf("expected routing_sampling strategy, got %q", cfg.RouterLearning.Adaptation.EffectiveStrategy())
	}
	if cfg.RouterLearning.Protection.EffectiveScope() != RouterLearningScopeSession {
		t.Fatalf("expected session protection scope, got %q", cfg.RouterLearning.Protection.EffectiveScope())
	}
	if cfg.RouterLearning.StateStore.Backend != "redis" ||
		cfg.RouterLearning.StateStore.Redis.Address != "redis:6379" {
		t.Fatalf("expected Redis state store, got %#v", cfg.RouterLearning.StateStore)
	}
}

func assertDecisionAdaptations(t *testing.T, adaptations DecisionAdaptationsConfig, defaultCandidateSet string) {
	t.Helper()
	if adaptations.AdaptationMode() != DecisionAdaptationModeObserve {
		t.Fatalf("expected adaptation observe mode, got %#v", adaptations)
	}
	if adaptations.AdaptationCandidateSet(defaultCandidateSet) != RouterLearningCandidateSetGlobal {
		t.Fatalf("expected decision candidate_set override, got %#v", adaptations.Adaptation)
	}
	if adaptations.ProtectionMode() != DecisionAdaptationModeApply {
		t.Fatalf("expected protection apply mode, got %#v", adaptations)
	}
	if adaptations.Protection == nil ||
		adaptations.Protection.StabilityWeight == nil ||
		*adaptations.Protection.StabilityWeight != 1.5 {
		t.Fatalf("expected protection stability weight override, got %#v", adaptations.Protection)
	}
	if adaptations.Protection.SwitchMargin == nil || *adaptations.Protection.SwitchMargin != 0.11 {
		t.Fatalf("expected protection switch margin override, got %#v", adaptations.Protection)
	}
}

func TestParseYAMLBytesDefaultsLearningComponentsWhenEnabled(t *testing.T) {
	cfg, err := testAuthoringParser(t).ParseYAMLBytes(canonicalLearningFixture("", `
learning:
  enabled: true
`))
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned error: %v", err)
	}
	if !cfg.RouterLearning.Enabled {
		t.Fatal("expected learning to be enabled")
	}
	if !cfg.RouterLearning.Adaptation.EffectiveEnabled() {
		t.Fatal("expected adaptation to default enabled")
	}
	if cfg.RouterLearning.Adaptation.EffectiveCandidateSet() != RouterLearningCandidateSetDecision {
		t.Fatalf("expected default decision candidate set, got %q", cfg.RouterLearning.Adaptation.EffectiveCandidateSet())
	}
	if cfg.RouterLearning.Adaptation.EffectiveStrategy() != RouterLearningStrategyRoutingSampling {
		t.Fatalf("expected default routing_sampling strategy, got %q", cfg.RouterLearning.Adaptation.EffectiveStrategy())
	}
	if !cfg.RouterLearning.Protection.EffectiveEnabled() {
		t.Fatal("expected protection to default enabled")
	}
	if cfg.RouterLearning.Protection.EffectiveScope() != RouterLearningScopeConversation {
		t.Fatalf("expected default conversation scope, got %q", cfg.RouterLearning.Protection.EffectiveScope())
	}
	if got := cfg.RouterLearning.Protection.HeaderName("session"); got != routerLearningDefaultSessionHeader {
		t.Fatalf("expected default session header %q, got %q", routerLearningDefaultSessionHeader, got)
	}
	if got := cfg.RouterLearning.Protection.HeaderName("conversation"); got != routerLearningDefaultConversationHeader {
		t.Fatalf("expected default conversation header %q, got %q", routerLearningDefaultConversationHeader, got)
	}
}

func TestParseYAMLBytesDisablesLearningComponentsIndependently(t *testing.T) {
	cfg, err := testAuthoringParser(t).ParseYAMLBytes(canonicalLearningFixture("", `
learning:
  enabled: true
  adaptation:
    enabled: false
  protection:
    enabled: false
`))
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned error: %v", err)
	}
	if !cfg.RouterLearning.Enabled {
		t.Fatal("expected learning to be enabled")
	}
	if cfg.RouterLearning.Adaptation.EffectiveEnabled() {
		t.Fatal("expected adaptation to be explicitly disabled")
	}
	if cfg.RouterLearning.Protection.EffectiveEnabled() {
		t.Fatal("expected protection to be explicitly disabled")
	}
}

func TestParseYAMLBytesAcceptsDecisionLearningBypass(t *testing.T) {
	cfg, err := testAuthoringParser(t).ParseYAMLBytes(canonicalLearningFixture(`
adaptations:
  mode: bypass
`, `
learning:
  enabled: true
  adaptation:
    enabled: true
  protection:
    enabled: true
`))
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned error: %v", err)
	}
	adaptations := canonicalFixtureDecision(t, cfg).Adaptations
	if adaptations.AdaptationMode() != DecisionAdaptationModeBypass ||
		adaptations.ProtectionMode() != DecisionAdaptationModeBypass {
		t.Fatalf("expected global decision bypass, got %#v", adaptations)
	}
}

func TestDecisionAdaptationsModeDefaultsComponents(t *testing.T) {
	cfg := DecisionAdaptationsConfig{Mode: DecisionAdaptationModeObserve}
	if cfg.AdaptationMode() != DecisionAdaptationModeObserve {
		t.Fatalf("expected decision-level observe to default adaptation mode, got %q", cfg.AdaptationMode())
	}
	if cfg.ProtectionMode() != DecisionAdaptationModeObserve {
		t.Fatalf("expected decision-level observe to default protection mode, got %q", cfg.ProtectionMode())
	}

	cfg.Protection = &DecisionLearningProtectionConfig{Mode: DecisionAdaptationModeApply}
	if cfg.ProtectionMode() != DecisionAdaptationModeObserve {
		t.Fatalf("expected decision-level observe to cap protection mode, got %q", cfg.ProtectionMode())
	}
}

func TestParseYAMLBytesRejectsDecisionObserveComponentApply(t *testing.T) {
	_, err := testAuthoringParser(t).ParseYAMLBytes(canonicalLearningFixture(`
adaptations:
  mode: observe
  adaptation:
    mode: apply
`, `
learning:
  enabled: true
`))
	if err == nil {
		t.Fatal("expected component apply to be rejected under decision observe")
	}
	if !strings.Contains(err.Error(), "adaptations.adaptation.mode cannot be \"apply\"") {
		t.Fatalf("expected adaptation mode boundary error, got %v", err)
	}
}

func TestParseYAMLBytesRejectsDecisionBypassComponentObserve(t *testing.T) {
	_, err := testAuthoringParser(t).ParseYAMLBytes(canonicalLearningFixture(`
adaptations:
  mode: bypass
  protection:
    mode: observe
`, `
learning:
  enabled: true
`))
	if err == nil {
		t.Fatal("expected component observe to be rejected under decision bypass")
	}
	if !strings.Contains(err.Error(), "adaptations.protection.mode cannot be \"observe\"") {
		t.Fatalf("expected protection mode boundary error, got %v", err)
	}
}

func TestDecisionAdaptationsCandidateSetDefaultsToGlobal(t *testing.T) {
	cfg := DecisionAdaptationsConfig{}
	if got := cfg.AdaptationCandidateSet(RouterLearningCandidateSetTier); got != RouterLearningCandidateSetTier {
		t.Fatalf("expected global candidate set, got %q", got)
	}
	cfg.Adaptation = &DecisionLearningAdaptationConfig{CandidateSet: RouterLearningCandidateSetGlobal}
	if got := cfg.AdaptationCandidateSet(RouterLearningCandidateSetTier); got != RouterLearningCandidateSetGlobal {
		t.Fatalf("expected decision candidate set override, got %q", got)
	}
}

func TestParseYAMLBytesRejectsUnknownDecisionAdaptation(t *testing.T) {
	_, err := testAuthoringParser(t).ParseYAMLBytes(canonicalLearningFixture(`
adaptations:
  session_aware:
    mode: apply
`, `
learning:
  enabled: true
  protection:
    enabled: true
`))
	if err == nil {
		t.Fatal("expected unknown decision adaptation to be rejected")
	}
	if !strings.Contains(err.Error(), "recipes[0].document.decisions[0].adaptations.session_aware") {
		t.Fatalf("expected unknown adaptation path in error, got %v", err)
	}
}

func TestParseYAMLBytesRejectsRemovedDecisionCoordination(t *testing.T) {
	_, err := testAuthoringParser(t).ParseYAMLBytes(canonicalLearningFixture(`
adaptations:
  coordination:
    protection_weight: 1.0
`, `
learning:
  enabled: true
`))
	if err == nil {
		t.Fatal("expected removed decision coordination block to be rejected")
	}
	if !strings.Contains(err.Error(), "recipes[0].document.decisions[0].adaptations.coordination") {
		t.Fatalf("expected coordination path in error, got %v", err)
	}
}

func TestParseYAMLBytesRejectsInvalidDecisionAdaptationCandidateSet(t *testing.T) {
	_, err := testAuthoringParser(t).ParseYAMLBytes(canonicalLearningFixture(`
adaptations:
  adaptation:
    candidate_set: cluster
`, `
learning:
  enabled: true
`))
	if err == nil {
		t.Fatal("expected invalid decision adaptation candidate_set to be rejected")
	}
	if !strings.Contains(err.Error(), "adaptations.adaptation.candidate_set") {
		t.Fatalf("expected candidate_set path in error, got %v", err)
	}
}

func TestParseYAMLBytesRejectsRemovedProtectionTuningFields(t *testing.T) {
	_, err := testAuthoringParser(t).ParseYAMLBytes(canonicalLearningFixture("", `
learning:
  enabled: true
  protection:
    tuning:
      cache_weight: 0.20
      handoff_penalty: 0.05
      switch_history_weight: 0.04
      max_cache_cost_multiplier: 2.5
      weight: 1.0
      protection_weight: 1.0
      handoff_penalty_weight: 1.0
`))
	if err == nil {
		t.Fatal("expected removed protection tuning fields to be rejected")
	}
	if !strings.Contains(err.Error(), "global.router.learning.protection.tuning.cache_weight") {
		t.Fatalf("expected cache_weight path in error, got %v", err)
	}
}

func TestParseYAMLBytesRejectsRemovedDecisionProtectionWeight(t *testing.T) {
	_, err := testAuthoringParser(t).ParseYAMLBytes(canonicalLearningFixture(`
adaptations:
  protection:
    weight: 1.0
`, `
learning:
  enabled: true
`))
	if err == nil {
		t.Fatal("expected removed decision protection weight to be rejected")
	}
	if !strings.Contains(err.Error(), "recipes[0].document.decisions[0].adaptations.protection.weight") {
		t.Fatalf("expected protection weight path in error, got %v", err)
	}
}

func TestParseYAMLBytesRejectsProtectionCandidateSet(t *testing.T) {
	_, err := testAuthoringParser(t).ParseYAMLBytes(canonicalLearningFixture(`
adaptations:
  protection:
    mode: apply
    candidate_set: tier
`, `
learning:
  enabled: true
`))
	if err == nil {
		t.Fatal("expected protection candidate_set to be rejected")
	}
	if !strings.Contains(err.Error(), "recipes[0].document.decisions[0].adaptations.protection.candidate_set") {
		t.Fatalf("expected protection candidate_set path in error, got %v", err)
	}
}

func TestParseYAMLBytesRejectsOldGlobalLearningAPI(t *testing.T) {
	_, err := testAuthoringParser(t).ParseYAMLBytes(canonicalLearningFixture("", `
learning:
  enabled: true
  adaptations:
    session_aware:
      enabled: true
`))
	if err == nil {
		t.Fatal("expected old learning API to be rejected")
	}
	if !strings.Contains(err.Error(), "global.router.learning.adaptations") {
		t.Fatalf("expected old learning path in error, got %v", err)
	}
}

func TestParseYAMLBytesRejectsMigratedLearningAlgorithms(t *testing.T) {
	for name, algorithmType := range map[string]string{
		"elo":             "elo",
		"rl_driven":       "rl_driven",
		"gmtrouter":       "gmtrouter",
		"bandit":          "bandit",
		"personalization": "personalization",
	} {
		t.Run(name, func(t *testing.T) {
			_, err := testAuthoringParser(t).ParseYAMLBytes(canonicalLearningFixture(`
algorithm:
  type: `+algorithmType+`
`, ""))
			if err == nil {
				t.Fatalf("expected algorithm.type=%s to be rejected", algorithmType)
			}
			if !strings.Contains(err.Error(), "global.router.learning.adaptation") {
				t.Fatalf("expected Router Learning migration error, got %v", err)
			}
		})
	}
}
