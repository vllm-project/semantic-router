package config

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
)

func TestNativeProfileRejectsUnavailableSignalContextThroughProjections(t *testing.T) {
	for _, kind := range []string{
		SignalTypeAuthz, SignalTypeMetadata, SignalTypeConversation, SignalTypeReask,
		SignalTypeUserFeedback, SignalTypeInputModality, SignalTypeKB, ProjectionInputKBMetric,
	} {
		t.Run(kind, func(t *testing.T) {
			profiles := []RoutingProfile{
				{Decisions: []Decision{{Name: "gate", Rules: RuleNode{Operator: "NOT", Conditions: []RuleNode{{Type: kind, Name: "signal"}}}}}},
				{Projections: Projections{Scores: []ProjectionScore{{Name: "derived", Inputs: []ProjectionScoreInput{{Type: kind, Name: "signal", Weight: 1}}}}}},
			}
			for _, profile := range profiles {
				if err := validateNativeProfileSurfaces(profile); err == nil || !strings.Contains(err.Error(), kind) {
					t.Fatalf("unsupported native context was accepted: %v", err)
				}
			}
		})
	}
	profile := RoutingProfile{
		Strategy:    RoutingStrategyConfidence,
		Projections: Projections{Scores: []ProjectionScore{{Name: "text", Inputs: []ProjectionScoreInput{{Type: SignalTypeKeyword, Name: "topic", Weight: 1}}}}},
		Decisions:   []Decision{{Name: "gate", Rules: RuleNode{Type: SignalTypeProjection, Name: "derived"}}},
	}
	if err := validateNativeProfileSurfaces(profile); err != nil {
		t.Fatal(err)
	}
}

func TestNativeConfigurationRejectsIgnoredChatControls(t *testing.T) {
	for _, test := range []struct {
		name   string
		mutate func(*RoutingProfile, *Decision)
	}{
		{"fallback", func(p *RoutingProfile, _ *Decision) { p.Fallback = &fallback.FallbackPolicy{Enabled: true} }},
		{"candidate_requirements", func(p *RoutingProfile, _ *Decision) {
			p.CandidateRequirements = &CandidateRequirements{Context: CandidateContextKnownLimits}
		}},
		{"reliability", func(_ *RoutingProfile, d *Decision) { d.Reliability = &DecisionReliability{TotalTimeout: "1ms"} }},
		{"output", func(_ *RoutingProfile, d *Decision) { d.OutputContract = "text" }},
		{"adaptations", func(_ *RoutingProfile, d *Decision) { d.Adaptations.Mode = "observe" }},
		{"iterations", func(_ *RoutingProfile, d *Decision) {
			d.CandidateIterations = []CandidateIterationConfig{{Variable: "model"}}
		}},
		{"emits", func(_ *RoutingProfile, d *Decision) { d.Emits = []EmitDirective{{Kind: "retention"}} }},
		{"weights", func(_ *RoutingProfile, d *Decision) { d.ModelRefs[0].Weight = 2 }},
		{"reasoning", func(_ *RoutingProfile, d *Decision) { value := false; d.ModelRefs[0].UseReasoning = &value }},
	} {
		t.Run(test.name, func(t *testing.T) {
			cfg, err := ParseYAMLBytes([]byte(nativeRoutingTestYAML))
			if err != nil {
				t.Fatal(err)
			}
			profile := findRecipe(cfg.Recipes, "native-decisions").Profile
			if profile.Decisions[0].ModelRefs[0].UseReasoning != nil {
				t.Fatal("native modelRefs inherited a Chat-only default")
			}
			test.mutate(&profile, &profile.Decisions[0])
			if err := validateNativeRecipe(cfg, RoutingRecipe{Name: "native-decisions", Profile: profile}, true, false); err == nil {
				t.Fatal("ignored native control accepted")
			}
		})
	}
}

func TestNativeRequestFactsFailDuringConfigurationParsing(t *testing.T) {
	raw := strings.Replace(nativeRoutingTestYAML, "      budget:", "      signals:\n        metadata:\n          - {name: tenant, key: tenant, predicate: {equals: private}}\n      budget:", 1)
	raw = strings.Replace(raw, "rules: {}", "rules: {type: metadata, name: tenant}", 1)
	if _, err := ParseYAMLBytes([]byte(raw)); err == nil || !strings.Contains(err.Error(), "native System One") {
		t.Fatalf("native metadata rule silently accepted: %v", err)
	}
}

func TestNativePreferenceRequiresRequestBoundJudgment(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(nativeRoutingTestYAML))
	if err != nil {
		t.Fatal(err)
	}
	profile := findRecipe(cfg.Recipes, "native-decisions").Profile
	profile.Signals.PreferenceRules = []PreferenceRule{{Name: "brief", Description: "Brief replies"}, {Name: "detailed", Description: "Detailed replies"}}
	profile.Decisions[0].Rules = RuleNode{Type: SignalTypePreference, Name: "brief"}
	for _, test := range []struct {
		name   string
		mutate func(*RouterConfig)
		valid  bool
	}{
		{"default judgment", func(*RouterConfig) {}, true},
		{"prototypes", func(c *RouterConfig) { enabled := true; c.PreferenceModel.UseContrastive = &enabled }, false},
		{"authored examples", func(c *RouterConfig) { c.PreferenceRules[0].Examples = []string{"be brief"} }, false},
		{"external LLM", func(c *RouterConfig) { c.ExternalModels = []ExternalModelConfig{{ModelRole: ModelRolePreference}} }, false},
		{"HTTP binding", func(c *RouterConfig) {
			c.ModelBindings["preference"] = ModelBinding{Deployment: "remote", Contract: DecisionTaskContract}
		}, false},
		{"explicit judgment overrides prototypes", func(c *RouterConfig) {
			enabled := true
			c.PreferenceModel.UseContrastive = &enabled
			c.ModelBindings["preference"] = ModelBinding{Deployment: "local-kai", Contract: DecisionTaskContract}
		}, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			recipe := RoutingRecipe{Name: "native-decisions", Profile: profile}
			local := cfg.ConfigForRecipe(&recipe)
			local.PreferenceRules = append([]PreferenceRule(nil), profile.Signals.PreferenceRules...)
			local.ModelBindings = map[string]ModelBinding{}
			test.mutate(local)
			err := validateNativeSignalBackends(local, profile)
			if (err == nil) != test.valid {
				t.Fatalf("valid=%v error=%v", test.valid, err)
			}
		})
	}
	cfg.MCPCategoryModel.Enabled = true
	profile.Decisions[0].Rules = RuleNode{Type: SignalTypeDomain, Name: "other"}
	if err := validateNativeSignalBackends(cfg, profile); err == nil {
		t.Fatal("MCP-only native domain adapter accepted")
	}
}

func TestNativeStageCannotShadowJudgeAbstention(t *testing.T) {
	raw := strings.Replace(nativeRoutingTestYAML, "name: fast, kind", "name: abstain, kind", 1)
	if _, err := ParseYAMLBytes([]byte(raw)); err == nil {
		t.Fatal("reserved judge selection accepted as a stage")
	}
}
