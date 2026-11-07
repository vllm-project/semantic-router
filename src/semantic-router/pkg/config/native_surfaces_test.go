package config

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
)

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
		{"retention", func(_ *RoutingProfile, d *Decision) { d.Retention = &DecisionRetention{} }},
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
	raw := strings.Replace(nativeRoutingTestYAML, "      decisions:", "      signals:\n        metadata:\n          - {name: tenant, key: tenant, predicate: {equals: private}}\n      decisions:", 1)
	raw = strings.Replace(raw, "rules: {}", "rules: {type: metadata, name: tenant}", 1)
	if _, err := ParseYAMLBytes([]byte(raw)); err == nil || !strings.Contains(err.Error(), "native System One") {
		t.Fatalf("native metadata rule silently accepted: %v", err)
	}
}

func TestNativeStageCannotShadowJudgeAbstention(t *testing.T) {
	raw := strings.Replace(nativeRoutingTestYAML, "name: fast, model", "name: abstain, model", 1)
	if _, err := ParseYAMLBytes([]byte(raw)); err == nil {
		t.Fatal("reserved judge selection accepted as a stage")
	}
}
