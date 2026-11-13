package config

import (
	"strings"
	"testing"
)

func TestRouterReplayShadowModelsDeduplicatesArms(t *testing.T) {
	cfg := RouterReplayShadowConfig{
		Model: "m1",
		Arms:  []string{"  ", "m2", "m1", "m3"},
	}
	got := cfg.ShadowModels()
	want := []string{"m1", "m2", "m3"}
	if len(got) != len(want) {
		t.Fatalf("ShadowModels() = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("ShadowModels()[%d] = %q, want %q (full %v)", i, got[i], want[i], got)
		}
	}
}

func TestRouterReplayShadowModelsArmsOnlyWhenModelEmpty(t *testing.T) {
	cfg := RouterReplayShadowConfig{Arms: []string{"a", "b"}}
	got := cfg.ShadowModels()
	if len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Fatalf("ShadowModels() = %v, want [a b]", got)
	}
}

func TestRouterReplayShadowBudgetValidation(t *testing.T) {
	valid := RouterReplayShadowConfig{Model: "m", Budget: ShadowDispatchBudgetConfig{MaxCallsPerRequest: 2, ReserveTokensPerArm: 128}}
	if err := valid.Validate(); err != nil {
		t.Fatalf("valid budget must pass: %v", err)
	}
	invalid := RouterReplayShadowConfig{
		Model:  "m",
		Budget: ShadowDispatchBudgetConfig{MaxCostPerRequest: -1},
	}
	if err := invalid.Validate(); err == nil {
		t.Fatal("negative budget must fail validation")
	}
}

func TestRouterReplayShadowArmsValidation(t *testing.T) {
	ok := RouterReplayShadowConfig{Model: "m", Arms: []string{"a", "b"}}
	if err := (&ok).Validate(); err != nil {
		t.Fatalf("non-empty arms must pass: %v", err)
	}
	// Empty arms with a real model is still valid (single-arm).
	if err := (&RouterReplayShadowConfig{Model: "m"}).Validate(); err != nil {
		t.Fatalf("single-arm must pass: %v", err)
	}
	// Arms-only (no Model) is valid.
	if err := (&RouterReplayShadowConfig{Arms: []string{"a"}}).Validate(); err != nil {
		t.Fatalf("arms-only must pass: %v", err)
	}
	emptyArm := &RouterReplayShadowConfig{Model: "m", Arms: []string{"", "a"}}
	if err := emptyArm.Validate(); err == nil {
		t.Fatal("empty arm entry must fail validation")
	}
	// Neither model nor arms is a config that can never dispatch.
	if err := (&RouterReplayShadowConfig{}).Validate(); err == nil || !strings.Contains(err.Error(), "model or arms is required") {
		t.Fatalf("empty shadow validation = %v, want model-or-arms rejection", err)
	}
}

// TestRouterReplayShadowEveryArmMustHaveBackend is the full-config regression
// for multi-arm validation (issue #3376, Xunzhuo review): every deduplicated
// ShadowModels() entry must resolve to a backend, not just the legacy Model.
func TestRouterReplayShadowEveryArmMustHaveBackend(t *testing.T) {
	cfg := &RouterConfig{
		BackendModels: BackendModels{
			ModelConfig: map[string]ModelParams{
				"primary": {PreferredEndpoints: []string{"ep"}},
				"arm-a":   {PreferredEndpoints: []string{"ep"}},
				"arm-b":   {PreferredEndpoints: []string{"ep"}},
			},
			VLLMEndpoints: []VLLMEndpoint{{Name: "ep", Address: "127.0.0.1", Port: 8000}},
		},
	}
	replay := func(shadow map[string]interface{}) []DecisionPlugin {
		return []DecisionPlugin{{
			Type: DecisionPluginRouterReplay,
			Configuration: MustStructuredPayload(map[string]interface{}{
				"enabled": true,
				"shadow":  shadow,
			}),
		}}
	}

	// Arms-only config (no legacy Model) with every arm resolvable -> valid.
	decision := &Decision{Name: "route", Plugins: replay(map[string]interface{}{"arms": []string{"arm-a"}})}
	if err := validateDecisionReplayShadow(cfg, decision); err != nil {
		t.Fatalf("arms-only config with resolvable arm rejected: %v", err)
	}

	// One deduplicated arm without a backend -> rejected; the legacy Model
	// path alone would miss this.
	decision.Plugins = replay(map[string]interface{}{"model": "arm-a", "arms": []string{"arm-b", "ghost"}})
	err := validateDecisionReplayShadow(cfg, decision)
	if err == nil || !strings.Contains(err.Error(), "ghost") || !strings.Contains(err.Error(), "no configured backend") {
		t.Fatalf("error = %v, want rejection naming the unknown arm 'ghost'", err)
	}
}
