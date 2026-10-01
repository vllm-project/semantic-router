package config

import (
	"strings"
	"testing"
)

func TestShadowModelsDeduplicatesArms(t *testing.T) {
	cfg := ShadowDispatchPluginConfig{
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

func TestShadowModelsArmsOnlyWhenModelEmpty(t *testing.T) {
	cfg := ShadowDispatchPluginConfig{Arms: []string{"a", "b"}}
	got := cfg.ShadowModels()
	if len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Fatalf("ShadowModels() = %v, want [a b]", got)
	}
}

func TestShadowDispatchBudgetValidation(t *testing.T) {
	valid := ShadowDispatchPluginConfig{Model: "m", Budget: ShadowDispatchBudgetConfig{MaxCallsPerRequest: 2, ReserveTokensPerArm: 128}}
	if err := valid.Validate(); err != nil {
		t.Fatalf("valid budget must pass: %v", err)
	}
	invalid := ShadowDispatchPluginConfig{
		Model:  "m",
		Budget: ShadowDispatchBudgetConfig{MaxCostPerRequest: -1},
	}
	if err := invalid.Validate(); err == nil {
		t.Fatal("negative budget must fail validation")
	}
}

func TestShadowDispatchArmsValidation(t *testing.T) {
	ok := ShadowDispatchPluginConfig{Model: "m", Arms: []string{"a", "b"}}
	if err := (&ok).Validate(); err != nil {
		t.Fatalf("non-empty arms must pass: %v", err)
	}
	// Empty arms with a real model is still valid (single-arm).
	if err := (&ShadowDispatchPluginConfig{Model: "m"}).Validate(); err != nil {
		t.Fatalf("single-arm must pass: %v", err)
	}
	// Arms-only (no Model) is valid.
	if err := (&ShadowDispatchPluginConfig{Arms: []string{"a"}}).Validate(); err != nil {
		t.Fatalf("arms-only must pass: %v", err)
	}
	emptyArm := &ShadowDispatchPluginConfig{Model: "m", Arms: []string{"", "a"}}
	if err := emptyArm.Validate(); err == nil {
		t.Fatal("empty arm entry must fail validation")
	}
}

// TestShadowDispatchEveryArmMustHaveBackend is the full-config regression for
// multi-arm validation (issue #3376, Xunzhuo review): every deduplicated
// ShadowModels() entry must resolve to a backend, not just the legacy Model.
func TestShadowDispatchEveryArmMustHaveBackend(t *testing.T) {
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

	// Arms-only config (no legacy Model) with every arm resolvable -> valid.
	decision := &Decision{
		Name: "route",
		Plugins: []DecisionPlugin{{
			Type: DecisionPluginShadowDispatch,
			Configuration: MustStructuredPayload(map[string]interface{}{
				"enabled": true,
				"arms":    []string{"arm-a"},
			}),
		}},
	}
	if err := validateDecisionShadowDispatchPlugin(cfg, decision); err != nil {
		t.Fatalf("arms-only config with resolvable arm rejected: %v", err)
	}

	// One deduplicated arm without a backend -> rejected; the legacy Model
	// path alone would miss this.
	decision.Plugins[0].Configuration = MustStructuredPayload(map[string]interface{}{
		"enabled": true,
		"model":   "arm-a",
		"arms":    []string{"arm-b", "ghost"},
	})
	err := validateDecisionShadowDispatchPlugin(cfg, decision)
	if err == nil || !strings.Contains(err.Error(), "ghost") || !strings.Contains(err.Error(), "no configured backend") {
		t.Fatalf("error = %v, want rejection naming the unknown arm 'ghost'", err)
	}
}
