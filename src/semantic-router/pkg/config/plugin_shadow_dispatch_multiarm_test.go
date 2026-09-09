package config

import (
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
