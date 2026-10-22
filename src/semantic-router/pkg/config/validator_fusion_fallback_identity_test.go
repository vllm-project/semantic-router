package config

import (
	"strings"
	"testing"
)

// A configured panel that names the fallback target by one of its provider
// external IDs must fail validation.
//
// Spelling equality alone lets a recipe route around the guard: the strings
// differ, capabilities match because both resolve to the same entry, and the
// recovery call then retries the provider model that just failed in the panel.
func TestFusionFallbackTargetRejectedWhenPanelNamesItsExternalID(t *testing.T) {
	cfg, decision := fusionFallbackTestConfig(&FusionAlgorithmConfig{
		QuorumFailurePolicy:  FusionQuorumFailurePolicyFallback,
		QuorumFallbackTarget: "backup-model",
		AnalysisModels:       []string{"vendor-backup", "panel-a"},
	})
	cfg.ModelConfig["backup-model"] = ModelParams{
		ExternalModelIDs: map[string]string{"openai": "vendor-backup"},
	}

	err := validateDecisionFusionFallbackTarget(cfg, decision)
	if err == nil {
		t.Fatal("validateDecisionFusionFallbackTarget() = nil, want an alias-conflict error")
	}
	if !strings.Contains(err.Error(), "dispatches the same model") {
		t.Fatalf("error = %v, want it to report an execution-identity conflict", err)
	}
	if !strings.Contains(err.Error(), "vendor-backup") {
		t.Fatalf("error = %v, want the conflicting panel identity named", err)
	}
}

// A LoRA adapter served by the fallback target is a different executable
// variant, so falling back from it to its base remains valid.
func TestFusionFallbackTargetAcceptsPanelAdapterOwnedByTarget(t *testing.T) {
	cfg, decision := fusionFallbackTestConfig(&FusionAlgorithmConfig{
		QuorumFailurePolicy:  FusionQuorumFailurePolicyFallback,
		QuorumFallbackTarget: "backup-model",
		AnalysisModels:       []string{"tuned-backup", "panel-a"},
	})
	cfg.ModelConfig["backup-model"] = ModelParams{
		LoRAs: []LoRAAdapter{{Name: "tuned-backup"}},
	}

	if err := validateDecisionFusionFallbackTarget(cfg, decision); err != nil {
		t.Fatalf("an adapter panel member may fall back to its base model, got %v", err)
	}
}

// A LoRA panel member contributes its base model's capabilities to the union
// the fallback target must cover.
//
// A direct catalog lookup omits the alias entirely, understating the requirement
// and letting an incompatible target pass recipe validation.
func TestFusionFallbackTargetRejectedWhenPanelAliasRaisesRequirement(t *testing.T) {
	cfg, decision := fusionFallbackTestConfig(&FusionAlgorithmConfig{
		QuorumFailurePolicy:  FusionQuorumFailurePolicyFallback,
		QuorumFallbackTarget: "backup-model",
		AnalysisModels:       []string{"vision-adapter"},
	})
	cfg.ModelConfig["panel-base"] = ModelParams{
		Capabilities: []string{"vision"},
		LoRAs:        []LoRAAdapter{{Name: "vision-adapter"}},
	}
	cfg.ModelConfig["backup-model"] = ModelParams{Capabilities: []string{"chat"}}

	err := validateDecisionFusionFallbackTarget(cfg, decision)
	if err == nil {
		t.Fatal("a chat-only target must not cover a panel requiring vision")
	}
	if !strings.Contains(err.Error(), "vision") {
		t.Fatalf("error = %v, want the uncovered capability named", err)
	}
}

// The same for a provider external ID naming a panel model.
func TestFusionFallbackTargetRejectedWhenPanelExternalIDRaisesRequirement(t *testing.T) {
	cfg, decision := fusionFallbackTestConfig(&FusionAlgorithmConfig{
		QuorumFailurePolicy:  FusionQuorumFailurePolicyFallback,
		QuorumFallbackTarget: "backup-model",
		AnalysisModels:       []string{"vendor-panel"},
	})
	cfg.ModelConfig["panel-base"] = ModelParams{
		Capabilities:     []string{"vision"},
		ExternalModelIDs: map[string]string{"openai": "vendor-panel"},
	}
	cfg.ModelConfig["backup-model"] = ModelParams{Capabilities: []string{"chat"}}

	if err := validateDecisionFusionFallbackTarget(cfg, decision); err == nil {
		t.Fatal("an external-ID panel member must contribute its capabilities")
	}
}

// A fallback recipe must supply metadata that resolves for every panel entry.
// An entirely unresolved panel is the case worth naming: the capability union is
// empty, so any declared target would otherwise look compatible.
func TestFusionFallbackRecipeRejectsUnprovablePanelIdentities(t *testing.T) {
	cases := []struct {
		name       string
		panel      []string
		mutate     func(*RouterConfig)
		wantReason string
		wantNamed  string
	}{
		{
			name:       "undeclared configured panel member",
			panel:      []string{"panel-a", "ghost-model"},
			wantReason: "no model metadata resolves",
			wantNamed:  "ghost-model",
		},
		{
			name:  "every configured panel member undeclared",
			panel: []string{"ghost-one", "ghost-two"},
			// Without rejection the union would be empty and the target would
			// appear compatible with a panel nothing is known about.
			wantReason: "no model metadata resolves",
			wantNamed:  "ghost-one",
		},
		{
			name:  "ambiguously aliased configured panel member",
			panel: []string{"panel-a", "shared-alias"},
			mutate: func(cfg *RouterConfig) {
				cfg.ModelConfig["alias-owner-a"] = ModelParams{
					Capabilities: []string{"chat"},
					LoRAs:        []LoRAAdapter{{Name: "shared-alias"}},
				}
				cfg.ModelConfig["alias-owner-b"] = ModelParams{
					Capabilities: []string{"chat"},
					LoRAs:        []LoRAAdapter{{Name: "shared-alias"}},
				}
			},
			wantReason: "model metadata is ambiguous",
			wantNamed:  "alias-owner-a, alias-owner-b",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg, decision := fusionFallbackTestConfig(&FusionAlgorithmConfig{
				QuorumFailurePolicy:  FusionQuorumFailurePolicyFallback,
				QuorumFallbackTarget: "backup-model",
				AnalysisModels:       tc.panel,
			})
			if tc.mutate != nil {
				tc.mutate(cfg)
			}

			err := validateDecisionFusionFallbackTarget(cfg, decision)
			if err == nil {
				t.Fatal("validateDecisionFusionFallbackTarget() = nil, want a rejection")
			}
			if !strings.Contains(err.Error(), tc.wantReason) {
				t.Fatalf("error = %v, want reason %q", err, tc.wantReason)
			}
			if !strings.Contains(err.Error(), tc.wantNamed) {
				t.Fatalf("error = %v, want %q named", err, tc.wantNamed)
			}
		})
	}
}

// A modelRefs panel dispatches its LoRA adapter, so falling back to the
// adapter's base remains valid.
func TestFusionFallbackTargetAcceptsModelRefAdapterOwnedByTarget(t *testing.T) {
	cfg, decision := fusionFallbackTestConfig(&FusionAlgorithmConfig{
		QuorumFailurePolicy:  FusionQuorumFailurePolicyFallback,
		QuorumFallbackTarget: "backup-model",
	})
	decision.ModelRefs = []ModelRef{
		{Model: "backup-model", LoRAName: "tuned-backup"},
		{Model: "panel-a"},
	}
	cfg.ModelConfig["backup-model"] = ModelParams{
		Capabilities: []string{"chat"},
		LoRAs:        []LoRAAdapter{{Name: "tuned-backup"}},
	}
	cfg.ModelConfig["panel-a"] = ModelParams{Capabilities: []string{"chat"}}

	if err := validateDecisionFusionFallbackTarget(cfg, decision); err != nil {
		t.Fatalf("a ref-expressed adapter panel may fall back to its base, got %v", err)
	}
}

// The base itself in a modelRefs panel is still a conflict: without a
// lora_name the ref dispatches the base, which is the target.
func TestFusionFallbackTargetRejectsModelRefNamingTargetDirectly(t *testing.T) {
	cfg, decision := fusionFallbackTestConfig(&FusionAlgorithmConfig{
		QuorumFailurePolicy:  FusionQuorumFailurePolicyFallback,
		QuorumFallbackTarget: "backup-model",
	})
	decision.ModelRefs = []ModelRef{{Model: "backup-model"}, {Model: "panel-a"}}
	cfg.ModelConfig["backup-model"] = ModelParams{Capabilities: []string{"chat"}}
	cfg.ModelConfig["panel-a"] = ModelParams{Capabilities: []string{"chat"}}

	err := validateDecisionFusionFallbackTarget(cfg, decision)
	if err == nil {
		t.Fatal("a ref naming the target directly must still be refused")
	}
	if !strings.Contains(err.Error(), "analysis models") {
		t.Fatalf("error = %v, want the panel-membership rejection", err)
	}
}
