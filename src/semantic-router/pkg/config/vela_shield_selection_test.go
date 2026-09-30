package config

import "testing"

const velaShieldSelectionPath = "models/Vela-1.0-Encoder-307M-Shield"

func TestVelaShieldSelectsGloballyThroughTheSafetyModule(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(`version: v0.3
routing:
  signals:
    safety:
      - {name: unsafe-content, threshold: 0.5}
global:
  model_catalog:
    modules:
      safety:
        safety:
          model_id: models/Vela-1.0-Encoder-307M-Shield
`))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.SafetyModels.Safety.ModelID != velaShieldSelectionPath || !cfg.SafetyModels.Safety.UseCPU {
		t.Fatalf("safety module resolved to %+v", cfg.SafetyModels.Safety)
	}
	if err := validateSafetySignalContracts(cfg); err != nil {
		t.Fatal(err)
	}
}

func TestVelaShieldBindsToOneRecipeSafetyRule(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(`version: v0.3
entrypoints:
  - model_names: [care]
    recipe: care
recipes:
  - name: care
    routing:
      model_bindings:
        safety.unsafe-content: {deployment: shield, adapter: modernbert, contract: label_distribution.v1}
      signals:
        safety:
          - {name: unsafe-content, threshold: 0.5}
global:
  model_catalog:
    deployments:
      shield:
        artifact: models/Vela-1.0-Encoder-307M-Shield
        revision: a981a99eeb05a2859b88b5cee9af4352897ec4ec
        provider: candle
`))
	if err != nil {
		t.Fatal(err)
	}
	plan, err := CompileModelBindings(cfg)
	if err != nil {
		t.Fatal(err)
	}
	bound, ok := plan.Lookup("care", "safety.unsafe-content")
	if !ok {
		t.Fatal("recipe safety rule did not resolve the Shield deployment")
	}
	shield := GetModelByPath(velaShieldSelectionPath)
	if bound.Deployment.Artifact != shield.LocalPath || bound.Deployment.Revision != shield.Revision || bound.Deployment.Provider != "candle" {
		t.Fatalf("unexpected Shield deployment: %+v", bound.Deployment)
	}
	if bound.Binding.Contract != RemoteClassifierContractLabelDistribution || bound.Binding.Adapter != "modernbert" {
		t.Fatalf("unexpected Shield binding: %+v", bound.Binding)
	}
	if _, leaked := plan.Lookup(DefaultRecipeName, "safety.unsafe-content"); leaked {
		t.Fatal("recipe Shield binding leaked into the default scope")
	}
	if cfg.SafetyModels.Safety.ModelID != DefaultSystemModels().Safety {
		t.Fatalf("recipe binding changed the global safety module to %q", cfg.SafetyModels.Safety.ModelID)
	}
}
