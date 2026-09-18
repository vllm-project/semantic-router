package config

import (
	"fmt"
	"reflect"
	"testing"

	"gopkg.in/yaml.v2"
)

const windowDocumentBudgetYAML = `version: v0.3
routing:
  model_bindings:
    prompt_guard: {deployment: scanner, adapter: modernbert, contract: label_distribution.v1}
    pii_classifier: {deployment: scanner, adapter: modernbert, contract: token_spans.v1}
    safety.unsafe: {deployment: scanner, adapter: modernbert, contract: label_distribution.v1}
  signals:
    safety:
      - {name: unsafe, threshold: 0.5}
global:
  model_catalog:
    deployments:
      scanner:
        artifact: models/custom-scanner
        provider: %s
        input: {max_tokens: 65536, overflow: window}
    modules:
      prompt_guard:
        variant: mmbert32k
        max_sequence_length: 512
        window: {size: 32768, overlap: 256}
      classifier:
        pii:
          use_mmbert_32k: true
          max_sequence_length: 512
          window: {size: 32768, overlap: 256}
      safety:
        safety:
          max_sequence_length: 512
          window: {size: 32768, overlap: 256}
`

func parseWindowDocumentBudgetConfig(t *testing.T, provider string) *RouterConfig {
	t.Helper()
	cfg, err := ParseYAMLBytes([]byte(fmt.Sprintf(windowDocumentBudgetYAML, provider)))
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

func TestWindowDocumentBudgetCanonicalRoundTrip(t *testing.T) {
	for _, provider := range []string{"candle", "ort"} {
		t.Run(provider, func(t *testing.T) {
			cfg := parseWindowDocumentBudgetConfig(t, provider)
			assertWindowDocumentBudget(t, cfg)
			encoded, err := yaml.Marshal(CanonicalConfigFromRouterConfig(cfg))
			if err != nil {
				t.Fatal(err)
			}
			restored, err := ParseYAMLBytes(encoded)
			if err != nil {
				t.Fatal(err)
			}
			assertWindowDocumentBudget(t, restored)
			if !reflect.DeepEqual(cfg.ModelDeployments["scanner"], restored.ModelDeployments["scanner"]) {
				t.Fatal("canonical round trip changed the document budget")
			}
		})
	}
}

func assertWindowDocumentBudget(t *testing.T, cfg *RouterConfig) {
	t.Helper()
	plan, err := CompileModelBindings(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, consumer := range []string{"prompt_guard", "pii_classifier", "safety.unsafe"} {
		bound, ok := plan.Lookup(DefaultRecipeName, consumer)
		if !ok || bound.Deployment.Input != (ModelInputBudget{MaxTokens: 65536, Overflow: "window"}) {
			t.Fatalf("%s lost its complete-document budget: %+v", consumer, bound)
		}
	}
	for name, validate := range windowDocumentValidators() {
		if err := validate(cfg); err != nil {
			t.Fatalf("%s rejected a 64K document budget with a 32K window: %v", name, err)
		}
	}
	for _, window := range []*SequenceHeadWindowConfig{cfg.PromptGuard.Window, cfg.PIIModel.Window, cfg.SafetyModels.Safety.Window} {
		if window == nil || *window != (SequenceHeadWindowConfig{Size: 32768, Overlap: 256}) {
			t.Fatalf("window geometry changed: %+v", window)
		}
	}
	if cfg.PromptGuard.MaxSequenceLength != 512 || cfg.PIIModel.MaxSequenceLength != 512 || cfg.SafetyModels.Safety.MaxSequenceLength != 512 {
		t.Fatal("bound document budget mutated the module defaults")
	}
}

func windowDocumentValidators() map[string]func(*RouterConfig) error {
	return map[string]func(*RouterConfig) error{
		"guard":  validatePromptGuardBackend,
		"pii":    ValidatePIIWindow,
		"safety": validateSafetySignalContracts,
	}
}

func TestWindowDocumentBudgetWithoutBindings(t *testing.T) {
	cfg := parseWindowDocumentBudgetConfig(t, "candle")
	cfg.ModelBindings = nil
	cfg.PromptGuard.MaxSequenceLength = 65536
	cfg.PIIModel.MaxSequenceLength = 65536
	cfg.SafetyModels.Safety.MaxSequenceLength = 65536
	cfg.SafetyModels.Safety.ModelID = "models/custom-safety"
	for name, validate := range windowDocumentValidators() {
		if err := validate(cfg); err != nil {
			t.Errorf("%s rejected a module document budget larger than its window: %v", name, err)
		}
	}
}

func TestWindowDocumentBudgetRejectsInvalidGeometry(t *testing.T) {
	for _, test := range []struct {
		name   string
		budget int
		window SequenceHeadWindowConfig
	}{
		{"negative document", -1, SequenceHeadWindowConfig{Size: 32768, Overlap: 256}},
		{"negative window", 65536, SequenceHeadWindowConfig{Size: -1}},
		{"zero window", 65536, SequenceHeadWindowConfig{}},
		{"window exceeds document", 65536, SequenceHeadWindowConfig{Size: 65537}},
		{"negative overlap", 65536, SequenceHeadWindowConfig{Size: 32768, Overlap: -1}},
		{"window cannot advance", 65536, SequenceHeadWindowConfig{Size: 32768, Overlap: 32768}},
	} {
		t.Run(test.name, func(t *testing.T) {
			cfg := parseWindowDocumentBudgetConfig(t, "candle")
			deployment := cfg.ModelDeployments["scanner"]
			deployment.Input.MaxTokens = test.budget
			cfg.ModelDeployments["scanner"] = deployment
			cfg.PromptGuard.Window, cfg.PIIModel.Window, cfg.SafetyModels.Safety.Window = &test.window, &test.window, &test.window
			for name, validate := range windowDocumentValidators() {
				if err := validate(cfg); err == nil {
					t.Errorf("%s accepted invalid window geometry or document budget", name)
				}
			}
			encoded, err := yaml.Marshal(CanonicalConfigFromRouterConfig(cfg))
			if err != nil {
				t.Fatal(err)
			}
			if _, err := ParseYAMLBytes(encoded); err == nil {
				t.Fatal("canonical loader accepted invalid window geometry or document budget")
			}
		})
	}
}

func TestWholeInputPoliciesRemainExplicitAfterCanonicalRoundTrip(t *testing.T) {
	for _, overflow := range []string{"reject", "truncate"} {
		t.Run(overflow, func(t *testing.T) {
			cfg := parseWindowDocumentBudgetConfig(t, "candle")
			deployment := cfg.ModelDeployments["scanner"]
			deployment.Input = ModelInputBudget{MaxTokens: 8192, Overflow: overflow}
			cfg.ModelDeployments["scanner"] = deployment
			cfg.PromptGuard.Window, cfg.PIIModel.Window, cfg.SafetyModels.Safety.Window = nil, nil, nil
			encoded, err := yaml.Marshal(CanonicalConfigFromRouterConfig(cfg))
			if err != nil {
				t.Fatal(err)
			}
			restored, err := ParseYAMLBytes(encoded)
			if err != nil {
				t.Fatal(err)
			}
			if restored.ModelDeployments["scanner"].Input != deployment.Input {
				t.Fatal("explicit whole-input policy changed")
			}
			if restored.PromptGuard.Window != nil || restored.PIIModel.Window != nil || restored.SafetyModels.Safety.Window != nil {
				t.Fatal("explicit whole-input policy acquired implicit windows")
			}
		})
	}
}
