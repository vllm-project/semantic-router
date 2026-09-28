package config

import (
	"testing"

	"gopkg.in/yaml.v2"
)

func TestPIIWindowConfigurationPreservesExplicitDocumentBudget(t *testing.T) {
	base := PIIModel{UseMmBERT32K: true, MaxSequenceLength: 32768, Window: &SequenceHeadWindowConfig{Size: 512, Overlap: 255}}
	if err := base.ValidateWindow(); err != nil {
		t.Fatal(err)
	}
	for name, change := range map[string]func(*PIIModel){
		"remote":   func(c *PIIModel) { c.Backend = &RemoteClassifierBackend{} },
		"adapter":  func(c *PIIModel) { c.UseMmBERT32K = false },
		"size":     func(c *PIIModel) { c.Window = &SequenceHeadWindowConfig{Size: 0} },
		"overlap":  func(c *PIIModel) { c.Window = &SequenceHeadWindowConfig{Size: 512, Overlap: 512} },
		"document": func(c *PIIModel) { c.MaxSequenceLength = 128 },
	} {
		t.Run(name, func(t *testing.T) {
			c := base
			change(&c)
			if c.ValidateWindow() == nil {
				t.Fatal("invalid window accepted")
			}
		})
	}
	cfg := &RouterConfig{}
	cfg.PIIModel = base
	cfg.ModelBindings = map[string]ModelBinding{"pii_classifier": {Deployment: "pii"}}
	deployment := ModelDeployment{Provider: "ort", Input: ModelInputBudget{MaxTokens: 1024, Overflow: "window"}}
	cfg.ModelDeployments = map[string]ModelDeployment{"pii": deployment}
	if err := ValidatePIIWindow(cfg); err != nil {
		t.Fatal(err)
	}
	deployment.Input.MaxTokens = 256
	cfg.ModelDeployments["pii"] = deployment
	if ValidatePIIWindow(cfg) == nil {
		t.Fatal("module budget overrode smaller deployment")
	}
	deployment.Input.MaxTokens = 1024
	deployment.Input.Overflow = "reject"
	cfg.ModelDeployments["pii"] = deployment
	if ValidatePIIWindow(cfg) == nil {
		t.Fatal("explicit reject became window")
	}
	cfg.PIIModel.Window = nil
	if err := ValidatePIIWindow(cfg); err != nil {
		t.Fatal(err)
	}
	deployment.Input.Overflow = "window"
	cfg.ModelDeployments["pii"] = deployment
	if ValidatePIIWindow(cfg) == nil {
		t.Fatal("window overflow silently ignored missing geometry")
	}
}

func TestPIIWindowCanonicalModuleRoundTrip(t *testing.T) {
	original := CanonicalPIIModule{PIIModel: PIIModel{UseMmBERT32K: true, MaxSequenceLength: 32768, Window: &SequenceHeadWindowConfig{Size: 512, Overlap: 255}}, ModelRef: "pii_classifier"}
	raw, err := yaml.Marshal(original)
	if err != nil {
		t.Fatal(err)
	}
	var restored CanonicalPIIModule
	if err = yaml.UnmarshalStrict(raw, &restored); err != nil {
		t.Fatal(err)
	}
	if restored.Window == nil || *restored.Window != *original.Window || restored.MaxSequenceLength != 32768 {
		t.Fatalf("lost window: %+v", restored)
	}
}
