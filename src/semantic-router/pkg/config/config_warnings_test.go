package config

import (
	"strings"
	"testing"
)

const modalityRoutingYAML = `
version: v0.3
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    model: m
  models:
    - name: m
      backend_refs:
        - endpoint: 127.0.0.1:8000
routing:
  modelCards:
    - name: m
  signals:
    modality:
      - name: AR
        description: Text-only requests.
      - name: DIFFUSION
        description: Image generation requests.
  decisions:
    - name: image-gen
      rules:
        operator: AND
        conditions:
          - type: modality
            name: DIFFUSION
      modelRefs:
        - model: m
`

func modalityWarnings(t *testing.T, document string) []ConfigWarning {
	t.Helper()
	cfg, err := ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatal(err)
	}
	var found []ConfigWarning
	for _, warning := range Warnings(cfg) {
		if warning.Code == warningModalityDetectorOff {
			found = append(found, warning)
		}
	}
	return found
}

// Modality rules without the detector never match, and the configuration is
// valid, so the Router warns and names what uses the signal.
func TestModalityRulesWithoutTheDetectorWarn(t *testing.T) {
	warnings := modalityWarnings(t, modalityRoutingYAML)
	if len(warnings) != 1 {
		t.Fatalf("warnings = %v, want one modality_detector_disabled", warnings)
	}
	for _, want := range []string{
		`decision "image-gen"`,
		`routing.signals.modality "AR"`,
		`routing.signals.modality "DIFFUSION"`,
		"global.model_catalog.modules.modality_detector is not enabled",
		"confidence_threshold: 0.51",
	} {
		if !strings.Contains(warnings[0].Message, want) {
			t.Errorf("warning %q does not say %q", warnings[0].Message, want)
		}
	}
	if warnings[0].Field != "global.model_catalog.modules.modality_detector" {
		t.Errorf("field = %q", warnings[0].Field)
	}
}

func TestAnEnabledModalityDetectorDoesNotWarn(t *testing.T) {
	enabled := modalityRoutingYAML + `
global:
  model_catalog:
    modules:
      modality_detector:
        enabled: true
        method: classifier
        confidence_threshold: 0.51
`
	if warnings := modalityWarnings(t, enabled); len(warnings) != 0 {
		t.Fatalf("warnings = %v, want none with the detector enabled", warnings)
	}
}

func TestARecipeThatUsesModalityWarnsWithItsName(t *testing.T) {
	recipes := `
version: v0.3
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    model: m
  models:
    - name: m
      backend_refs:
        - endpoint: 127.0.0.1:8000
entrypoints:
  - model_names: [vllm-sr/images]
    recipe: images
routing:
  modelCards:
    - name: m
recipes:
  - name: images
    routing:
      signals:
        modality:
          - name: DIFFUSION
            description: Image generation requests.
      decisions:
        - name: draw
          rules:
            operator: AND
            conditions:
              - type: modality
                name: DIFFUSION
          modelRefs:
            - model: m
`
	cfg, err := ParseYAMLBytes([]byte(recipes))
	if err != nil {
		t.Fatal(err)
	}
	var message string
	for _, warning := range Warnings(cfg) {
		if warning.Code == warningModalityDetectorOff {
			message = warning.Message
		}
	}
	if !strings.Contains(message, `recipe "images": decision "draw"`) {
		t.Fatalf("warning %q does not name the recipe's decision", message)
	}
}

// One message lists every problem of an enabled detector, so a single edit
// fixes it instead of one apply per problem.
func TestModalityDetectorReportsAllItsProblemsAtOnce(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.ModalityDetector = ModalityDetectorConfig{Enabled: true, ModalityDetectionConfig: ModalityDetectionConfig{
		Method: ModalityDetectionHybrid,
	}}
	err := validateGlobalModalityContracts(cfg)
	if err == nil {
		t.Fatal("an incomplete hybrid detector must be refused")
	}
	for _, want := range []string{
		"requires at least one of classifier or keywords",
		"confidence_threshold is required",
		"lower_threshold_ratio is required",
	} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not say %q", err, want)
		}
	}

	cfg.ModalityDetector.Method = ""
	err = validateGlobalModalityContracts(cfg)
	if err == nil || !strings.Contains(err.Error(), "method is required") || !strings.Contains(err.Error(), "confidence_threshold") {
		t.Fatalf("a detector without a method must say what each method needs, got %v", err)
	}
}
