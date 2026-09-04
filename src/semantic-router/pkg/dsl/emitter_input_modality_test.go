package dsl

import (
	"testing"

	"gopkg.in/yaml.v3"
)

const inputModalityEmitterDSL = `
MODEL "vision-model" {}

SIGNAL input_modality "image_input" {
  modality: "image"
}

ROUTE "vision-route" {
  PRIORITY 100
  WHEN input_modality("image_input")
  MODEL "vision-model"
}`

// The user YAML emitter must nest input_modality rules under signals, where
// the loader expects every family, instead of leaving them as an unsupported
// top-level key.
func TestEmitUserYAMLNestsInputModalitySignals(t *testing.T) {
	cfg := mustCompilePolicyDSL(t, inputModalityEmitterDSL)
	userYAML, err := EmitUserYAML(cfg)
	if err != nil {
		t.Fatalf("EmitUserYAML error: %v", err)
	}
	var raw map[string]interface{}
	if unmarshalErr := yaml.Unmarshal(userYAML, &raw); unmarshalErr != nil {
		t.Fatalf("emitted YAML is invalid: %v\n%s", unmarshalErr, userYAML)
	}
	if _, leaked := raw["input_modality"]; leaked {
		t.Fatalf("input_modality leaked as a top-level key:\n%s", userYAML)
	}
	signals, _ := raw["signals"].(map[string]interface{})
	rules, _ := signals["input_modality"].([]interface{})
	if len(rules) != 1 {
		t.Fatalf("signals.input_modality = %v, want one rule:\n%s", signals["input_modality"], userYAML)
	}
	rule, _ := rules[0].(map[string]interface{})
	if rule["name"] != "image_input" || rule["modality"] != "image" {
		t.Fatalf("signals.input_modality[0] = %v, want image_input/image", rule)
	}
}

// The CRD emitter must carry the rule declaration alongside the decision that
// references it so the CR stays self-contained.
func TestEmitCRDCarriesInputModalitySignals(t *testing.T) {
	cfg := mustCompilePolicyDSL(t, inputModalityEmitterDSL)
	crd, err := EmitCRD(cfg, "demo", "default")
	if err != nil {
		t.Fatalf("EmitCRD error: %v", err)
	}
	var raw map[string]interface{}
	if unmarshalErr := yaml.Unmarshal(crd, &raw); unmarshalErr != nil {
		t.Fatalf("emitted CRD is invalid: %v\n%s", unmarshalErr, crd)
	}
	spec, _ := raw["spec"].(map[string]interface{})
	configSpec, _ := spec["config"].(map[string]interface{})
	rules, _ := configSpec["input_modality"].([]interface{})
	if len(rules) != 1 {
		t.Fatalf("spec.config.input_modality = %v, want one rule:\n%s", configSpec["input_modality"], crd)
	}
}
