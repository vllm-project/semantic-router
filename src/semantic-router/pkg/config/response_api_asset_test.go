package config

import (
	"reflect"
	"testing"
)

func TestResponseAPIProfileDeclaresTextAndVisionCapabilities(t *testing.T) {
	cfg, err := ParseYAMLBytes(readValuesConfigAsset(t, repoRel("e2e", "profiles", "response-api", "values.yaml")))
	if err != nil {
		t.Fatalf("parse Response API profile: %v", err)
	}
	for model, want := range map[string][]string{
		"openai/gpt-oss-20b":    {"chat"},
		"mock/vision":           {"chat", "image_input"},
		"mock/native-responses": nil,
	} {
		params, ok := cfg.ModelConfig[model]
		if !ok {
			t.Fatalf("Response API profile is missing model %q", model)
		}
		if !reflect.DeepEqual(params.Capabilities, want) {
			t.Errorf("model %q declared capabilities = %v, want %v", model, params.Capabilities, want)
		}
	}
}
