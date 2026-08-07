package config

import (
	"testing"

	"gopkg.in/yaml.v2"
)

// TestClassifierEnabledFlag covers the tri-state: absent must keep the historical
// behaviour, because adding a plain bool would have disabled PII classification for
// every config that never mentioned the field.
func TestClassifierEnabledFlag(t *testing.T) {
	tests := []struct {
		name        string
		enabled     *bool
		wantPII     bool
		wantCategor bool
	}{
		{"absent keeps the model-configured default", nil, true, true},
		{"explicitly enabled", boolPtr(true), true, true},
		{"explicitly disabled", boolPtr(false), false, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &RouterConfig{}
			cfg.PIIModel = PIIModel{Enabled: tt.enabled, ModelID: "pii", PIIMappingPath: "pii.json"}
			cfg.CategoryModel = CategoryModel{Enabled: tt.enabled, ModelID: "cat", CategoryMappingPath: "cat.json"}

			if got := cfg.IsPIIClassifierEnabled(); got != tt.wantPII {
				t.Errorf("IsPIIClassifierEnabled() = %v, want %v", got, tt.wantPII)
			}
			if got := cfg.IsCategoryClassifierEnabled(); got != tt.wantCategor {
				t.Errorf("IsCategoryClassifierEnabled() = %v, want %v", got, tt.wantCategor)
			}
		})
	}
}

// TestClassifierEnabledDoesNotOverrideMissingConfig keeps enabled from implying a
// model exists: enabled: true with nothing configured is still off.
func TestClassifierEnabledDoesNotOverrideMissingConfig(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.PIIModel = PIIModel{Enabled: boolPtr(true)}
	cfg.CategoryModel = CategoryModel{Enabled: boolPtr(true)}

	if cfg.IsPIIClassifierEnabled() {
		t.Error("IsPIIClassifierEnabled() = true with no model_id or mapping path")
	}
	if cfg.IsCategoryClassifierEnabled() {
		t.Error("IsCategoryClassifierEnabled() = true with no model_id or mapping path")
	}
}

// TestClassifierEnabledParsesFromYAML proves the field reaches the struct: before
// this change `enabled` under these modules was an unknown field and was dropped.
func TestClassifierEnabledParsesFromYAML(t *testing.T) {
	raw := []byte(`
model_id: pii-model
pii_mapping_path: pii.json
enabled: false
`)
	var m PIIModel
	if err := yaml.Unmarshal(raw, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if m.Enabled == nil {
		t.Fatal("enabled was dropped during unmarshal")
	}
	if *m.Enabled {
		t.Errorf("enabled = true, want false")
	}
	if m.Active() {
		t.Error("Active() = true for an explicitly disabled module")
	}
}
