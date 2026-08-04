package config

import "testing"

// classifierOverride resolves a global override that touches the classifier modules,
// exercising the same defaults merge the loader uses.
func classifierOverride(t *testing.T, domain, pii map[string]interface{}) CanonicalClassifierModule {
	t.Helper()
	payload, err := NewStructuredPayload(map[string]interface{}{
		"model_catalog": map[string]interface{}{
			"modules": map[string]interface{}{
				"classifier": map[string]interface{}{"domain": domain, "pii": pii},
			},
		},
	})
	if err != nil {
		t.Fatalf("build override: %v", err)
	}
	global, err := resolveCanonicalGlobal(nil, payload)
	if err != nil {
		t.Fatalf("resolve global: %v", err)
	}
	return global.ModelCatalog.Modules.Classifier
}

// An override that never mentions enabled must not turn the classifiers off, since
// the merge writes the override over DefaultCanonicalGlobal().
func TestClassifierEnabledDefaultsToOn(t *testing.T) {
	c := classifierOverride(t,
		map[string]interface{}{"threshold": 0.7},
		map[string]interface{}{"threshold": 0.8})

	if !c.Domain.Enabled {
		t.Error("domain classifier disabled by an override that only set threshold")
	}
	if !c.PII.Enabled {
		t.Error("pii classifier disabled by an override that only set threshold")
	}
}

// enabled: false was previously parsed and dropped, leaving both classifiers running.
func TestClassifierEnabledHonoursExplicitFalse(t *testing.T) {
	c := classifierOverride(t,
		map[string]interface{}{"enabled": false},
		map[string]interface{}{"enabled": false})

	if c.Domain.Enabled {
		t.Error("domain classifier still enabled after enabled: false")
	}
	if c.PII.Enabled {
		t.Error("pii classifier still enabled after enabled: false")
	}
}

// enabled must gate the runtime predicates without implying a model is configured.
func TestClassifierEnabledGatesRuntimePredicates(t *testing.T) {
	tests := []struct {
		name    string
		enabled bool
		modelID string
		want    bool
	}{
		{"enabled and configured", true, "model", true},
		{"disabled but configured", false, "model", false},
		{"enabled without a model", true, "", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &RouterConfig{}
			cfg.PIIModel = PIIModel{Enabled: tt.enabled, ModelID: tt.modelID, PIIMappingPath: "pii.json"}
			cfg.CategoryModel = CategoryModel{Enabled: tt.enabled, ModelID: tt.modelID, CategoryMappingPath: "cat.json"}

			if got := cfg.IsPIIClassifierEnabled(); got != tt.want {
				t.Errorf("IsPIIClassifierEnabled() = %v, want %v", got, tt.want)
			}
			if got := cfg.IsCategoryClassifierEnabled(); got != tt.want {
				t.Errorf("IsCategoryClassifierEnabled() = %v, want %v", got, tt.want)
			}
		})
	}
}
