package authoring

import "testing"

func TestEmitCRDFileFromSharedFirstSliceAuthoringFixture(t *testing.T) {
	t.Parallel()

	crdBytes, err := EmitCRDFile(
		sharedFixturePath(t, "td001-first-slice-authoring.yaml"),
		"fixture-router",
		"testing",
	)
	if err != nil {
		t.Fatalf("EmitCRDFile() error = %v", err)
	}

	raw := loadYAMLDocument(t, crdBytes)
	assertAuthoringCRDMetadata(t, raw)

	spec := assertAuthoringCRDSpec(t, raw)
	assertAuthoringCRDConfig(t, spec)
	assertAuthoringCRDVLLMEndpoints(t, spec)
}

func assertAuthoringCRDMetadata(t *testing.T, raw map[string]interface{}) {
	t.Helper()

	assertYAMLEqual(t, map[string]interface{}{
		"apiVersion": "vllm.ai/v1alpha1",
		"kind":       "SemanticRouter",
		"metadata": map[string]interface{}{
			"name":      "fixture-router",
			"namespace": "testing",
		},
	}, selectTopLevelKeys(raw, "apiVersion", "kind", "metadata"))
}

func assertAuthoringCRDSpec(t *testing.T, raw map[string]interface{}) map[string]interface{} {
	t.Helper()

	spec, ok := raw["spec"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected spec map, got %#v", raw["spec"])
	}
	return spec
}

func assertAuthoringCRDConfig(t *testing.T, spec map[string]interface{}) {
	t.Helper()

	configSpec, ok := spec["config"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected spec.config map, got %#v", spec["config"])
	}

	assertYAMLEqual(t, map[string]interface{}{
		"default_model":            "qwen3-4b",
		"default_reasoning_effort": "medium",
		"reasoning_families": map[string]interface{}{
			"qwen3": map[string]interface{}{
				"type":      "effort",
				"parameter": "reasoning_effort",
			},
		},
		"keyword_rules": []interface{}{
			map[string]interface{}{
				"name":           "billing_keywords",
				"operator":       "contains",
				"keywords":       []interface{}{"invoice", "refund"},
				"case_sensitive": false,
			},
		},
	}, selectTopLevelKeys(
		configSpec,
		"default_model",
		"default_reasoning_effort",
		"reasoning_families",
		"keyword_rules",
	))
}

func assertAuthoringCRDVLLMEndpoints(t *testing.T, spec map[string]interface{}) {
	t.Helper()

	vllmEndpoints, ok := spec["vllmEndpoints"].([]interface{})
	if !ok {
		t.Fatalf("expected spec.vllmEndpoints array, got %#v", spec["vllmEndpoints"])
	}
	if len(vllmEndpoints) != 2 {
		t.Fatalf("expected 2 vllmEndpoints entries, got %d", len(vllmEndpoints))
	}

	assertYAMLEqual(t, []interface{}{
		map[string]interface{}{
			"name":            "qwen3-4b_primary",
			"model":           "qwen3-4b",
			"reasoningFamily": "qwen3",
			"weight":          100,
			"backend": map[string]interface{}{
				"type": "service",
				"service": map[string]interface{}{
					"name": "router.internal",
					"port": 8000,
				},
			},
		},
		map[string]interface{}{
			"name":            "qwen3-32b_secure",
			"model":           "qwen3-32b",
			"reasoningFamily": "qwen3",
			"weight":          50,
			"backend": map[string]interface{}{
				"type": "service",
				"service": map[string]interface{}{
					"name": "api.example.com",
					"port": 443,
				},
			},
		},
	}, vllmEndpoints)
}
