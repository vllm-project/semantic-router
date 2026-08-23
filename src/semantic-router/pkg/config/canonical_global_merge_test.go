package config

import "testing"

func TestCanonicalGlobalSparseNestedOverridePreservesServiceDefaults(t *testing.T) {
	override := MustStructuredPayload(map[string]interface{}{
		"services": map[string]interface{}{
			"observability": map[string]interface{}{
				"tracing": map[string]interface{}{"enabled": false},
			},
		},
	})
	resolved, err := mergeCanonicalGlobalDefaults(DefaultCanonicalGlobal(), nil, override)
	if err != nil {
		t.Fatal(err)
	}
	if resolved.Services.BackendDispatch != DefaultBackendDispatchConfig() {
		t.Fatalf("backend dispatch defaults were erased: %+v", resolved.Services.BackendDispatch)
	}
	if resolved.Services.ManagementAPI.BindAddress != DefaultManagementAPIConfig().BindAddress {
		t.Fatalf("management listener default was erased: %+v", resolved.Services.ManagementAPI)
	}
	if resolved.Services.Observability.Tracing.Enabled {
		t.Fatal("explicit nested false override was not preserved")
	}
}

func TestCanonicalGlobalSparseOverrideReplacesCollectionsAndExplicitZeroValues(t *testing.T) {
	override := MustStructuredPayload(map[string]interface{}{
		"router": map[string]interface{}{
			"clear_route_cache": false,
		},
		"integrations": map[string]interface{}{
			"tools": map[string]interface{}{"enabled": true, "top_k": 0},
		},
	})
	resolved, err := mergeCanonicalGlobalDefaults(DefaultCanonicalGlobal(), nil, override)
	if err != nil {
		t.Fatal(err)
	}
	if resolved.Router.ClearRouteCache {
		t.Fatalf("explicit boolean overrides changed: %+v", resolved.Router)
	}
	if !resolved.Integrations.Tools.Enabled || resolved.Integrations.Tools.TopK != 0 {
		t.Fatalf("explicit integration values changed: %+v", resolved.Integrations.Tools)
	}
	if resolved.Services.BackendDispatch.BindAddress == "" {
		t.Fatal("unrelated service defaults were erased")
	}
}
