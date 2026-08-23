package providercatalog

import "testing"

func TestBuiltinIntegrationsAreCompleteValidCatalogData(t *testing.T) {
	options := testRegistryOptions()
	options.Integrations = BuiltinIntegrations()
	registry, err := NewRegistry(options)
	if err != nil {
		t.Fatal(err)
	}
	snapshot := registry.Snapshot()
	if got := len(snapshot.List()); got != 40 {
		t.Fatalf("built-in provider count = %d, want 40", got)
	}
	for _, providerID := range []string{"vllm", "sglang", "amd-atom", "openai-compatible", "anthropic"} {
		if _, found := snapshot.Get(providerID); !found {
			t.Fatalf("built-in provider %q is missing", providerID)
		}
	}
	atom, found := snapshot.Get("amd-atom")
	if !found || atom.Origin.Mode != OriginUserSupplied || atom.Origin.DefaultURL != "" {
		t.Fatalf("AMD ATOM must use the operator's local endpoint, got %+v", atom.Origin)
	}
	for _, provider := range snapshot.List() {
		if provider.Origin.Mode == OriginFixed && provider.Origin.DefaultURL == "" {
			t.Fatalf("fixed provider %q has no default URL", provider.ID)
		}
	}
	for _, providerID := range []string{"vllm", "anthropic"} {
		provider, found := snapshot.Get(providerID)
		if !found || !hasBuiltinCapability(provider.Capabilities, "text") {
			t.Fatalf("built-in provider %q does not advertise text", providerID)
		}
		for _, providerInterface := range provider.Interfaces {
			if !hasBuiltinCapability(providerInterface.Capabilities, "text") {
				t.Fatalf(
					"built-in provider %q interface %q does not advertise text",
					providerID, providerInterface.ID,
				)
			}
		}
	}
}

func hasBuiltinCapability(capabilities []string, expected string) bool {
	for _, capability := range capabilities {
		if capability == expected {
			return true
		}
	}
	return false
}
