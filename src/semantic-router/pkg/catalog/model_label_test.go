package catalog

import "testing"

// A model reads by the name its card gives, the catalog's or the operator's,
// else by the card a provider maps its provider model id to, else by that
// id; a name the configuration does not hold reads as itself.
func TestEffectiveModelLabel(t *testing.T) {
	registry, err := BuiltIn()
	if err != nil {
		t.Fatal(err)
	}
	house := "House Model"
	effective, err := registry.Compile(CompileInput{
		Providers: []ProviderInstance{{Name: "lab", Catalog: "vllm", BaseURL: "http://127.0.0.1:8000/v1"}},
		Models: []ModelAlias{
			{Name: "catalog-backed", Catalog: "qwen/qwen3.6-35b-a3b", Providers: []ModelProviderBinding{{Name: "lab", ModelID: "qwen36"}}},
			{Name: "known-id", Catalog: "known-id", Providers: []ModelProviderBinding{{Name: "lab", ModelID: "Qwen/Qwen3.6-35B-A3B"}}},
			{Name: "unknown-id", Catalog: "unknown-id", Providers: []ModelProviderBinding{{Name: "lab", ModelID: "qwen36"}}},
			{Name: "operator-named", Catalog: "operator-named", Providers: []ModelProviderBinding{{Name: "lab", ModelID: "house-v2"}}},
		},
		ModelCards: []ModelCardOverlay{{Name: "known-id"}, {Name: "unknown-id"}, {Name: "operator-named", DisplayName: &house}},
	})
	if err != nil {
		t.Fatal(err)
	}
	for name, want := range map[string]string{
		"catalog-backed": "Qwen3.6 35B A3B",
		"known-id":       "Qwen3.6 35B A3B",
		"unknown-id":     "qwen36",
		"operator-named": "House Model",
		"not-configured": "not-configured",
	} {
		if got := effective.ModelLabel(name); got != want {
			t.Errorf("ModelLabel(%q) = %q, want %q", name, got, want)
		}
	}
}
