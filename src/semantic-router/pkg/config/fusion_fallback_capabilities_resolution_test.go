/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package config

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func aliasCatalog() map[string]ModelParams {
	return map[string]ModelParams{
		"base-a": {
			Capabilities:     []string{"vision"},
			ExternalModelIDs: map[string]string{"openai": "vendor-a"},
			LoRAs:            []LoRAAdapter{{Name: "tuned-a"}},
		},
		"base-b": {
			Capabilities:     []string{"chat"},
			ExternalModelIDs: map[string]string{"openai": "vendor-b"},
			LoRAs:            []LoRAAdapter{{Name: "tuned-b"}},
		},
	}
}

func TestModelIdentityResolutionPrefersExactCatalogKey(t *testing.T) {
	// base-b also declares "base-a" as an alias. The exact key must still win,
	// so an entry can never be shadowed by another claiming its name.
	catalog := aliasCatalog()
	shadowing := catalog["base-b"]
	shadowing.LoRAs = append(shadowing.LoRAs, LoRAAdapter{Name: "base-a"})
	catalog["base-b"] = shadowing

	identity := resolveModelIdentity(catalog, "base-a")

	require.True(t, identity.resolved())
	assert.Equal(t, "base-a", identity.owner)
	assert.Equal(t, []string{"vision"}, identity.capabilities)
}

func TestModelIdentityResolutionResolvesUniqueAliases(t *testing.T) {
	catalog := aliasCatalog()

	external := resolveModelIdentity(catalog, "vendor-b")
	require.True(t, external.resolved(), "a unique external model ID must resolve")
	assert.Equal(t, "base-b", external.owner)
	assert.Equal(t, []string{"chat"}, external.capabilities)

	lora := resolveModelIdentity(catalog, "tuned-a")
	require.True(t, lora.resolved(), "a unique LoRA name must resolve to its base model")
	assert.Equal(t, "base-a", lora.owner)
	assert.Equal(t, []string{"vision"}, lora.capabilities)
}

func TestModelIdentityResolutionReportsUnknownIdentity(t *testing.T) {
	identity := resolveModelIdentity(aliasCatalog(), "nowhere")

	assert.False(t, identity.resolved())
	assert.Empty(t, identity.ambiguous, "unknown is not the same as ambiguous")

	// An empty catalog is also unresolved rather than vacuously compatible.
	assert.False(t, resolveModelIdentity(nil, "base-a").resolved())
	assert.False(t, resolveModelIdentity(aliasCatalog(), "  ").resolved())
}

// Neither LoRA names nor external IDs are required to be unique across model
// cards, so two entries can claim one runtime identity. Choosing either would
// make a safety decision depend on map iteration order.
func TestModelIdentityResolutionRejectsDuplicateAliases(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(map[string]ModelParams)
		alias   string
		wantOwn []string
	}{
		{
			name: "duplicate external model IDs",
			mutate: func(catalog map[string]ModelParams) {
				params := catalog["base-a"]
				params.ExternalModelIDs = map[string]string{"openai": "shared-vendor"}
				catalog["base-a"] = params
				other := catalog["base-b"]
				other.ExternalModelIDs = map[string]string{"anthropic": "shared-vendor"}
				catalog["base-b"] = other
			},
			alias:   "shared-vendor",
			wantOwn: []string{"base-a", "base-b"},
		},
		{
			name: "duplicate LoRA adapter names",
			mutate: func(catalog map[string]ModelParams) {
				for _, key := range []string{"base-a", "base-b"} {
					params := catalog[key]
					params.LoRAs = append(params.LoRAs, LoRAAdapter{Name: "shared"})
					catalog[key] = params
				}
			},
			alias:   "shared",
			wantOwn: []string{"base-a", "base-b"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			catalog := aliasCatalog()
			test.mutate(catalog)

			identity := resolveModelIdentity(catalog, test.alias)

			require.False(t, identity.resolved(),
				"an ambiguous identity must never be treated as proven")
			// Requiring both owners, sorted, is what makes this independent of map
			// order: a first-match resolver reports one owner whichever entry it
			// happened to see first.
			assert.Equal(t, test.wantOwn, identity.ambiguous)
			assert.Empty(t, identity.owner)
		})
	}
}

// The fallback rule surfaces both failure modes separately, because they need
// different diagnostics: unknown metadata names the identity, ambiguity names
// the competing owners.
func TestResolveFusionFallbackCapabilitiesSeparatesUnknownFromAmbiguous(t *testing.T) {
	catalog := aliasCatalog()
	for _, key := range []string{"base-a", "base-b"} {
		params := catalog[key]
		params.LoRAs = append(params.LoRAs, LoRAAdapter{Name: "shared"})
		catalog[key] = params
	}

	ambiguous := resolveFusionFallbackCapabilities(catalog, []string{"shared"}, "base-a")
	require.Len(t, ambiguous.ambiguous, 1)
	assert.Contains(t, ambiguous.ambiguous[0], "shared")
	assert.Contains(t, ambiguous.ambiguous[0], "base-a, base-b")
	assert.Empty(t, ambiguous.unresolved)

	unknown := resolveFusionFallbackCapabilities(catalog, []string{"base-a"}, "nowhere")
	assert.Equal(t, []string{"nowhere"}, unknown.unresolved)
	assert.Empty(t, unknown.ambiguous)

	// A clean panel and target still produce an ordinary capability comparison.
	clean := resolveFusionFallbackCapabilities(catalog, []string{"base-b"}, "base-a")
	assert.Empty(t, clean.unresolved)
	assert.Empty(t, clean.ambiguous)
	assert.Equal(t, []string{"chat"}, clean.gap.missing,
		"base-a declares vision only, so the panel's chat requirement is missing")
}

// A catalog key and the provider external IDs that entry declares dispatch the
// same model, so falling back from one to another retries what just failed. A
// LoRA adapter is a different executable variant served by the same base, so it
// remains a legitimate fallback source.
func TestExecutionIdentitySeparatesBaseAliasesFromAdapters(t *testing.T) {
	catalog := map[string]ModelParams{
		"base-a": {
			Capabilities:     []string{"chat"},
			ExternalModelIDs: map[string]string{"openai": "vendor-a", "azure": "azure-a"},
			LoRAs:            []LoRAAdapter{{Name: "tuned-a"}},
		},
		"base-b": {Capabilities: []string{"chat"}},
	}

	// Every spelling of base-a resolves to the same execution identity.
	key := resolveModelIdentity(catalog, "base-a")
	external := resolveModelIdentity(catalog, "vendor-a")
	otherExternal := resolveModelIdentity(catalog, "azure-a")
	require.True(t, key.resolved())
	assert.Equal(t, key.execution, external.execution,
		"an external ID dispatches the model that declares it")
	assert.Equal(t, key.execution, otherExternal.execution)

	// The adapter shares capability metadata but not execution identity.
	adapter := resolveModelIdentity(catalog, "tuned-a")
	require.True(t, adapter.resolved())
	assert.Equal(t, key.capabilities, adapter.capabilities,
		"an adapter inherits the base model's declared capabilities")
	assert.NotEqual(t, key.execution, adapter.execution,
		"an adapter is a distinct executable variant")

	// A different catalog entry is distinct even with identical capabilities.
	assert.NotEqual(t, key.execution, resolveModelIdentity(catalog, "base-b").execution)
}

// Alias equivalence is detected through the full resolver, which is what the
// config-load validator consumes.
func TestResolveFusionFallbackCapabilitiesDetectsAliasEquivalence(t *testing.T) {
	catalog := map[string]ModelParams{
		"base-a": {
			Capabilities:     []string{"chat"},
			ExternalModelIDs: map[string]string{"openai": "vendor-a"},
			LoRAs:            []LoRAAdapter{{Name: "tuned-a"}},
		},
		"base-b": {Capabilities: []string{"chat"}},
	}

	// The panel names the target by its provider external ID.
	assert.Equal(t, []string{"vendor-a"},
		resolveFusionFallbackCapabilities(catalog, []string{"vendor-a", "base-b"}, "base-a").
			conflictingPanelModels,
		"an external-ID alias must be recognised as the target itself")

	// And the reverse spelling.
	assert.Equal(t, []string{"base-a"},
		resolveFusionFallbackCapabilities(catalog, []string{"base-a"}, "vendor-a").
			conflictingPanelModels)

	// An adapter owned by the target is a different variant, so it is allowed.
	adapter := resolveFusionFallbackCapabilities(catalog, []string{"tuned-a"}, "base-a")
	assert.Empty(t, adapter.conflictingPanelModels,
		"falling back from an adapter to its base is recovery, not a retry")

	// Distinct owners with identical capabilities are unaffected.
	assert.Empty(t, resolveFusionFallbackCapabilities(catalog, []string{"base-b"}, "base-a").
		conflictingPanelModels)

	// An undeclared panel member is now a rejection rather than a skip, so it is
	// reported instead of quietly leaving the union short.
	assert.Equal(t, []string{"undeclared"},
		resolveFusionFallbackCapabilities(catalog, []string{"undeclared"}, "base-a").unresolved)
}
