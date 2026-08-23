package config

import (
	"fmt"
	"strings"
	"testing"
)

func responseCacheRecipeFixture(pluginType, configuration string) []byte {
	configuration = strings.ReplaceAll(strings.TrimSpace(configuration), "\n", "\n          ")
	document := fmt.Sprintf(`
decisions:
  - name: route
    priority: 1
    rules:
      operator: AND
      conditions: []
    plugins:
      - type: %s
        configuration:
          %s
`, pluginType, configuration)
	return canonicalRecipeFixture(document, "")
}

func TestResponseCacheUsesCanonicalIdentifier(t *testing.T) {
	canonical := responseCacheRecipeFixture(DecisionPluginResponseCache, `enabled: true
semantic:
  similarity_threshold: 0.9`)
	if _, err := testAuthoringParser(t).ParseYAMLBytes(canonical); err != nil {
		t.Fatalf("canonical response_cache plugin rejected: %v", err)
	}

	for _, alias := range []string{"semantic-cache", "semantic_cache", "response-cache"} {
		t.Run(alias, func(t *testing.T) {
			input := responseCacheRecipeFixture(alias, "enabled: true")
			if _, err := testAuthoringParser(t).ParseYAMLBytes(input); err == nil {
				t.Fatalf("removed plugin identifier %q was accepted", alias)
			}
		})
	}
}

func TestResponseCacheRejectsRemovedFlatFields(t *testing.T) {
	for name, field := range map[string]string{
		"similarity threshold":  "similarity_threshold: 0.9",
		"request controls flag": "allow_request_controls: true",
		"control header":        "control_header: x-vsr-cache-control",
	} {
		t.Run(name, func(t *testing.T) {
			input := responseCacheRecipeFixture(
				DecisionPluginResponseCache,
				"enabled: true\n"+field,
			)
			if _, err := testAuthoringParser(t).ParseYAMLBytes(input); err == nil {
				t.Fatalf("removed response_cache field %q was accepted", field)
			}
		})
	}
}

func TestResponseCacheRejectsRemovedStoreAlias(t *testing.T) {
	input := []byte(`
version: v0.4
global:
  control_plane:
    mode: managed
  stores:
    semantic_cache:
      enabled: true
`)
	if _, err := ParseYAMLBytes(input); err == nil {
		t.Fatal("global.stores.semantic_cache was accepted")
	}
}
