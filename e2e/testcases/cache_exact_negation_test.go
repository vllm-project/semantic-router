package testcases

import (
	"os"
	"testing"

	"gopkg.in/yaml.v3"
)

// The runtime E2E proves the exact policy's request behavior. This assertion
// keeps the public examples on that policy; high-recall remains an explicit
// semantic-cache opt-in and is intentionally outside this list.
func TestShippedCacheExamplesRequireExactMatching(t *testing.T) {
	for _, sample := range []struct {
		path       string
		cacheCount int
	}{
		{"../../config/config.yaml", 1},
		{"../../config/recipes/multi-objective/config.yaml", 2},
		{"../../config/fragments/plugin/response-cache/memory.yaml", 1},
	} {
		t.Run(sample.path, func(t *testing.T) {
			content, err := os.ReadFile(sample.path)
			if err != nil {
				t.Fatal(err)
			}
			var document any
			if err := yaml.Unmarshal(content, &document); err != nil {
				t.Fatal(err)
			}
			count := 0
			var inspect func(any)
			inspect = func(value any) {
				switch node := value.(type) {
				case map[string]any:
					if node["type"] == "response_cache" {
						configuration, ok := node["configuration"].(map[string]any)
						if ok && configuration["enabled"] == true {
							count++
							if configuration["mode"] != "exact" {
								t.Errorf("enabled sample cache mode = %v, want exact", configuration["mode"])
							}
						}
					}
					for _, child := range node {
						inspect(child)
					}
				case []any:
					for _, child := range node {
						inspect(child)
					}
				}
			}
			inspect(document)
			if count != sample.cacheCount {
				t.Fatalf("enabled response-cache examples = %d, want %d", count, sample.cacheCount)
			}
		})
	}
}
