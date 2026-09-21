package testcases

import (
	"encoding/json"
	"testing"
)

func TestPluginMutationProviderHeaderCardinality(t *testing.T) {
	tests := []struct {
		name      string
		values    []string
		wantError bool
	}{
		{"overwrite", []string{"updated-by-router"}, false},
		{"append client first", []string{"client-value", "updated-by-router"}, true},
		{"append router first", []string{"updated-by-router", "client-value"}, true},
		{"comma joined", []string{"client-value,updated-by-router"}, true},
		{"duplicated correct value", []string{"updated-by-router", "updated-by-router"}, true},
		{"missing", nil, true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			observed := map[string]any{
				"body": map[string]any{
					"messages":              []any{map[string]any{"role": "system", "content": pluginMutationSystemPrompt}, map[string]any{"role": "user", "content": "__PLUGIN_REQUEST_MUTATIONS__ preserve this user message"}},
					"max_completion_tokens": 64, "n": 1,
				},
				// The lossy compatibility view must not hide duplicate values.
				"headers":       map[string]string{"x-vsr-e2e-added": "added-by-router", "x-vsr-e2e-updated": "updated-by-router"},
				"header_values": map[string][]string{"x-vsr-e2e-added": {"added-by-router"}, "x-vsr-e2e-updated": tc.values},
			}
			data, err := json.Marshal(observed)
			if err != nil {
				t.Fatal(err)
			}
			err = validatePluginMutationProviderRequest(data)
			if (err != nil) != tc.wantError {
				t.Fatalf("validation error=%v, wantError=%t", err, tc.wantError)
			}
		})
	}
}
