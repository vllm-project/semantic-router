package dsl

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestMemoryAndToolSelectionPluginFieldsSurviveRoundTrip(t *testing.T) {
	for _, fallbackToEmpty := range []bool{false, true} {
		t.Run(fmt.Sprintf("fallback_to_empty=%v", fallbackToEmpty), func(t *testing.T) {
			cfg := mustParseRoutingPluginConfigTest(t, fmt.Sprintf(`
version: v0.3
routing:
  modelCards:
    - name: "test-model"
  signals:
    domains:
      - name: test
        description: test
  decisions:
    - name: plugin_route
      priority: 100
      rules:
        operator: AND
        conditions:
          - type: domain
            name: test
      modelRefs:
        - model: test-model
      plugins:
        - type: memory
          configuration:
            enabled: true
            retrieval_limit: 6
            hybrid_search: true
            hybrid_mode: weighted
            reflection:
              enabled: true
              algorithm: heuristic
              max_inject_tokens: 384
              block_patterns: ["api key", "password"]
        - type: tool_selection
          configuration:
            enabled: true
            mode: add
            top_k: 5
            fallback_to_empty: %v
            advanced_filtering:
              enabled: true
              candidate_pool_size: 30
              min_combined_score: 0.4
`, fallbackToEmpty))

			dslText := mustDecompileRoutingPluginConfigTest(t, cfg)
			if count := strings.Count(dslText, "fallback_to_empty:"); count != 1 {
				t.Fatalf("decompiled DSL has %d fallback_to_empty fields, want 1:\n%s", count, dslText)
			}
			compiled := mustCompileRoutingPluginConfigTest(t, dslText)
			for _, pluginType := range []string{"memory", "tool_selection"} {
				want := pluginConfigMapForTest(t, findDecisionPluginForTest(t, cfg.Decisions[0], pluginType))
				got := pluginConfigMapForTest(t, findDecisionPluginForTest(t, compiled.Decisions[0], pluginType))
				if !reflect.DeepEqual(got, want) {
					t.Fatalf("%s after round trip = %v, want %v", pluginType, got, want)
				}
			}
		})
	}
}

func TestCompileRejectsNonObjectMemoryReflection(t *testing.T) {
	_, errs := Compile(`
SIGNAL domain test { description: "test" }
ROUTE memory_route {
  PRIORITY 100
  WHEN domain("test")
  MODEL "m:1b"
  PLUGIN memory {
    enabled: true
    reflection: true
  }
}
`)
	if len(errs) == 0 || !strings.Contains(fmt.Sprint(errs), `plugin field "reflection" must be an object`) {
		t.Fatalf("compile errors = %v, want a reflection object error", errs)
	}
}

func pluginConfigMapForTest(t *testing.T, plugin config.DecisionPlugin) map[string]interface{} {
	t.Helper()

	fields, err := plugin.Configuration.AsStringMap()
	if err != nil {
		t.Fatalf("%s configuration: %v", plugin.Type, err)
	}
	return fields
}
