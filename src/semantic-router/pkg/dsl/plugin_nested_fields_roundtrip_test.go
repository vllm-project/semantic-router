package dsl

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestMemoryAndToolsPluginFieldsSurviveRoundTrip(t *testing.T) {
	for mode, selectFields := range map[string]string{
		"filter": "preserve_count: 2\n              on_error: strip\n              on_overflow: reject",
		"add":    "tools_db_path: config/tools_db.json\n              on_error: skip",
	} {
		t.Run(mode, func(t *testing.T) {
			cfg := mustParseRoutingPluginConfigTest(t, fmt.Sprintf(`
version: v0.3
routing:
  modelCards:
    - name: "test-model"
  signals:
    domains:
      - name: test
        labels: [other]
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
        - type: context
          configuration:
            enabled: true
            sources:
              - type: memory
                match:
                  recall: {method: embedding, top_k: 6, min_similarity: 0.5}
                hybrid_search: true
                hybrid_mode: weighted
                reflection:
                  enabled: true
                  algorithm: heuristic
                  max_inject_tokens: 384
                  block_patterns: ["api key", "password"]
        - type: tools
          configuration:
            enabled: true
            policy:
              mode: filtered
              block_tools: [admin.delete]
            select:
              mode: %s
              %s
              match:
                judge: {preset: tool_rank, top_k: 5, min_ratio: 0.2}
              need_threshold: 0.4
              input: {history: true}
              timeout_ms: 800
`, mode, selectFields))

			dslText := mustDecompileRoutingPluginConfigTest(t, cfg)
			if count := strings.Count(dslText, "need_threshold:"); count != 1 {
				t.Fatalf("decompiled DSL has %d need_threshold fields, want 1:\n%s", count, dslText)
			}
			compiled := mustCompileRoutingPluginConfigTest(t, dslText)
			for _, pluginType := range []string{"context", "tools"} {
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
  PLUGIN context {
    enabled: true
    sources: [{ type: "memory", reflection: true }]
  }
}
`)
	if len(errs) == 0 || !strings.Contains(fmt.Sprint(errs), "failed to decode plugin fields") {
		t.Fatalf("compile errors = %v, want a reflection decode error", errs)
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
