package dsl

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestDecompileRoutingPreservesRawPluginConfigMaps(t *testing.T) {
	cfg := mustParseRoutingPluginConfigTest(t, `
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
        - type: semantic-cache
          configuration:
            enabled: true
            similarity_threshold: 0.81
        - type: router_replay
          configuration:
            enabled: true
            max_records: 1000
            capture_request_body: true
            capture_response_body: true
            max_body_bytes: 4096
`)

	dslText := mustDecompileRoutingPluginConfigTest(t, cfg)
	assertDecompiledPluginConfigContains(t, dslText, []string{
		`similarity_threshold: 0.81`,
		`enabled: true`,
		`max_records: 1000`,
		`capture_request_body: true`,
		`capture_response_body: true`,
		`max_body_bytes: 4096`,
	})
	compiled := mustCompileRoutingPluginConfigTest(t, dslText)
	assertSemanticCachePluginRoundTrip(t, compiled.Decisions[0])
	assertRouterReplayPluginRoundTrip(t, compiled.Decisions[0])
}

func mustParseRoutingPluginConfigTest(t *testing.T, configYAML string) *config.RouterConfig {
	t.Helper()

	cfg, err := config.ParseYAMLBytes([]byte(configYAML))
	if err != nil {
		t.Fatalf("ParseYAMLBytes error: %v", err)
	}
	return cfg
}

func mustDecompileRoutingPluginConfigTest(t *testing.T, cfg *config.RouterConfig) string {
	t.Helper()

	dslText, err := DecompileRouting(cfg)
	if err != nil {
		t.Fatalf("DecompileRouting error: %v", err)
	}
	return dslText
}

func assertDecompiledPluginConfigContains(t *testing.T, dslText string, wants []string) {
	t.Helper()

	for _, want := range wants {
		if !strings.Contains(dslText, want) {
			t.Fatalf("decompiled DSL missing %q:\n%s", want, dslText)
		}
	}
}

func mustCompileRoutingPluginConfigTest(t *testing.T, dslText string) *config.RouterConfig {
	t.Helper()

	compiled, errs := Compile(dslText)
	if len(errs) > 0 {
		t.Fatalf("Compile errors: %v", errs)
	}
	if len(compiled.Decisions) != 1 {
		t.Fatalf("compiled decisions = %d", len(compiled.Decisions))
	}
	return compiled
}

func assertSemanticCachePluginRoundTrip(t *testing.T, decision config.Decision) {
	t.Helper()

	plugin := findDecisionPluginForTest(t, decision, "semantic-cache")
	var pluginConfig config.SemanticCachePluginConfig
	if err := config.UnmarshalPluginConfig(plugin.Configuration, &pluginConfig); err != nil {
		t.Fatalf("semantic-cache decode error: %v", err)
	}
	if !pluginConfig.Enabled {
		t.Fatalf("semantic-cache.enabled = false, want true")
	}
	if pluginConfig.SimilarityThreshold == nil || *pluginConfig.SimilarityThreshold != 0.81 {
		t.Fatalf("semantic-cache.similarity_threshold = %#v", pluginConfig.SimilarityThreshold)
	}
}

func assertRouterReplayPluginRoundTrip(t *testing.T, decision config.Decision) {
	t.Helper()

	plugin := findDecisionPluginForTest(t, decision, "router_replay")
	var pluginConfig config.RouterReplayPluginConfig
	if err := config.UnmarshalPluginConfig(plugin.Configuration, &pluginConfig); err != nil {
		t.Fatalf("router_replay decode error: %v", err)
	}
	if !pluginConfig.Enabled {
		t.Fatalf("router_replay.enabled = false, want true")
	}
	if pluginConfig.MaxRecords != 1000 || !pluginConfig.CaptureRequestBody || !pluginConfig.CaptureResponseBody || pluginConfig.MaxBodyBytes != 4096 {
		t.Fatalf("router_replay config = %#v", pluginConfig)
	}
}

func findDecisionPluginForTest(t *testing.T, decision config.Decision, pluginType string) config.DecisionPlugin {
	t.Helper()

	plugin := decision.GetPlugin(pluginType)
	if plugin == nil {
		t.Fatalf("%s plugin missing after roundtrip compile", pluginType)
	}
	return *plugin
}
