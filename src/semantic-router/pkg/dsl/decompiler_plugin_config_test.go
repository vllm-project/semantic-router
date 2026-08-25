package dsl

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestDecompileRoutingPreservesRawPluginConfigMaps(t *testing.T) {
	cfg := mustParseRoutingPluginConfigTest(t, `
routing:
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
      plugins:
        - type: response_cache
          configuration:
            enabled: true
            semantic:
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
	assertResponseCachePluginRoundTrip(t, compiled.Decisions[0])
	assertRouterReplayPluginRoundTrip(t, compiled.Decisions[0])
}

func TestDecompileRoutingRoundTripsHeaderAndResponsePlugins(t *testing.T) {
	headerPayload := config.MustStructuredPayload(
		config.HeaderMutationPluginConfig{
			Add:    []config.HeaderPair{{Name: "x-policy", Value: "strict"}},
			Delete: []string{"x-remove"},
		},
	)
	responsePayload := config.MustStructuredPayload(
		config.ResponseJailbreakPluginConfig{
			Enabled:   true,
			Threshold: 0.7,
			Action:    "block",
		},
	)
	cfg := &config.RouterConfig{IntelligentRouting: config.IntelligentRouting{
		Decisions: []config.Decision{{
			Name:      "plugins",
			ModelRefs: []config.ModelRef{{Model: "model-a"}},
			Plugins: []config.DecisionPlugin{
				{Type: config.DecisionPluginHeaderMutation, Configuration: headerPayload},
				{Type: config.DecisionPluginResponseJailbreak, Configuration: responsePayload},
			},
		}},
	}}

	source := mustDecompileRoutingPluginConfigTest(t, cfg)
	compiled := mustCompileRoutingPluginConfigTest(t, source)
	decision := compiled.Decisions[0]
	header := decision.GetHeaderMutationConfig()
	if header == nil || len(header.Add) != 1 ||
		header.Add[0].Name != "x-policy" ||
		len(header.Delete) != 1 {
		t.Fatalf("header mutation round trip = %#v", header)
	}
	response := decision.GetResponseJailbreakConfig()
	if response == nil || !response.Enabled ||
		response.Threshold != 0.7 ||
		response.Action != "block" {
		t.Fatalf("response jailbreak round trip = %#v", response)
	}
}

func TestDecompileRoutingRoundTripsToolsDynamicRetrievalPluginConfig(t *testing.T) {
	cfg := mustParseRoutingPluginConfigTest(t, `
routing:
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
      plugins:
        - type: tools
          configuration:
            enabled: true
            mode: passthrough
            semantic_selection: true
            strategy: custom
            dynamic_retrieval:
              enabled: true
              strategy: hybrid_history
              history_window: 8
              min_history_confidence: 0.4
              fallback_on_low_confidence: true
              weights:
                semantic: 1.0
                history: 0.7
                decision_prior: 0.2
                repetition_penalty: 0.1
`)

	dslText := mustDecompileRoutingPluginConfigTest(t, cfg)
	assertDecompiledPluginConfigContains(t, dslText, []string{
		`PLUGIN tools`,
		`strategy: "custom"`,
		`dynamic_retrieval: {`,
		`strategy: "hybrid_history"`,
		`history_window: 8`,
		`min_history_confidence: 0.4`,
		`fallback_on_low_confidence: true`,
		`weights: { decision_prior: 0.2, history: 0.7, repetition_penalty: 0.1, semantic: 1 }`,
	})

	compiled := mustCompileRoutingPluginConfigTest(t, dslText)
	assertToolsPluginDynamicRetrievalRoundTrip(t, compiled.Decisions[0])
}

func mustParseRoutingPluginConfigTest(t *testing.T, configYAML string) *config.RouterConfig {
	t.Helper()

	cfg, err := config.ParseRoutingYAMLBytes([]byte(configYAML))
	if err != nil {
		t.Fatalf("ParseRoutingYAMLBytes error: %v", err)
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

func assertResponseCachePluginRoundTrip(t *testing.T, decision config.Decision) {
	t.Helper()

	plugin := findDecisionPluginForTest(t, decision, config.DecisionPluginResponseCache)
	var pluginConfig config.ResponseCachePluginConfig
	if err := config.UnmarshalPluginConfig(plugin.Configuration, &pluginConfig); err != nil {
		t.Fatalf("response_cache decode error: %v", err)
	}
	if !pluginConfig.Enabled {
		t.Fatalf("response_cache.enabled = false, want true")
	}
	if pluginConfig.Semantic == nil || pluginConfig.Semantic.SimilarityThreshold == nil ||
		*pluginConfig.Semantic.SimilarityThreshold != 0.81 {
		t.Fatalf("response_cache.semantic.similarity_threshold = %#v", pluginConfig.Semantic)
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

func TestContextCompressionPluginNestedRoundTrip(t *testing.T) {
	cfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{{
				Name:      "compress",
				ModelRefs: []config.ModelRef{{Model: "model"}},
				Plugins: []config.DecisionPlugin{{
					Type: config.DecisionPluginContextCompression,
					Configuration: config.MustStructuredPayload(map[string]interface{}{
						"enabled": true,
						"mode":    "auto",
						"targets": map[string]interface{}{
							"tool_outputs": map[string]interface{}{
								"mode":          "extractive",
								"min_tokens":    2000,
								"target_tokens": 1000,
							},
							"rag": map[string]interface{}{"mode": "preserve"},
						},
						"request_controls": map[string]interface{}{
							"enabled": true,
							"allowed": []string{"bypass", "target"},
						},
					}),
				}},
			}},
		},
	}
	dslText := mustDecompileRoutingPluginConfigTest(t, cfg)
	assertDecompiledPluginConfigContains(t, dslText, []string{
		"PLUGIN context_compression",
		"tool_outputs:",
		"request_controls:",
	})
	compiled := mustCompileRoutingPluginConfigTest(t, dslText)
	plugin := findDecisionPluginForTest(
		t,
		compiled.Decisions[0],
		config.DecisionPluginContextCompression,
	)
	var pluginConfig config.ContextCompressionPluginConfig
	if err := config.UnmarshalPluginConfig(plugin.Configuration, &pluginConfig); err != nil {
		t.Fatalf("context_compression decode error: %v", err)
	}
	if pluginConfig.Targets == nil ||
		pluginConfig.Targets.ToolOutputs.TargetTokens != 1000 {
		t.Fatalf("context_compression targets = %#v", pluginConfig.Targets)
	}
}

func assertToolsPluginDynamicRetrievalRoundTrip(t *testing.T, decision config.Decision) {
	t.Helper()

	plugin := findDecisionPluginForTest(t, decision, "tools")
	var pluginConfig config.ToolsPluginConfig
	if err := config.UnmarshalPluginConfig(plugin.Configuration, &pluginConfig); err != nil {
		t.Fatalf("tools decode error: %v", err)
	}
	if !pluginConfig.Enabled {
		t.Fatal("tools.enabled = false, want true")
	}
	assertRoundTripToolsPluginBasics(t, &pluginConfig)
	assertDynamicRetrievalConfig(t, pluginConfig.DynamicRetrieval)
}

func assertRoundTripToolsPluginBasics(t *testing.T, cfg *config.ToolsPluginConfig) {
	t.Helper()

	if cfg.Mode != config.ToolsPluginModePassthrough {
		t.Fatalf("tools.mode = %q", cfg.Mode)
	}
	if cfg.SemanticSelection == nil || !*cfg.SemanticSelection {
		t.Fatalf("tools.semantic_selection = %#v", cfg.SemanticSelection)
	}
	assertToolsPluginStrategy(t, cfg)
}

func TestDecompileRoutingRoundTripsToolsStripHistory(t *testing.T) {
	cfg := mustParseRoutingPluginConfigTest(t, `
routing:
  decisions:
    - name: privacy_route
      priority: 100
      plugins:
        - type: tools
          configuration:
            enabled: true
            mode: none
            strip_tool_history: true
`)

	dslText := mustDecompileRoutingPluginConfigTest(t, cfg)
	assertDecompiledPluginConfigContains(t, dslText, []string{
		`PLUGIN tools`,
		`mode: "none"`,
		`strip_tool_history: true`,
	})
	compiled := mustCompileRoutingPluginConfigTest(t, dslText)
	toolsCfg := compiled.Decisions[0].GetToolsConfig()
	if toolsCfg == nil || !toolsCfg.StripToolHistory {
		t.Fatal("tools.strip_tool_history was not preserved")
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
