package dsl

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestCompileToolsPlugin(t *testing.T) {
	input := `
ROUTE security_containment {
  PRIORITY 300
  PLUGIN tools { enabled: true, mode: "none", semantic_selection: false, strip_tool_history: true }
  MODEL "local-guard"
}

ROUTE privacy_policy {
  PRIORITY 250
  PLUGIN tools { enabled: true, mode: "passthrough", semantic_selection: true }
  WHEN keyword("sensitive_kw")
  MODEL "local-model"
}

ROUTE frontier_reasoning {
  PRIORITY 200
  PLUGIN tools { enabled: true, mode: "filtered", semantic_selection: true, allow_tools: ["search_web"], block_tools: ["exec_cmd"] }
  MODEL "frontier-model"
}

ROUTE local_standard {
  PRIORITY 100
  PLUGIN tools { enabled: true, mode: "passthrough", semantic_selection: false }
  MODEL "default-model"
}
`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("Compile errors: %v", errs)
	}

	if len(cfg.Decisions) != 4 {
		t.Fatalf("expected 4 decisions, got %d", len(cfg.Decisions))
	}
	assertToolsMode := func(index int, want string) {
		t.Helper()
		cfg := cfg.Decisions[index].GetToolsConfig()
		if cfg == nil {
			t.Fatalf("decision[%d] missing tools plugin", index)
		}
		if cfg.EffectiveMode() != want {
			t.Errorf("decision[%d] tools mode = %q, want %q", index, cfg.EffectiveMode(), want)
		}
	}
	assertToolsMode(0, config.ToolsPluginModeNone)
	if toolsCfg := cfg.Decisions[0].GetToolsConfig(); toolsCfg == nil || !toolsCfg.StripToolHistory {
		t.Fatal("security_containment strip_tool_history was not compiled")
	}
	assertToolsMode(1, config.ToolsPluginModePassthrough)
	assertToolsMode(2, config.ToolsPluginModeFiltered)
	assertToolsMode(3, config.ToolsPluginModePassthrough)
}

func TestCompileToolsPluginDynamicRetrieval(t *testing.T) {
	input := `
ROUTE adaptive_tools {
  PRIORITY 200
  PLUGIN tools {
    enabled: true
    mode: "passthrough"
    semantic_selection: true
    strategy: "custom"
    dynamic_retrieval: {
      enabled: true
      strategy: "hybrid_history"
      history_window: 8
      weights: { semantic: 1.0, history: 0.7, decision_prior: 0.2, repetition_penalty: 0.1 }
      min_history_confidence: 0.4
      fallback_on_low_confidence: true
    }
  }
  MODEL "adaptive-model"
}
`

	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("Compile errors: %v", errs)
	}
	if len(cfg.Decisions) != 1 {
		t.Fatalf("expected 1 decision, got %d", len(cfg.Decisions))
	}

	toolsCfg := cfg.Decisions[0].GetToolsConfig()
	if toolsCfg == nil {
		t.Fatal("expected tools plugin")
	}
	assertToolsPluginStrategy(t, toolsCfg)
	assertDynamicRetrievalConfig(t, toolsCfg.DynamicRetrieval)
}

func assertToolsPluginStrategy(t *testing.T, cfg *config.ToolsPluginConfig) {
	t.Helper()

	if cfg.Strategy != "custom" {
		t.Fatalf("tools strategy = %q, want custom", cfg.Strategy)
	}
}

func assertDynamicRetrievalConfig(t *testing.T, dyn *config.DynamicRetrievalConfig) {
	t.Helper()

	if dyn == nil {
		t.Fatal("expected dynamic_retrieval config")
	}
	assertDynamicRetrievalCore(t, dyn)
	assertDynamicRetrievalWeights(t, dyn.Weights)
}

func assertDynamicRetrievalCore(t *testing.T, dyn *config.DynamicRetrievalConfig) {
	t.Helper()

	if !dyn.Enabled {
		t.Error("dynamic_retrieval.enabled = false, want true")
	}
	if dyn.Strategy != config.DynamicRetrievalStrategyHybridHistory {
		t.Errorf("dynamic_retrieval.strategy = %q", dyn.Strategy)
	}
	if dyn.HistoryWindow != 8 {
		t.Errorf("dynamic_retrieval.history_window = %d", dyn.HistoryWindow)
	}
	if dyn.MinHistoryConfidence != 0.4 {
		t.Errorf("dynamic_retrieval.min_history_confidence = %v", dyn.MinHistoryConfidence)
	}
	if !dyn.FallbackOnLowConfidence {
		t.Error("dynamic_retrieval.fallback_on_low_confidence = false, want true")
	}
}

func assertDynamicRetrievalWeights(t *testing.T, weights *config.DynamicRetrievalWeights) {
	t.Helper()

	if weights == nil {
		t.Fatal("expected dynamic_retrieval.weights")
	}
	if weights.Semantic != 1.0 || weights.History != 0.7 || weights.DecisionPrior != 0.2 || weights.RepetitionPenalty != 0.1 {
		t.Errorf("unexpected dynamic_retrieval.weights = %+v", *weights)
	}
}
