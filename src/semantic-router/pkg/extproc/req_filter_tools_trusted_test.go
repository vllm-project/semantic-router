package extproc

import (
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
)

// trustedFactsTestConfig builds an enabled tools plugin config carrying the
// given trusted-facts block. Semantic selection stays off so allow-path tests
// stop at the commit step without touching retrieval.
func trustedFactsTestConfig(t *testing.T, trusted *config.TrustedFactsConfig) *config.ToolsPluginConfig {
	t.Helper()
	return &config.ToolsPluginConfig{
		Enabled:           true,
		Mode:              config.ToolsPluginModePassthrough,
		SemanticSelection: boolPtr(false),
		TrustedFacts:      trusted,
	}
}

func trustedFactsTestContext(t *testing.T, cfg *config.ToolsPluginConfig) *RequestContext {
	t.Helper()
	return &RequestContext{
		VSRSelectedDecision: &config.Decision{
			Name:    "trusted-facts-decision",
			Plugins: []config.DecisionPlugin{mustToolsDecisionPlugin(t, cfg)},
		},
	}
}

func trustedFactsTestRequest() *llmprotocol.Request {
	req := testNeutralRequest("model", "look up the weather")
	req.ToolChoice = llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto}
	req.Tools = []llmprotocol.Tool{{Name: "search"}, {Name: "calculator"}}
	return req
}

func TestResolveTrustedFactsGateMapsDistinctFacts(t *testing.T) {
	req := trustedFactsTestRequest()
	cfg := trustedFactsTestConfig(t, &config.TrustedFactsConfig{
		Enabled:      true,
		Enforcement:  config.TrustedEnforcementAuthoritative,
		TrustSources: []string{config.TrustedSourceOperatorPolicy},
		StageRoles:   []string{config.TrustedStageCandidate},
	})
	facts, ok := resolveTrustedFactsGate(req, cfg, true)
	if !ok {
		t.Fatal("gate should engage for an enabled trusted-facts block")
	}
	if facts.Enforcement != llmprotocol.TrustedAuthoritative {
		t.Fatalf("enforcement = %q, want authoritative", facts.Enforcement)
	}
	if !facts.Capable {
		t.Fatal("request carrying tools must be capable")
	}
	if !facts.Authorized {
		t.Fatal("operator-policy declaration must authorize")
	}
	if !facts.Available {
		t.Fatal("live tools database must count as available")
	}
	if facts.Stage != llmprotocol.TrustedStageCandidate {
		t.Fatalf("stage = %q, want candidate", facts.Stage)
	}
	if len(facts.AllowedStages) != 1 || facts.AllowedStages[0] != llmprotocol.TrustedStageCandidate {
		t.Fatalf("allowed stages = %v, want [candidate]", facts.AllowedStages)
	}
}

func TestResolveTrustedFactsGateSkipsWhenDisabled(t *testing.T) {
	req := trustedFactsTestRequest()
	for name, cfg := range map[string]*config.ToolsPluginConfig{
		"nil config":     nil,
		"nil block":      {Enabled: true, Mode: config.ToolsPluginModePassthrough},
		"disabled block": {Enabled: true, TrustedFacts: &config.TrustedFactsConfig{Enabled: false}},
	} {
		t.Run(name, func(t *testing.T) {
			if _, ok := resolveTrustedFactsGate(req, cfg, true); ok {
				t.Fatal("gate must not engage")
			}
		})
	}
	if _, ok := resolveTrustedFactsGate(nil, trustedFactsTestConfig(t, &config.TrustedFactsConfig{Enabled: true}), true); ok {
		t.Fatal("gate must not engage for a nil request")
	}
}

func TestResolveTrustedFactsGateSeparatesAuthorizationFromAvailability(t *testing.T) {
	req := trustedFactsTestRequest()
	// Gateway-attested and runtime-fresh declarations alone authorize nothing:
	// attestation has no per-request verification signal yet and runtime
	// evidence can only narrow.
	cfg := trustedFactsTestConfig(t, &config.TrustedFactsConfig{
		Enabled:          true,
		Enforcement:      config.TrustedEnforcementAuthoritative,
		TrustSources:     []string{config.TrustedSourceGatewayAttested, config.TrustedSourceRuntimeFresh},
		FreshnessSeconds: 60,
		StageRoles:       []string{config.TrustedStageCandidate},
	})
	facts, ok := resolveTrustedFactsGate(req, cfg, true)
	if !ok {
		t.Fatal("gate should engage")
	}
	if facts.Authorized {
		t.Fatal("gateway/runtime declarations must not authorize")
	}
	if got := llmprotocol.EvaluateTrustedFacts(facts); got != llmprotocol.TrustedDeny {
		t.Fatalf("outcome = %q, want deny", got)
	}
}

func TestHandleToolSelectionTrustedFactsDenyStripsTools(t *testing.T) {
	// Runtime-only source: fresh availability must not authorize.
	cfg := trustedFactsTestConfig(t, &config.TrustedFactsConfig{
		Enabled:          true,
		Enforcement:      config.TrustedEnforcementAuthoritative,
		TrustSources:     []string{config.TrustedSourceRuntimeFresh},
		FreshnessSeconds: 60,
		StageRoles:       []string{config.TrustedStageCandidate},
	})
	router := &OpenAIRouter{}
	req := trustedFactsTestRequest()
	resp := &ext_proc.ProcessingResponse{}
	if err := router.handleToolSelection(req, "look up the weather", nil, &resp, trustedFactsTestContext(t, cfg)); err != nil {
		t.Fatalf("handleToolSelection returned unexpected error: %v", err)
	}
	if len(req.Tools) != 0 {
		t.Fatalf("deny must strip all tools, got %v", req.Tools)
	}
}

func TestHandleToolSelectionTrustedFactsDenyDisallowedStage(t *testing.T) {
	cfg := trustedFactsTestConfig(t, &config.TrustedFactsConfig{
		Enabled:      true,
		Enforcement:  config.TrustedEnforcementAuthoritative,
		TrustSources: []string{config.TrustedSourceOperatorPolicy},
		StageRoles:   []string{config.TrustedStageFinal},
	})
	router := &OpenAIRouter{}
	req := trustedFactsTestRequest()
	resp := &ext_proc.ProcessingResponse{}
	if err := router.handleToolSelection(req, "look up the weather", nil, &resp, trustedFactsTestContext(t, cfg)); err != nil {
		t.Fatalf("handleToolSelection returned unexpected error: %v", err)
	}
	if len(req.Tools) != 0 {
		t.Fatalf("candidate-stage request against a final-only policy must be denied, got %v", req.Tools)
	}
}

func TestHandleToolSelectionTrustedFactsNarrowKeepsExplicitToolsOnly(t *testing.T) {
	// Authorized but the tools database is unavailable: narrow keeps
	// explicitly allowed tools and skips retrieval expansion.
	cfg := trustedFactsTestConfig(t, &config.TrustedFactsConfig{
		Enabled:      true,
		Enforcement:  config.TrustedEnforcementAuthoritative,
		TrustSources: []string{config.TrustedSourceOperatorPolicy},
		StageRoles:   []string{config.TrustedStageCandidate},
	})
	cfg.AllowTools = []string{"search"}
	router := &OpenAIRouter{}
	req := trustedFactsTestRequest()
	resp := &ext_proc.ProcessingResponse{}
	if err := router.handleToolSelection(req, "look up the weather", nil, &resp, trustedFactsTestContext(t, cfg)); err != nil {
		t.Fatalf("handleToolSelection returned unexpected error: %v", err)
	}
	if len(req.Tools) != 1 || req.Tools[0].Name != "search" {
		t.Fatalf("narrow must keep only explicitly allowed tools, got %v", req.Tools)
	}
}

func TestHandleToolSelectionTrustedFactsAllowAndObserveContinue(t *testing.T) {
	db := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{Enabled: true})
	for _, enforcement := range []string{"", config.TrustedEnforcementAuthoritative} {
		name := "advisory observes"
		if enforcement != "" {
			name = "authoritative allows"
		}
		t.Run(name, func(t *testing.T) {
			cfg := trustedFactsTestConfig(t, &config.TrustedFactsConfig{
				Enabled:      true,
				Enforcement:  enforcement,
				TrustSources: []string{config.TrustedSourceOperatorPolicy},
				StageRoles:   []string{config.TrustedStageCandidate},
			})
			router := &OpenAIRouter{ToolsDatabase: db}
			req := trustedFactsTestRequest()
			resp := &ext_proc.ProcessingResponse{}
			if err := router.handleToolSelection(req, "look up the weather", nil, &resp, trustedFactsTestContext(t, cfg)); err != nil {
				t.Fatalf("handleToolSelection returned unexpected error: %v", err)
			}
			if len(req.Tools) != 2 {
				t.Fatalf("allow/observe must leave tools untouched, got %v", req.Tools)
			}
		})
	}
}

func TestHandleToolSelectionWithoutTrustedFactsUnchanged(t *testing.T) {
	cfg := &config.ToolsPluginConfig{
		Enabled:           true,
		Mode:              config.ToolsPluginModePassthrough,
		SemanticSelection: boolPtr(false),
	}
	router := &OpenAIRouter{}
	req := trustedFactsTestRequest()
	resp := &ext_proc.ProcessingResponse{}
	if err := router.handleToolSelection(req, "look up the weather", nil, &resp, trustedFactsTestContext(t, cfg)); err != nil {
		t.Fatalf("handleToolSelection returned unexpected error: %v", err)
	}
	if len(req.Tools) != 2 {
		t.Fatalf("absent trusted-facts block must leave tools untouched, got %v", req.Tools)
	}
}

func TestHandleToolSelectionDisabledParentDisengagesGate(t *testing.T) {
	// Disabled parent plus enabled nested block: the gate must not engage,
	// since the nested policy never passed startup validation for a disabled
	// parent. Tools flow reaches the parent-disabled guard unchanged.
	cfg := &config.ToolsPluginConfig{
		Enabled:           false,
		Mode:              config.ToolsPluginModePassthrough,
		SemanticSelection: boolPtr(false),
		TrustedFacts: &config.TrustedFactsConfig{
			Enabled:      true,
			Enforcement:  config.TrustedEnforcementAuthoritative,
			TrustSources: []string{config.TrustedSourceOperatorPolicy},
			StageRoles:   []string{config.TrustedStageCandidate},
		},
	}
	router := &OpenAIRouter{}
	req := trustedFactsTestRequest()
	resp := &ext_proc.ProcessingResponse{}
	if err := router.handleToolSelection(req, "look up the weather", nil, &resp, trustedFactsTestContext(t, cfg)); err != nil {
		t.Fatalf("handleToolSelection returned unexpected error: %v", err)
	}
	if len(req.Tools) != 2 {
		t.Fatalf("disabled parent must leave tools untouched, got %v", req.Tools)
	}
}
