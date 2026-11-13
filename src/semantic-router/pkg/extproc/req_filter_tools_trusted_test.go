package extproc

import (
	"encoding/json"
	"fmt"
	"testing"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

// trustedFactsCatalogPath names the catalog the test selections add from;
// its read is the availability evidence the gate weighs.
const trustedFactsCatalogPath = "tools.json"

// trustedFactsTestConfig builds an enabled tools plugin config carrying the
// given trusted-facts block. Its catalog is empty, so a selection the gate
// allows has no candidate and asks no model.
func trustedFactsTestConfig(t *testing.T, trusted *config.TrustedFactsConfig) *config.ToolsPluginConfig {
	t.Helper()
	return &config.ToolsPluginConfig{
		Enabled: true,
		Policy:  &config.ToolsPolicyConfig{TrustedFacts: trusted},
		Select:  &config.ToolsSelectConfig{Mode: config.ToolsSelectModeAdd, ToolsDBPath: trustedFactsCatalogPath},
	}
}

// trustedFactsRouter serves catalog as the decision's, none when nil.
func trustedFactsRouter(catalog *toolCatalog) *OpenAIRouter {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	if catalog != nil {
		router.toolCatalogs = map[string]*toolCatalog{trustedFactsCatalogPath: catalog}
	}
	return router
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

// loadedTrustedFactsCatalog is a catalog read now, so it carries fresh
// availability evidence.
func loadedTrustedFactsCatalog() *toolCatalog {
	return &toolCatalog{loadedAt: time.Now()}
}

// trustedFactsTestRequest carries function tools with object schemas so the
// strict encoder accepts them on paths that serialize the request.
func trustedFactsTestRequest() *llmprotocol.Request {
	req := testNeutralRequest("model", "look up the weather")
	req.ToolChoice = llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto}
	schema := json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}}}`)
	req.Tools = []llmprotocol.Tool{{Name: "search", InputSchema: schema}, {Name: "calculator", InputSchema: schema}}
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
	facts, ok := resolveTrustedFactsGate(req, cfg, true, trustedFactsRequestStage)
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
		t.Fatal("a catalog read now must count as available")
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
		"nil block":      {Enabled: true, Policy: &config.ToolsPolicyConfig{}},
		"disabled block": {Enabled: true, Policy: &config.ToolsPolicyConfig{TrustedFacts: &config.TrustedFactsConfig{Enabled: false}}},
	} {
		t.Run(name, func(t *testing.T) {
			if _, ok := resolveTrustedFactsGate(req, cfg, true, trustedFactsRequestStage); ok {
				t.Fatal("gate must not engage")
			}
		})
	}
	if _, ok := resolveTrustedFactsGate(nil, trustedFactsTestConfig(t, &config.TrustedFactsConfig{Enabled: true}), true, trustedFactsRequestStage); ok {
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
	facts, ok := resolveTrustedFactsGate(req, cfg, true, trustedFactsRequestStage)
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
	router := trustedFactsRouter(nil)
	req := trustedFactsTestRequest()
	resp := &ext_proc.ProcessingResponse{}
	router.applyToolsPlugin(req, resp, trustedFactsTestContext(t, cfg))
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
	router := trustedFactsRouter(nil)
	req := trustedFactsTestRequest()
	resp := &ext_proc.ProcessingResponse{}
	router.applyToolsPlugin(req, resp, trustedFactsTestContext(t, cfg))
	if len(req.Tools) != 0 {
		t.Fatalf("candidate-stage request against a final-only policy must be denied, got %v", req.Tools)
	}
}

func TestHandleToolSelectionTrustedFactsNarrowKeepsExplicitToolsOnly(t *testing.T) {
	// Authorized but the decision has no catalog: narrow keeps
	// explicitly allowed tools and skips retrieval expansion.
	cfg := trustedFactsTestConfig(t, &config.TrustedFactsConfig{
		Enabled:      true,
		Enforcement:  config.TrustedEnforcementAuthoritative,
		TrustSources: []string{config.TrustedSourceOperatorPolicy},
		StageRoles:   []string{config.TrustedStageCandidate},
	})
	cfg.Policy.Mode, cfg.Policy.AllowTools = config.ToolsPolicyModeFiltered, []string{"search"}
	router := trustedFactsRouter(nil)
	req := trustedFactsTestRequest()
	resp := &ext_proc.ProcessingResponse{}
	router.applyToolsPlugin(req, resp, trustedFactsTestContext(t, cfg))
	if len(req.Tools) != 1 || req.Tools[0].Name != "search" {
		t.Fatalf("narrow must keep only explicitly allowed tools, got %v", req.Tools)
	}
}

func TestHandleToolSelectionTrustedFactsNarrowReconcilesNamedToolChoice(t *testing.T) {
	cases := map[string]struct {
		forced     string
		wantChoice llmprotocol.ToolChoice
	}{
		"removed tool clears the choice": {forced: "calculator"},
		"kept tool keeps the choice": {
			forced:     "search",
			wantChoice: llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceNamed, Name: "search"},
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			cfg := trustedFactsTestConfig(t, &config.TrustedFactsConfig{
				Enabled:      true,
				Enforcement:  config.TrustedEnforcementAuthoritative,
				TrustSources: []string{config.TrustedSourceOperatorPolicy},
				StageRoles:   []string{config.TrustedStageCandidate},
			})
			cfg.Policy.Mode, cfg.Policy.AllowTools = config.ToolsPolicyModeFiltered, []string{"search"}
			req := trustedFactsTestRequest()
			req.ToolChoice = llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceNamed, Name: tc.forced}
			require.NoError(t, llmprotocol.ValidateRequest(*req, llmprotocol.DefaultPolicy().Limits), "fixture must start valid")

			resp := &ext_proc.ProcessingResponse{}
			trustedFactsRouter(nil).applyToolsPlugin(req, resp, trustedFactsTestContext(t, cfg))

			require.Len(t, req.Tools, 1)
			require.Equal(t, "search", req.Tools[0].Name)
			require.Equal(t, tc.wantChoice, req.ToolChoice)
			require.NoError(t, llmprotocol.ValidateRequest(*req, llmprotocol.DefaultPolicy().Limits))
		})
	}
}

func TestHandleToolSelectionTrustedFactsAllowAndObserveContinue(t *testing.T) {
	db := loadedTrustedFactsCatalog()
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
			router := trustedFactsRouter(db)
			req := trustedFactsTestRequest()
			resp := &ext_proc.ProcessingResponse{}
			router.applyToolsPlugin(req, resp, trustedFactsTestContext(t, cfg))
			if len(req.Tools) != 2 {
				t.Fatalf("allow/observe must leave tools untouched, got %v", req.Tools)
			}
		})
	}
}

func TestHandleToolSelectionWithoutTrustedFactsUnchanged(t *testing.T) {
	cfg := &config.ToolsPluginConfig{Enabled: true, Policy: &config.ToolsPolicyConfig{}}
	router := trustedFactsRouter(nil)
	req := trustedFactsTestRequest()
	resp := &ext_proc.ProcessingResponse{}
	router.applyToolsPlugin(req, resp, trustedFactsTestContext(t, cfg))
	if len(req.Tools) != 2 {
		t.Fatalf("absent trusted-facts block must leave tools untouched, got %v", req.Tools)
	}
}

func TestHandleToolSelectionDisabledParentDisengagesGate(t *testing.T) {
	// Disabled parent plus enabled nested block: the gate must not engage,
	// since the nested policy never passed startup validation for a disabled
	// parent. Tools flow reaches the parent-disabled guard unchanged.
	cfg := &config.ToolsPluginConfig{
		Enabled: false,
		Policy: &config.ToolsPolicyConfig{TrustedFacts: &config.TrustedFactsConfig{
			Enabled:      true,
			Enforcement:  config.TrustedEnforcementAuthoritative,
			TrustSources: []string{config.TrustedSourceOperatorPolicy},
			StageRoles:   []string{config.TrustedStageCandidate},
		}},
	}
	router := trustedFactsRouter(nil)
	req := trustedFactsTestRequest()
	resp := &ext_proc.ProcessingResponse{}
	router.applyToolsPlugin(req, resp, trustedFactsTestContext(t, cfg))
	if len(req.Tools) != 2 {
		t.Fatalf("disabled parent must leave tools untouched, got %v", req.Tools)
	}
}

func TestHandleToolSelectionTrustedFactsAvailabilityNeedsBoundedEvidence(t *testing.T) {
	cfg := trustedFactsTestConfig(t, &config.TrustedFactsConfig{
		Enabled:          true,
		Enforcement:      config.TrustedEnforcementAuthoritative,
		TrustSources:     []string{config.TrustedSourceOperatorPolicy, config.TrustedSourceRuntimeFresh},
		FreshnessSeconds: 60,
		StageRoles:       []string{config.TrustedStageCandidate},
	})
	cfg.Policy.Mode, cfg.Policy.AllowTools = config.ToolsPolicyModeFiltered, []string{"search"}
	cases := map[string]struct {
		db   *toolCatalog
		age  time.Duration
		want llmprotocol.TrustedOutcome
	}{
		// A decision without a catalog has no availability evidence.
		"no catalog narrows": {db: nil, want: llmprotocol.TrustedNarrow},
		"stale read narrows": {db: loadedTrustedFactsCatalog(), age: 2 * time.Minute, want: llmprotocol.TrustedNarrow},
		"fresh read allows":  {db: loadedTrustedFactsCatalog(), want: llmprotocol.TrustedAllow},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			trustedFactsNow = func() time.Time { return time.Now().Add(tc.age) }
			t.Cleanup(func() { trustedFactsNow = time.Now })
			router := trustedFactsRouter(tc.db)
			req := trustedFactsTestRequest()
			ctx := trustedFactsTestContext(t, cfg)
			router.applyToolsPlugin(req, &ext_proc.ProcessingResponse{}, ctx)
			if len(req.Tools) != 1 || len(ctx.pendingToolsOutcomes) == 0 || ctx.pendingToolsOutcomes[0].Verdict != string(tc.want) {
				t.Fatalf("tools = %v, outcomes %+v, want %s", req.Tools, ctx.pendingToolsOutcomes, tc.want)
			}
		})
	}
}

// The gate returns before ordinary mode handling, so deny and narrow must
// still honor mode none. Every database state has to produce the same request
// a fresh load produces, where the gate allows and mode none runs as usual.
func TestHandleToolSelectionTrustedFactsKeepsModeNone(t *testing.T) {
	cases := map[string]struct {
		sources []string
		db      *toolCatalog
		age     time.Duration
	}{
		"no database narrows":       {sources: []string{config.TrustedSourceOperatorPolicy, config.TrustedSourceRuntimeFresh}},
		"unloaded database narrows": {sources: []string{config.TrustedSourceOperatorPolicy, config.TrustedSourceRuntimeFresh}, db: nil},
		"stale load narrows":        {sources: []string{config.TrustedSourceOperatorPolicy, config.TrustedSourceRuntimeFresh}, db: loadedTrustedFactsCatalog(), age: 2 * time.Minute},
		"unauthorized denies":       {sources: []string{config.TrustedSourceRuntimeFresh}, db: loadedTrustedFactsCatalog()},
		"fresh load allows":         {sources: []string{config.TrustedSourceOperatorPolicy, config.TrustedSourceRuntimeFresh}, db: loadedTrustedFactsCatalog()},
	}
	for name, tc := range cases {
		for _, stripHistory := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/strip_tool_history=%t", name, stripHistory), func(t *testing.T) {
				trustedFactsNow = func() time.Time { return time.Now().Add(tc.age) }
				t.Cleanup(func() { trustedFactsNow = time.Now })
				cfg := trustedFactsTestConfig(t, &config.TrustedFactsConfig{
					Enabled:          true,
					Enforcement:      config.TrustedEnforcementAuthoritative,
					TrustSources:     tc.sources,
					FreshnessSeconds: 60,
					StageRoles:       []string{config.TrustedStageCandidate},
				})
				cfg.Policy.Mode = config.ToolsPolicyModeNone
				cfg.Policy.StripToolHistory = stripHistory
				cfg.Policy.AllowTools = []string{"search"}
				req := trustedFactsTestRequest()
				req.ParallelToolCalls = boolPtr(true)
				req.Messages = append(req.Messages, llmprotocol.Message{
					Role:    llmprotocol.RoleTool,
					Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "sunny"}},
				})
				router := trustedFactsRouter(tc.db)
				resp := &ext_proc.ProcessingResponse{}
				router.applyToolsPlugin(req, resp, trustedFactsTestContext(t, cfg))

				require.Nil(t, req.Tools)
				require.Equal(t, llmprotocol.ToolChoice{}, req.ToolChoice)
				require.Nil(t, req.ParallelToolCalls)
				wantMessages := 2
				if stripHistory {
					wantMessages = 1
				}
				require.Len(t, req.Messages, wantMessages)
			})
		}
	}
}

// Responses image_generation is normalized outside Request.Tools. An
// authoritative deny must drop it and its forced tool_choice from the encoded
// request, while allow keeps both.
func TestHandleToolSelectionTrustedFactsGatesHostedImageGeneration(t *testing.T) {
	body := []byte(`{"model":"model","input":"draw a chart and look up the weather",` +
		`"tools":[{"type":"function","name":"search","parameters":{"type":"object","properties":{"query":{"type":"string"}}}},{"type":"image_generation"}],` +
		`"tool_choice":{"type":"image_generation"}}`)
	cases := map[string]struct {
		trusted   *config.TrustedFactsConfig
		wantImage bool
	}{
		"deny clears hosted tool": {
			trusted: &config.TrustedFactsConfig{
				Enabled:          true,
				Enforcement:      config.TrustedEnforcementAuthoritative,
				TrustSources:     []string{config.TrustedSourceRuntimeFresh},
				FreshnessSeconds: 60,
				StageRoles:       []string{config.TrustedStageCandidate},
			},
		},
		"allow keeps hosted tool": {
			trusted: &config.TrustedFactsConfig{
				Enabled:      true,
				Enforcement:  config.TrustedEnforcementAuthoritative,
				TrustSources: []string{config.TrustedSourceOperatorPolicy},
				StageRoles:   []string{config.TrustedStageCandidate},
			},
			wantImage: true,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			router := &OpenAIRouter{
				Config:            &config.RouterConfig{},
				toolCatalogs:      map[string]*toolCatalog{trustedFactsCatalogPath: loadedTrustedFactsCatalog()},
				ResponseAPIFilter: NewResponseAPIFilter(NewMockResponseStore()),
			}
			ctx := trustedFactsTestContext(t, trustedFactsTestConfig(t, tc.trusted))
			ctx.SourceFormat = llmprotocol.OpenAIResponsesV1
			ctx.TraceContext = t.Context()
			req, immediate := router.prepareProtocolRequest(body, ctx)
			require.Nil(t, immediate)
			require.NotNil(t, req.ImageGeneration, "fixture must normalize the hosted tool")

			router.applyToolsPlugin(req, &ext_proc.ProcessingResponse{}, ctx)

			engine, err := router.protocolEngine()
			require.NoError(t, err)
			encoded, err := engine.EncodeRequest(llmprotocol.OpenAIResponsesV1, *ctx.SemanticRequest, ctx.ProtocolEnvelope)
			require.NoError(t, err)
			var wire struct {
				Tools      []map[string]any `json:"tools"`
				ToolChoice any              `json:"tool_choice"`
			}
			require.NoError(t, json.Unmarshal(encoded.Body, &wire))
			hasImage := false
			for _, tool := range wire.Tools {
				if tool["type"] == "image_generation" {
					hasImage = true
				}
			}
			require.Equal(t, tc.wantImage, hasImage, "encoded tools: %s", encoded.Body)
			if tc.wantImage {
				require.Equal(t, map[string]any{"type": "image_generation"}, wire.ToolChoice)
				return
			}
			require.Empty(t, wire.Tools)
			require.Nil(t, wire.ToolChoice, "deny must drop the forced hosted tool_choice")
		})
	}
}

// The gate runs before the ordinary path starts its Replay record, so the
// outcome must survive until the record exists.
func TestEntrypointRoutingReplayRecordsTrustedFactsOutcome(t *testing.T) {
	calls := 0
	r, ctx, model := dispatchReplayFixture(t, false, renderMock(t, &calls, 0, 0))
	cfg := trustedFactsTestConfig(t, &config.TrustedFactsConfig{
		Enabled:          true,
		Enforcement:      config.TrustedEnforcementAuthoritative,
		TrustSources:     []string{config.TrustedSourceRuntimeFresh},
		FreshnessSeconds: 60,
		StageRoles:       []string{config.TrustedStageCandidate},
	})
	ctx.VSRSelectedDecision.Plugins = append(ctx.VSRSelectedDecision.Plugins, mustToolsDecisionPlugin(t, cfg))
	ctx.SemanticRequest.Tools = trustedFactsTestRequest().Tools
	ctx.SemanticRequest.ToolChoice = llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto}

	_, err := r.handleEntrypointModelRouting(ctx.SemanticRequest, "auto", ctx.VSRSelectedDecision.Name, entropy.ReasoningDecision{}, model, ctx)
	require.NoError(t, err)
	require.Empty(t, ctx.SemanticRequest.Tools)
	requireTrustedFactsReplayOutcome(t, r.ReplayRecorder, ctx, llmprotocol.TrustedDeny)
}

func requireTrustedFactsReplayOutcome(t *testing.T, recorder *routerreplay.Recorder, ctx *RequestContext, want llmprotocol.TrustedOutcome) {
	t.Helper()
	require.NotEmpty(t, ctx.RouterReplayID, "Replay record was not created")
	record, ok := recorder.GetRecord(ctx.RouterReplayID)
	require.True(t, ok)
	var verdicts []string
	for _, outcome := range record.Outcomes {
		if outcome.Target == "trusted_facts" {
			verdicts = append(verdicts, outcome.Verdict)
		}
	}
	require.Equal(t, []string{string(want)}, verdicts, "trusted_facts outcomes on the Replay record")
	require.Empty(t, ctx.pendingToolsOutcomes, "retained outcomes must be appended exactly once")
}

// A narrowed request keeps a hosted tool only when the filtered policy
// allows it by its type, as the policy does without trusted facts.
func TestHandleToolSelectionTrustedFactsNarrowFiltersHostedTools(t *testing.T) {
	cfg := trustedFactsTestConfig(t, &config.TrustedFactsConfig{
		Enabled:      true,
		Enforcement:  config.TrustedEnforcementAuthoritative,
		TrustSources: []string{config.TrustedSourceOperatorPolicy},
		StageRoles:   []string{config.TrustedStageCandidate},
	})
	cfg.Policy.Mode, cfg.Policy.AllowTools = config.ToolsPolicyModeFiltered, []string{"search"}
	req := trustedFactsTestRequest()
	req.ImageGeneration = &llmprotocol.ImageGenerationOptions{}
	trustedFactsRouter(nil).applyToolsPlugin(req, &ext_proc.ProcessingResponse{}, trustedFactsTestContext(t, cfg))
	if req.ImageGeneration != nil || len(req.Tools) != 1 || req.Tools[0].Name != "search" {
		t.Fatalf("narrow kept image_generation=%v tools=%v", req.ImageGeneration != nil, req.Tools)
	}
}
