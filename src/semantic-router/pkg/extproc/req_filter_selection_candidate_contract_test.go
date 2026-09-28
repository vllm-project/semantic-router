package extproc

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

// PR #3685: an exact-effort winner must survive component composition, session
// protection, ExtProc resolution, and the final provider request encoding.
// These are regression expectations, not assertions of the current broken behavior.
func TestSelectionCandidateEffortReachesProviderRequest(t *testing.T) {
	for _, test := range []struct {
		name      string
		algorithm string
		scope     string
		toolLoop  bool
	}{
		{name: "multi_factor", algorithm: config.DecisionAlgorithmMultiFactor},
		{name: "automix", algorithm: config.DecisionAlgorithmAutoMix},
		{name: "hybrid", algorithm: config.DecisionAlgorithmHybrid},
		{name: "protection_session", algorithm: config.DecisionAlgorithmMultiFactor, scope: config.RouterLearningScopeSession},
		{name: "protection_conversation", algorithm: config.DecisionAlgorithmMultiFactor, scope: config.RouterLearningScopeConversation},
		{name: "protection_tool_loop", algorithm: config.DecisionAlgorithmMultiFactor, scope: config.RouterLearningScopeConversation, toolLoop: true},
	} {
		for _, winningEffort := range []string{"low", "high"} {
			t.Run(test.name+"/"+winningEffort, func(t *testing.T) {
				sessiontelemetry.ResetRouterSessionMemoryForTesting()
				t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)

				router, decision := candidateEffortTestRouter(t, test.algorithm, winningEffort)
				request := testNeutralRequest("virtual", "solve this")
				ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
				ctx.VSRSelectedDecision = decision
				if test.scope != "" {
					router.Config.RouterLearning = routerLearningProtectionOnlyTestConfig(test.scope).RouterLearning
					ctx.Headers = routerLearningRequestContext("session-a", "conversation-a").Headers
					ctx.SessionID = "session-a"
					ctx.VSRConversationFacts = classification.ConversationFacts{LastMessageToolResult: test.toolLoop}
					memoryKey := "session-a"
					if test.scope == config.RouterLearningScopeConversation {
						memoryKey += "/conversation-a"
					}
					sessiontelemetry.RecordSessionDecision(sessiontelemetry.SessionDecisionParams{
						SessionID: memoryKey, SelectedModel: decision.ModelRefs[0].Model,
						DecisionName: decision.Name, TurnIndex: 2,
						ActiveToolLoop: test.toolLoop, Timestamp: time.Now(),
					})
				}

				selCtx := router.buildSelectionContext(decision.ModelRefs, decision.Name, "solve this", decision.Algorithm, "", nil, ctx)
				selector := router.selectorForDecisionMethod(router.getSelectionMethod(decision.Algorithm), decision.Algorithm, ctx)
				require.NotNil(t, selector)
				base, err := selector.Select(ctx.TraceContext, selCtx)
				require.NoError(t, err)
				require.NotNil(t, base)
				assert.Len(t, base.AllScores, 2, "component scores must not collapse or add model-only entries")
				if assert.NotNil(t, base.SelectedCandidate, "selector must preserve the exact winning ref") {
					assert.Equal(t, winningEffort, base.SelectedCandidate.ReasoningEffort, "base selector winner")
				}

				selected, method, err := router.selectModelFromCandidates(selCtx, decision.Algorithm, ctx)
				require.NoError(t, err)
				require.NotNil(t, selected)
				assert.Equal(t, test.algorithm, method)
				assert.Equal(t, winningEffort, selected.ReasoningEffort, "ExtProc/Router Learning must retain the winner")
				if test.scope != "" {
					policy, ok := ctx.VSRLearningPolicies.Policy(routerLearningMethodProtection)
					require.True(t, ok, "test must exercise protection, not bypass it")
					require.NotNil(t, policy.Details.Protection)
					trace := policy.Details.Protection.trace
					require.NotNil(t, trace)
					assert.Equal(t, decision.ModelRefs[0].Model, trace.CurrentModel)
					switch {
					case test.toolLoop:
						assert.True(t, trace.HardLocked)
						assert.Equal(t, "hard_lock=tool_loop", trace.HardLockReason)
					case test.scope == config.RouterLearningScopeSession:
						assert.Equal(t, "session_scope_protect", trace.DecisionReason)
					default:
						assert.Equal(t, "stay_has_best_adjusted_score", trace.DecisionReason)
					}
				}

				// Follow the production caller's reasoning projection and dispatch path;
				// do not replace the decision's refs with the expected winner in the test.
				reasoning := applyReasoningModeFromSelectedModel(selected, decision.Name, 1, ctx)
				response, err := router.handleEntrypointModelRouting(request, "virtual", decision.Name, reasoning, selected.Model, ctx)
				require.NoError(t, err)
				require.NotNil(t, response.GetRequestBody(), "expected an outbound request, not an immediate error")
				body := response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
				var wire struct {
					Model           string `json:"model"`
					ReasoningEffort string `json:"reasoning_effort"`
				}
				require.NoError(t, json.Unmarshal(body, &wire))
				assert.Equal(t, "provider-model", wire.Model)
				assert.Equal(t, winningEffort, wire.ReasoningEffort, "winning effort must reach provider encoding")
			})
		}
	}
}

// A later request may prefer low effort, but an active tool loop must retain
// the exact high-effort choice recorded on the previous request, not just its name.
func TestSelectionProtectionRestoresExactSessionCandidate(t *testing.T) {
	for _, format := range []string{config.APIFormatOpenAI, config.APIFormatResponses} {
		t.Run(format, func(t *testing.T) {
			sessiontelemetry.ResetRouterSessionMemoryForTesting()
			t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)
			router, decision := candidateEffortTestRouter(t, config.DecisionAlgorithmMultiFactor, "low")
			model := decision.ModelRefs[0].Model
			params := router.Config.ModelConfig[model]
			params.APIFormat = format
			router.Config.ModelConfig[model] = params
			router.Config.RouterLearning = routerLearningProtectionOnlyTestConfig(config.RouterLearningScopeConversation).RouterLearning
			sessiontelemetry.RecordSessionDecision(sessiontelemetry.SessionDecisionParams{
				SessionID: "session-a/conversation-a", SelectedModel: model,
				SelectedCandidate: &decision.ModelRefs[1], DecisionName: decision.Name,
				TurnIndex: 2, ActiveToolLoop: true, Timestamp: time.Now(),
			})
			request := testNeutralRequest("virtual", "continue the tool result")
			ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
			ctx.Headers = routerLearningRequestContext("session-a", "conversation-a").Headers
			ctx.SessionID = "session-a"
			ctx.VSRSelectedDecision = decision
			ctx.VSRConversationFacts = classification.ConversationFacts{LastMessageToolResult: true}
			selCtx := router.buildSelectionContext(decision.ModelRefs, decision.Name, "continue", decision.Algorithm, "", nil, ctx)
			selected, _, err := router.selectModelFromCandidates(selCtx, decision.Algorithm, ctx)
			require.NoError(t, err)
			require.NotNil(t, selected)
			require.Equal(t, "high", selected.ReasoningEffort)
			reasoning := applyReasoningModeFromSelectedModel(selected, decision.Name, 1, ctx)
			response, err := router.handleEntrypointModelRouting(request, "virtual", decision.Name, reasoning, model, ctx)
			require.NoError(t, err)
			require.NotNil(t, response.GetRequestBody())
			var wire map[string]any
			require.NoError(t, json.Unmarshal(response.GetRequestBody().GetResponse().GetBodyMutation().GetBody(), &wire))
			if format == config.APIFormatResponses {
				reasoningObject, ok := wire["reasoning"].(map[string]any)
				require.True(t, ok)
				assert.Equal(t, "high", reasoningObject["effort"])
			} else {
				assert.Equal(t, "high", wire["reasoning_effort"])
			}
			assert.Equal(t, "low", decision.ModelRefs[0].ReasoningEffort, "shared decision must not be rewritten")
		})
	}
}

func candidateEffortTestRouter(t *testing.T, algorithm, winningEffort string) (*OpenAIRouter, *config.Decision) {
	t.Helper()
	const index = "test/intelligence@1.0.0"
	router, model := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	params := router.Config.ModelConfig[model]
	params.ReasoningFamily = "test-effort"
	params.QualityIndex = index
	params.IndexResultsByEffort = make(map[string]map[string]modelcatalog.IndexResult)
	for _, effort := range []string{"low", "high"} {
		score := 60.0
		if effort == winningEffort {
			score = 90
		}
		params.IndexResultsByEffort[effort] = map[string]modelcatalog.IndexResult{
			index: {Model: model, ReasoningEffort: effort, Index: index, Status: "available", Score: &score, Coverage: 1},
		}
	}
	router.Config.ModelConfig[model] = params
	router.Config.ReasoningFamilies = map[string]config.ReasoningFamilyConfig{
		"test-effort": {Type: config.ReasoningFamilyTypeTopLevelReasoningEffort, Parameter: "reasoning_effort", Levels: []string{"low", "high"}},
	}

	decision := &config.Decision{
		Name: "candidate-effort",
		ModelRefs: []config.ModelRef{
			{Model: model, ModelReasoningControl: config.ModelReasoningControl{UseReasoning: extprocBoolPtr(true), ReasoningEffort: "low"}},
			{Model: model, ModelReasoningControl: config.ModelReasoningControl{UseReasoning: extprocBoolPtr(true), ReasoningEffort: "high"}},
		},
		Algorithm: &config.AlgorithmConfig{Type: algorithm},
	}
	registry := selection.NewRegistry()
	t.Cleanup(func() { require.NoError(t, registry.Close()) })
	autoMixConfig := selection.DefaultAutoMixConfig()
	autoMixConfig.CostAwareRouting = false
	autoMixConfig.UsePOMDPRouter = false
	autoMix := selection.NewAutoMixSelector(autoMixConfig)
	autoMix.InitializeFromConfig(router.Config.ModelConfig)
	registry.Register(selection.MethodAutoMix, autoMix)
	router.ModelSelector = registry

	switch algorithm {
	case config.DecisionAlgorithmMultiFactor:
		decision.Algorithm.MultiFactor = &config.MultiFactorSelectionConfig{
			Weights: &config.MultiFactorWeightsConfig{Quality: 1},
			Quality: &config.QualityEvidenceConfig{Index: index, OnMissing: config.QualityEvidenceOnMissingExclude},
		}
	case config.DecisionAlgorithmHybrid:
		// Real AutoMix component, isolated from unrelated experience/embedding signals.
		decision.Algorithm.Hybrid = &config.HybridSelectionConfig{AutoMixWeight: 1, NormalizeScores: true}
	}
	return router, decision
}
