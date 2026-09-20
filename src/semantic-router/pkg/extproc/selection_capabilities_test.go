package extproc

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

type capabilityCountingSelector struct {
	selection.Selector
	calls      int
	candidates []config.ModelRef
}

func (s *capabilityCountingSelector) Select(ctx context.Context, input *selection.SelectionContext) (*selection.SelectionResult, error) {
	s.calls++
	s.candidates = cloneModelRefs(input.CandidateModels)
	return s.Selector.Select(ctx, input)
}

// Capability rejection must precede scoring: the winner among the surviving
// refs comes from the configured algorithm, not the first-fit dispatch fallback.
func TestCapabilitiesFilterBeforeScoringAndPreserveExactWinner(t *testing.T) {
	for _, primaryFormat := range []string{config.APIFormatOpenAI, config.APIFormatResponses} {
		for _, cScore := range []float64{80, 95} {
			name := primaryFormat + "/B-high-wins"
			if cScore > 90 {
				name = primaryFormat + "/C-wins-not-first-fit"
			}
			t.Run(name, func(t *testing.T) {
				router, decision, counter := capabilityRankingTestRouter(t, primaryFormat, cScore)
				request := testNeutralRequest("virtual", "draw a cat")
				request.ImageGeneration = &llmprotocol.ImageGenerationOptions{}
				ctx := routingTestContext(llmprotocol.OpenAIResponsesV1, request)
				ctx.VSRSelectedDecision = decision
				// A prior hard policy excluded B/low. Keep the full decision intact
				// to expose any accidental name-based reconstruction during encoding.
				refs := []config.ModelRef{decision.ModelRefs[0], decision.ModelRefs[2], decision.ModelRefs[3]}
				ctx.VSREligibleModelRefs = cloneModelRefs(refs)
				ctx.VSRPolicyEligibleModelRefs = cloneModelRefs(refs)
				input := &selection.SelectionContext{DecisionName: decision.Name, CandidateModels: refs}
				selected, _, err := router.selectModelFromCandidates(input, decision.Algorithm, ctx)
				require.NoError(t, err)
				require.NotNil(t, selected)
				want := "B"
				if cScore > 90 {
					want = "C"
				}
				assert.Equal(t, want, selected.Model)
				assert.Equal(t, "high", selected.ReasoningEffort)
				assert.Equal(t, 1, counter.calls, "selection must not restart during dispatch")
				assert.Equal(t, []config.ModelRef{decision.ModelRefs[2], decision.ModelRefs[3]}, counter.candidates,
					"incapable A must never be scored")
				reasoning := applyReasoningModeFromSelectedModel(selected, decision.Name, 1, ctx)
				response, err := router.handleEntrypointModelRouting(request, "virtual", decision.Name, reasoning, selected.Model, ctx)
				require.NoError(t, err)
				require.NotNil(t, response.GetRequestBody())
				var wire struct {
					Model     string `json:"model"`
					Reasoning struct {
						Effort string `json:"effort"`
					} `json:"reasoning"`
				}
				require.NoError(t, json.Unmarshal(response.GetRequestBody().GetResponse().GetBodyMutation().GetBody(), &wire))
				assert.Equal(t, want+"-wire", wire.Model)
				assert.Equal(t, "high", wire.Reasoning.Effort, "must not inherit A/off or excluded B/low")
				assert.Equal(t, want, ctx.VSRSelectedCandidate.Model)
				assert.Equal(t, 1, counter.calls)
				assert.Len(t, decision.ModelRefs, 4, "shared decision was mutated")
				for _, ref := range router.learningCandidateModels(input, ctx, config.RouterLearningCandidateSetGlobal) {
					assert.NotEqual(t, "A", ref.Model, "learning resurrected an incapable candidate")
					assert.NotEqual(t, "low", ref.ReasoningEffort, "learning resurrected an excluded effort")
				}
			})
		}
	}
}

func TestCapabilitiesRejectEmptyAndInsufficientPoolsBeforeScoring(t *testing.T) {
	for _, test := range []struct {
		name    string
		count   int
		minimum int
	}{
		{"incapable_single", 1, 0},
		{"no_capable_candidates", 2, 0},
		{"below_minimum", 3, 2},
	} {
		t.Run(test.name, func(t *testing.T) {
			router, decision, counter := capabilityRankingTestRouter(t, config.APIFormatOpenAI, 80)
			refs := []config.ModelRef{decision.ModelRefs[0], decision.ModelRefs[0], decision.ModelRefs[2]}[:test.count]
			decision.ModelRefs = refs
			decision.Algorithm.MinimumCandidates = test.minimum
			request := testNeutralRequest("virtual", "draw a cat")
			request.ImageGeneration = &llmprotocol.ImageGenerationOptions{}
			ctx := routingTestContext(llmprotocol.OpenAIResponsesV1, request)
			ctx.VSRSelectedDecision = decision
			selected, _, err := router.selectModelFromCandidates(&selection.SelectionContext{
				DecisionName: decision.Name, CandidateModels: refs,
			}, decision.Algorithm, ctx)
			require.Error(t, err)
			assert.Nil(t, selected)
			assert.Nil(t, ctx.VSRSelectedCandidate)
			assert.Zero(t, counter.calls)
		})
	}
}

func TestCapabilitiesFinalValidationDoesNotRerouteOrRescore(t *testing.T) {
	router, decision, counter := capabilityRankingTestRouter(t, config.APIFormatOpenAI, 80)
	request := testNeutralRequest("virtual", "hello")
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.VSRSelectedDecision = decision
	selected, _, err := router.selectModelFromCandidates(&selection.SelectionContext{
		DecisionName: decision.Name, CandidateModels: decision.ModelRefs,
	}, decision.Algorithm, ctx)
	require.NoError(t, err)
	require.Equal(t, "A", selected.Model)
	// A later mutation introduces an unsupported task. Final validation must
	// reject it, not replace a protected/scored choice with a first-fit sibling.
	request.ImageGeneration = &llmprotocol.ImageGenerationOptions{}
	dispatch, err := router.prepareProviderDispatch(request, selected.Model, decision.Name, false, ctx)
	require.Error(t, err)
	assert.Nil(t, dispatch)
	assert.Equal(t, "A", ctx.VSRSelectedCandidate.Model)
	assert.Equal(t, 1, counter.calls)
}

func TestCapabilitiesFinalEncodingRechecksModelTasks(t *testing.T) {
	router, decision, counter := capabilityRankingTestRouter(t, config.APIFormatResponses, 80)
	request := testNeutralRequest("virtual", "hello")
	ctx := routingTestContext(llmprotocol.OpenAIResponsesV1, request)
	ctx.VSRSelectedDecision = decision
	dispatch, err := selectCapabilityTestDispatch(router, request, decision, ctx)
	require.NoError(t, err)
	require.Equal(t, "A", dispatch.logicalModel)
	// The Responses codec supports this task, but the selected model does not.
	request.ImageGeneration = &llmprotocol.ImageGenerationOptions{}
	response, err := router.finalizeProviderDispatchResponse(dispatch, router.buildProviderDispatchResponse(dispatch, ctx), ctx)
	require.Error(t, err)
	assert.Nil(t, response)
	assert.Equal(t, "A", ctx.VSRSelectedCandidate.Model)
	assert.Equal(t, 1, counter.calls)
}

func TestCapabilitiesPreviewRequestParamsWithoutExecutingPluginTwice(t *testing.T) {
	router, decision, _ := capabilityRankingTestRouter(t, config.APIFormatResponses, 80)
	router.Config.CandidateRequirements = &config.CandidateRequirements{Context: config.CandidateContextKnownLimits}
	for model, params := range router.Config.ModelConfig {
		params.ContextWindowSize, params.MaxOutputTokens = 1000, 100
		router.Config.ModelConfig[model] = params
	}
	payload, err := config.NewStructuredPayload(config.RequestParamsPluginConfig{
		BlockedParams: []string{"top_k"}, MaxN: extprocIntPtr(1), DefaultMaxTokens: config.FixedOutputTokenDefault(64),
	})
	require.NoError(t, err)
	decision.Plugins = []config.DecisionPlugin{{Type: "request_params", Configuration: payload}}
	request := testNeutralRequest("virtual", "hello")
	request.CandidateCount = llmprotocol.Int64(5)
	request.Sampling.TopK = llmprotocol.Int64(3)
	ctx := routingTestContext(llmprotocol.OpenAIResponsesV1, request)
	ctx.VSRSelectedDecision = decision
	// Strict demand uses the policy-adjusted request, not a stale ingress budget.
	ctx.VSRContextTokenCount = 10000
	counter := metrics.RequestParamsMaxNCapped.WithLabelValues(config.RoutingDecisionKey(ctx.Routing.RecipeName(), decision.Name))
	before := testutil.ToFloat64(counter)
	selected, _, err := router.selectModelFromCandidates(&selection.SelectionContext{
		DecisionName: decision.Name, CandidateModels: decision.ModelRefs,
	}, decision.Algorithm, ctx)
	require.NoError(t, err)
	require.Equal(t, "A", selected.Model, "preview must account for parameters removed before encoding")
	assert.Equal(t, before, testutil.ToFloat64(counter), "capability preview executed plugin metrics")
	assert.Equal(t, int64(5), *request.CandidateCount, "preview mutated client parameters")
	assert.Nil(t, request.Sampling.MaxOutputTokens, "preview materialized the default on the live request")
	reasoning := applyReasoningModeFromSelectedModel(selected, decision.Name, 1, ctx)
	_, err = router.prepareProviderDispatch(request, selected.Model, decision.Name, reasoning.UseReasoning, ctx)
	require.NoError(t, err)
	assert.Equal(t, before+1, testutil.ToFloat64(counter))
	assert.Equal(t, int64(1), *request.CandidateCount)
	require.NotNil(t, request.Sampling.MaxOutputTokens)
	assert.Equal(t, int64(64), *request.Sampling.MaxOutputTokens)
	assert.Nil(t, request.Sampling.TopK)
}

func TestCapabilitiesIncludeRetainedHistoryWithoutDuplicatingIt(t *testing.T) {
	router, decision, _ := capabilityRankingTestRouter(t, config.APIFormatOpenAI, 80)
	params := router.Config.ModelConfig["A"]
	params.Capabilities = []string{"audio_input"}
	router.Config.ModelConfig["A"] = params
	params = router.Config.ModelConfig["B"]
	params.Capabilities = []string{"image_input"}
	router.Config.ModelConfig["B"] = params
	request := testNeutralRequest("virtual", "describe the earlier image")
	request.PreviousResponseID = "resp_previous"
	request.Store = extprocBoolPtr(true)
	ctx := routingTestContext(llmprotocol.OpenAIResponsesV1, request)
	ctx.VSRSelectedDecision = decision
	ctx.VSREligibleModelRefs = []config.ModelRef{decision.ModelRefs[0], decision.ModelRefs[2], decision.ModelRefs[3]}
	imageContent, err := json.Marshal([]responseapi.ContentPart{{
		Type: responseapi.ContentTypeInputImage, ImageURL: "https://example.com/history.png",
	}})
	require.NoError(t, err)
	ctx.ResponseObjectState = &ResponseObjectState{ConversationHistory: []*responseapi.StoredResponse{{
		Input:      []responseapi.InputItem{{Type: "message", Role: "user", Content: imageContent}},
		OutputText: "earlier answer",
	}}}
	dispatch, err := selectCapabilityTestDispatch(router, request, decision, ctx)
	require.NoError(t, err)
	assert.Equal(t, "B", dispatch.logicalModel, "history's image capability was ignored")
	assert.Len(t, request.Messages, 3, "retained history was prepended more than once")
	assert.True(t, ctx.ResponseObjectState.ProviderContextApplied)
	assert.Empty(t, request.PreviousResponseID)
	assert.Nil(t, request.Store)
}

// Exercise the normal selector-to-dispatch boundary in the capability fixtures.
// A final dispatch check alone must never choose a different candidate.
func selectCapabilityTestDispatch(router *OpenAIRouter, request *llmprotocol.Request, decision *config.Decision, ctx *RequestContext) (*providerDispatch, error) {
	selected, _, err := router.selectModelFromCandidates(&selection.SelectionContext{
		DecisionName: decision.Name, CandidateModels: decision.ModelRefs,
	}, decision.Algorithm, ctx)
	if err != nil {
		return nil, err
	}
	reasoning := applyReasoningModeFromSelectedModel(selected, decision.Name, 1, ctx)
	return router.prepareProviderDispatch(request, selected.Model, decision.Name, reasoning.UseReasoning, ctx)
}

func capabilityRankingTestRouter(t *testing.T, primaryFormat string, cScore float64) (*OpenAIRouter, *config.Decision, *capabilityCountingSelector) {
	t.Helper()
	router, decision := candidateEffortTestRouter(t, config.DecisionAlgorithmAutoMix, "high")
	original := decision.ModelRefs[0].Model
	base := router.Config.ModelConfig[original]
	delete(router.Config.ModelConfig, original)
	index := base.QualityIndex
	for _, model := range []struct {
		name, format string
		score        float64
		capabilities []string
	}{
		{"A", primaryFormat, 99, []string{"image_input"}},
		{"B", config.APIFormatResponses, 90, []string{"image_generation"}},
		{"C", config.APIFormatResponses, cScore, []string{"image_generation"}},
	} {
		params := base
		params.APIFormat, params.Capabilities = model.format, model.capabilities
		params.ExternalModelIDs = map[string]string{"vllm": model.name + "-wire"}
		params.IndexResultsByEffort = make(map[string]map[string]modelcatalog.IndexResult)
		for _, effort := range []string{"low", "high"} {
			score := model.score
			if effort == "low" {
				score = 60
			}
			params.IndexResultsByEffort[effort] = map[string]modelcatalog.IndexResult{
				index: {Model: model.name, Index: index, ReasoningEffort: effort, Status: "available", Score: &score, Coverage: 1},
			}
		}
		router.Config.ModelConfig[model.name] = params
	}
	router.Config.DefaultModel = "A"
	decision.ModelRefs = []config.ModelRef{
		{Model: "A", ModelReasoningControl: config.ModelReasoningControl{UseReasoning: extprocBoolPtr(false), ReasoningEffort: "high"}},
		{Model: "B", ModelReasoningControl: config.ModelReasoningControl{UseReasoning: extprocBoolPtr(false), ReasoningEffort: "low"}},
		{Model: "B", ModelReasoningControl: config.ModelReasoningControl{UseReasoning: extprocBoolPtr(true), ReasoningEffort: "high"}},
		{Model: "C", ModelReasoningControl: config.ModelReasoningControl{UseReasoning: extprocBoolPtr(true), ReasoningEffort: "high"}},
	}
	router.Config.Decisions = []config.Decision{*decision}
	cfg := selection.DefaultAutoMixConfig()
	cfg.CostAwareRouting, cfg.UsePOMDPRouter = false, false
	selector := selection.NewAutoMixSelector(cfg)
	selector.InitializeFromConfig(router.Config.ModelConfig)
	counter := &capabilityCountingSelector{Selector: selector}
	router.ModelSelector.Register(selection.MethodAutoMix, counter)
	return router, decision, counter
}
