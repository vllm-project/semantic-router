package extproc

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func TestContextEligibleModelRefsFiltersOnlyKnownInsufficientWindows(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			"small": {ContextWindowSize: 8_192},
			"large": {ContextWindowSize: 32_768},
			"unset": {},
		}},
	}}

	eligible, excluded := router.contextEligibleModelRefs(nil, []config.ModelRef{
		{Model: "small"},
		{Model: "large"},
		{Model: "unset"},
		{Model: "unregistered"},
	}, 16_000)

	assert.Equal(t, 1, excluded)
	assertModelRefs(t, eligible, []string{"large", "unset", "unregistered"})
}

func TestSelectDecisionRuntimeModelRejectsAllKnownInsufficientWindows(t *testing.T) {
	decisionConfig := &config.Decision{
		Name: "known-small-models",
		ModelRefs: []config.ModelRef{
			{Model: "small-a"},
			{Model: "small-b"},
		},
	}
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			"small-a": {ContextWindowSize: 4_096},
			"small-b": {ContextWindowSize: 8_192},
		}},
	}}
	ctx := &RequestContext{VSRContextTokenCount: 12_000}
	withCounts(router, ctx, map[string]int{"small-a": 12_000, "small-b": 12_000})

	_, _, err := router.selectDecisionRuntimeModel(
		&decision.DecisionResult{Decision: decisionConfig},
		decisionConfig.Name,
		"long request",
		"",
		1,
		ctx,
	)

	require.Error(t, err)
	assert.ErrorIs(t, err, errNoContextEligibleDecisionModel)
	var refusal *contextLengthError
	require.ErrorAs(t, err, &refusal)
	assert.Equal(t, 8_192, refusal.window, "the refusal names the widest window")
	assert.Equal(t, 12_000, refusal.tokens)
	assert.Empty(t, ctx.VSREligibleModelRefs)
}

// Only the estimate says no candidate fits: nothing proves the request too
// long, so the widest candidate serves it and its backend decides (#6319).
func TestSelectDecisionRuntimeModelKeepsTheWidestCandidateOnlyTheEstimateRulesOut(t *testing.T) {
	decisionConfig := &config.Decision{
		Name:      "known-small-models",
		ModelRefs: []config.ModelRef{{Model: "small-a"}, {Model: "small-b"}},
		Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmStatic},
	}
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			"small-a": {ContextWindowSize: 4_096},
			"small-b": {ContextWindowSize: 8_192},
		}},
	}}
	ctx := &RequestContext{VSRContextTokenCount: 12_000}

	selected, _, err := router.selectDecisionRuntimeModel(
		&decision.DecisionResult{Decision: decisionConfig},
		decisionConfig.Name,
		"long request",
		"",
		1,
		ctx,
	)

	require.NoError(t, err)
	assert.Equal(t, "small-b", selected)
	assertModelRefs(t, ctx.VSREligibleModelRefs, []string{"small-b"})
	assert.False(t, router.modelExceedsContextWindow(ctx, "small-b", ctx.VSRContextTokenCount),
		"later checks of the request keep the candidate the filter kept")
	assert.True(t, router.modelExceedsContextWindow(ctx, "small-a", ctx.VSRContextTokenCount))
}

// Reserves in the estimate (structured bytes beyond the text rate, media) do
// not exclude a candidate on their own: the candidates the estimate without
// reserves fits serve the request. Above every window the widest candidate
// still serves it, unless the Router's counts rule every candidate out.
func TestSelectDecisionRuntimeModelDropsReservesOrKeepsTheWidestAboveEveryWindow(t *testing.T) {
	candidates := []config.ModelRef{{Model: "small-a"}, {Model: "small-b"}}
	for _, tc := range []struct {
		name     string
		refs     []config.ModelRef
		reserve  int
		counts   map[string]int
		selected string
		dropped  bool
	}{
		{name: "candidates fit without reserves", refs: candidates, reserve: 9_000, selected: "small-a", dropped: true},
		{name: "one candidate fits without reserves", refs: candidates, reserve: 5_000, selected: "small-b", dropped: true},
		{name: "no candidate fits without reserves", refs: candidates, reserve: 1_000, selected: "small-b"},
		{name: "counts rule out every candidate", refs: candidates, reserve: 1_000, counts: map[string]int{"small-a": 11_000, "small-b": 11_000}},
		{name: "default model fits without reserves", reserve: 9_000, selected: "small-a", dropped: true},
		{name: "default model does not fit without reserves", reserve: 1_000, selected: "small-a"},
		{name: "a count rules out the default model", reserve: 1_000, counts: map[string]int{"small-a": 11_000}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router := &OpenAIRouter{Config: &config.RouterConfig{
				BackendModels: config.BackendModels{DefaultModel: "small-a", ModelConfig: map[string]config.ModelParams{
					"small-a": {ContextWindowSize: 4_096},
					"small-b": {ContextWindowSize: 8_192},
				}},
			}}
			decisionConfig := &config.Decision{
				Name: "known-small-models", ModelRefs: tc.refs,
				Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmStatic},
			}
			ctx := &RequestContext{VSRContextTokenCount: 12_000, VSRContextReserveTokens: tc.reserve}
			if tc.counts != nil {
				withCounts(router, ctx, tc.counts)
			}

			selected, _, err := router.selectDecisionRuntimeModel(
				&decision.DecisionResult{Decision: decisionConfig},
				decisionConfig.Name,
				"long request",
				"",
				1,
				ctx,
			)

			if tc.selected == "" {
				assert.ErrorIs(t, err, errNoContextEligibleDecisionModel)
				var refusal *contextLengthError
				assert.ErrorAs(t, err, &refusal)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tc.selected, selected)
			if tc.dropped {
				assert.Equal(t, 12_000-tc.reserve, ctx.VSRContextTokenCount, "the request continues without its reserves")
				assert.Zero(t, ctx.VSRContextReserveTokens)
			} else {
				assert.Equal(t, 12_000, ctx.VSRContextTokenCount)
			}
		})
	}
}

func TestSelectModelForEvalRefusesOnlyWithoutReservesAboveEveryWindow(t *testing.T) {
	decisionConfig := &config.Decision{
		Name:      "known-small-models",
		ModelRefs: []config.ModelRef{{Model: "small"}},
		Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmStatic},
	}
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			"small": {ContextWindowSize: 3_000},
		}},
	}}
	request := testNeutralRequest("vllm-sr/auto", "write the file")
	request.Messages = append(request.Messages, llmprotocol.Message{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{
		Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: "call_1", Name: "write", Arguments: strings.Repeat("x", 6000)},
	}}})
	estimate := llmprotocol.EstimateInput(request)

	selection := router.SelectModelForEval(services.EvalModelSelectionInput{
		Decision: decisionConfig, SemanticRequest: request,
		ContextTokenCount: estimate.Tokens, ContextReserveTokens: estimate.ReserveTokens,
	})
	assert.Equal(t, services.EvalSelectionSelected, selection.Status)
	assert.Equal(t, "small", selection.SelectedModel)

	selection = router.SelectModelForEval(services.EvalModelSelectionInput{
		Decision: decisionConfig, SemanticRequest: request, ContextTokenCount: estimate.Tokens,
	})
	assert.Equal(t, services.EvalSelectionUnavailable, selection.Status)
}

func TestSelectDecisionRuntimeModelFiltersKnownInsufficientWindowBeforeSelection(t *testing.T) {
	decisionConfig := &config.Decision{
		Name:      "mixed-context-models",
		ModelRefs: []config.ModelRef{{Model: "small"}, {Model: "large"}},
		Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmStatic},
	}
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			"small": {ContextWindowSize: 4_096},
			"large": {ContextWindowSize: 32_768},
		}},
	}}
	ctx := &RequestContext{VSRContextTokenCount: 12_000}

	selected, _, err := router.selectDecisionRuntimeModel(
		&decision.DecisionResult{Decision: decisionConfig},
		decisionConfig.Name,
		"long request",
		"",
		1,
		ctx,
	)

	require.NoError(t, err)
	assert.Equal(t, "large", selected)
	assertModelRefs(t, ctx.VSREligibleModelRefs, []string{"large"})
}

func TestSelectDecisionRuntimeModelPreservesMinimumCandidateContract(t *testing.T) {
	decisionConfig := &config.Decision{
		Name:      "quorum-context-models",
		ModelRefs: []config.ModelRef{{Model: "small"}, {Model: "large"}},
		Algorithm: &config.AlgorithmConfig{
			Type:              config.DecisionAlgorithmFusion,
			MinimumCandidates: 2,
			Fusion:            &config.FusionAlgorithmConfig{},
		},
	}
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			"small": {ContextWindowSize: 4_096},
			"large": {ContextWindowSize: 32_768},
		}},
	}}
	ctx := &RequestContext{VSRContextTokenCount: 12_000}
	withCounts(router, ctx, map[string]int{"small": 12_000})

	_, _, err := router.selectDecisionRuntimeModel(
		&decision.DecisionResult{Decision: decisionConfig},
		decisionConfig.Name,
		"long request",
		"",
		1,
		ctx,
	)

	require.Error(t, err)
	assert.ErrorIs(t, err, errNoContextEligibleDecisionModel)
	assert.Contains(t, err.Error(), "requires at least 2 eligible candidates")
	assertModelRefs(t, ctx.VSREligibleModelRefs, []string{"large"})
}

// The estimate alone does not cost a decision the candidates its minimum
// needs: the one it rules out stays.
func TestSelectDecisionRuntimeModelMeetsTheMinimumDespiteTheEstimate(t *testing.T) {
	decisionConfig := &config.Decision{
		Name:      "quorum-context-models",
		ModelRefs: []config.ModelRef{{Model: "small"}, {Model: "large"}},
		Algorithm: &config.AlgorithmConfig{
			Type:              config.DecisionAlgorithmFusion,
			MinimumCandidates: 2,
			Fusion:            &config.FusionAlgorithmConfig{},
		},
	}
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			"small": {ContextWindowSize: 4_096},
			"large": {ContextWindowSize: 32_768},
		}},
	}}
	ctx := &RequestContext{VSRContextTokenCount: 12_000}

	_, _, err := router.selectDecisionRuntimeModel(
		&decision.DecisionResult{Decision: decisionConfig},
		decisionConfig.Name,
		"long request",
		"",
		1,
		ctx,
	)

	assert.NotErrorIs(t, err, errNoContextEligibleDecisionModel)
	assertModelRefs(t, ctx.VSREligibleModelRefs, []string{"small", "large"})
}

// The fallback counts distinct candidates as the minimum check does: a
// duplicate ref that fits must not hide that the minimum needs the one the
// reserves exclude.
func TestSelectDecisionRuntimeModelMeetsTheMinimumWithoutReservesDespiteDuplicates(t *testing.T) {
	decisionConfig := &config.Decision{
		Name:      "quorum-context-models",
		ModelRefs: []config.ModelRef{{Model: "large"}, {Model: "large"}, {Model: "small"}},
		Algorithm: &config.AlgorithmConfig{
			Type:              config.DecisionAlgorithmFusion,
			MinimumCandidates: 2,
			Fusion:            &config.FusionAlgorithmConfig{},
		},
	}
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			"small": {ContextWindowSize: 4_096},
			"large": {ContextWindowSize: 32_768},
		}},
	}}
	ctx := &RequestContext{VSRContextTokenCount: 12_000, VSRContextReserveTokens: 9_000}

	_, _, err := router.selectDecisionRuntimeModel(
		&decision.DecisionResult{Decision: decisionConfig},
		decisionConfig.Name,
		"long request",
		"",
		1,
		ctx,
	)

	assert.NotErrorIs(t, err, errNoContextEligibleDecisionModel)
	assertModelRefs(t, ctx.VSREligibleModelRefs, []string{"large", "large", "small"})
}

func TestSelectDecisionRuntimeModelRejectsUnmaterializedMinimumCandidateContract(t *testing.T) {
	decisionConfig := &config.Decision{
		Name: "unmaterialized-panel",
		Algorithm: &config.AlgorithmConfig{
			Type:              config.DecisionAlgorithmFusion,
			MinimumCandidates: 2,
			Fusion:            &config.FusionAlgorithmConfig{},
		},
	}
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{DefaultModel: "fallback"},
	}}
	ctx := &RequestContext{VSRContextTokenCount: 128}

	_, _, err := router.selectDecisionRuntimeModel(
		&decision.DecisionResult{Decision: decisionConfig},
		decisionConfig.Name,
		"request",
		"",
		1,
		ctx,
	)

	require.Error(t, err)
	assert.ErrorIs(t, err, errNoContextEligibleDecisionModel)
	assert.Contains(t, err.Error(), "requires at least 2 eligible candidates")
}

func TestBuildLooperRequestUsesContextEligibleDecisionModels(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	decisionConfig := &config.Decision{
		Name:      "panel",
		ModelRefs: []config.ModelRef{{Model: "small"}, {Model: "large"}},
		Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmReMoM},
	}
	ctx := &RequestContext{
		VSRContextTokenCount: 12_000,
		VSREligibleModelRefs: []config.ModelRef{{Model: "large"}},
	}

	request, response := router.buildLooperRequest(
		&llmprotocol.Request{
			Model:      "vllm-sr/auto",
			Generation: 1,
			Messages: []llmprotocol.Message{{
				Role: llmprotocol.RoleUser,
				Content: []llmprotocol.Content{{
					Kind: llmprotocol.ContentText,
					Text: "test request",
				}},
			}},
		},
		decisionConfig,
		ctx,
	)

	require.Nil(t, response)
	require.NotNil(t, request)
	assertModelRefs(t, request.ModelRefs, []string{"large"})
	assert.Equal(t, 12_000, request.BaseContextTokens)
}

func TestSelectDecisionRuntimeModelRejectsIneligibleExplicitLooperModel(t *testing.T) {
	decisionConfig := &config.Decision{
		Name:      "workflow",
		ModelRefs: []config.ModelRef{{Model: "large"}, {Model: "small-planner"}},
		Algorithm: &config.AlgorithmConfig{
			Type: config.DecisionAlgorithmWorkflows,
			Workflows: &config.WorkflowsAlgorithmConfig{
				Mode:    config.WorkflowModeDynamic,
				Planner: config.WorkflowPlannerConfig{Model: "small-planner"},
			},
		},
	}
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			"large":         {ContextWindowSize: 32_768},
			"small-planner": {ContextWindowSize: 4_096},
		}},
	}}
	ctx := &RequestContext{VSRContextTokenCount: 12_000}
	withCounts(router, ctx, map[string]int{"small-planner": 12_000})

	_, _, err := router.selectDecisionRuntimeModel(
		&decision.DecisionResult{Decision: decisionConfig},
		decisionConfig.Name,
		"long request",
		"",
		1,
		ctx,
	)

	require.Error(t, err)
	assert.ErrorIs(t, err, errNoContextEligibleDecisionModel)
	var refusal *contextLengthError
	require.ErrorAs(t, err, &refusal)
	assert.Equal(t, 4_096, refusal.window)
}

// Looper stage checks refuse a stage whose base plus growth exceeds a window,
// so the base they start from leaves out the estimate's reserves.
func TestBuildLooperRequestCountsTheBaseWithoutReserves(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	decisionConfig := &config.Decision{
		Name:      "panel",
		ModelRefs: []config.ModelRef{{Model: "large"}},
		Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmReMoM},
	}
	ctx := &RequestContext{VSRContextTokenCount: 12_000, VSRContextReserveTokens: 9_000}

	request, response := router.buildLooperRequest(
		&llmprotocol.Request{
			Model:      "vllm-sr/auto",
			Generation: 1,
			Messages: []llmprotocol.Message{{
				Role:    llmprotocol.RoleUser,
				Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "test request"}},
			}},
		},
		decisionConfig,
		ctx,
	)

	require.Nil(t, response)
	require.NotNil(t, request)
	assert.Equal(t, 3_000, request.BaseContextTokens)
}

func TestSelectDecisionRuntimeModelKeepsExplicitLooperModelAboveItsWindowOnlyByReserves(t *testing.T) {
	decisionConfig := &config.Decision{
		Name:      "workflow",
		ModelRefs: []config.ModelRef{{Model: "large"}, {Model: "small-planner"}},
		Algorithm: &config.AlgorithmConfig{
			Type: config.DecisionAlgorithmWorkflows,
			Workflows: &config.WorkflowsAlgorithmConfig{
				Mode:    config.WorkflowModeDynamic,
				Planner: config.WorkflowPlannerConfig{Model: "small-planner"},
			},
		},
	}
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			"large":         {ContextWindowSize: 32_768},
			"small-planner": {ContextWindowSize: 4_096},
		}},
	}}

	_, _, err := router.selectDecisionRuntimeModel(
		&decision.DecisionResult{Decision: decisionConfig},
		decisionConfig.Name,
		"long request",
		"",
		1,
		&RequestContext{VSRContextTokenCount: 12_000, VSRContextReserveTokens: 9_000},
	)

	assert.NotErrorIs(t, err, errNoContextEligibleDecisionModel)
}

func TestSelectModelForEvalFiltersContextIneligibleCandidate(t *testing.T) {
	decisionConfig := &config.Decision{
		Name:      "mixed-context-models",
		ModelRefs: []config.ModelRef{{Model: "small"}, {Model: "large"}},
		Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmStatic},
	}
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			"small": {ContextWindowSize: 4_096},
			"large": {ContextWindowSize: 32_768},
		}},
	}}

	selection := router.SelectModelForEval(services.EvalModelSelectionInput{
		Decision:          decisionConfig,
		ContextTokenCount: 12_000,
	})

	assert.Equal(t, services.EvalSelectionSelected, selection.Status)
	assert.Equal(t, "large", selection.SelectedModel)
}

func TestSelectModelForEvalPreservesMinimumCandidateContract(t *testing.T) {
	decisionConfig := &config.Decision{
		Name:      "eval-quorum-context-models",
		ModelRefs: []config.ModelRef{{Model: "small"}, {Model: "large"}},
		Algorithm: &config.AlgorithmConfig{
			Type:              config.DecisionAlgorithmStatic,
			MinimumCandidates: 2,
		},
	}
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			"small": {ContextWindowSize: 4_096},
			"large": {ContextWindowSize: 32_768},
		}},
	}}

	selection := router.SelectModelForEval(services.EvalModelSelectionInput{
		Decision:          decisionConfig,
		ContextTokenCount: 12_000,
	})

	assert.Equal(t, services.EvalSelectionUnavailable, selection.Status)
	assert.Contains(t, selection.Reason, "requires at least 2 eligible candidates")
}

func TestRouterLearningExpansionFiltersKnownInsufficientWindows(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{
			DefaultModel: "small",
			ModelConfig: map[string]config.ModelParams{
				"small": {ContextWindowSize: 8_192},
				"large": {ContextWindowSize: 32_768},
			},
		},
		IntelligentRouting: config.IntelligentRouting{Decisions: []config.Decision{
			{Name: "small", Tier: 2, ModelRefs: []config.ModelRef{{Model: "small"}}},
			{Name: "large", Tier: 2, ModelRefs: []config.ModelRef{{Model: "large"}}},
		}},
	}}
	ctx := &RequestContext{
		VSRContextTokenCount: 16_000,
		VSRSelectedDecision:  &router.Config.Decisions[0],
	}
	selCtx := &selection.SelectionContext{
		DecisionName:    "small",
		CandidateModels: []config.ModelRef{{Model: "small"}},
	}

	assertModelRefs(
		t,
		router.learningCandidateModels(selCtx, ctx, config.RouterLearningCandidateSetTier),
		[]string{"large"},
	)
}
