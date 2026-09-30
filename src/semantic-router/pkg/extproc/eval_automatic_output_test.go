package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/lookuptable"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

func nativePreviewInput(t *testing.T, ctx *RequestContext) services.EvalModelSelectionInput {
	t.Helper()
	return services.EvalModelSelectionInput{Context: t.Context(), Decision: ctx.VSRSelectedDecision, SemanticRequest: ctx.SemanticRequest, ContextTokenCount: 7, Query: "synthetic request"}
}

func addWidePreviewCandidate(r *OpenAIRouter, ctx *RequestContext) string {
	first := ctx.VSRSelectedDecision.ModelRefs[0].Model
	wide := r.Config.ModelConfig[first]
	wide.ContextWindowSize = 65536
	wide.MaxOutputTokens = 65536
	wide.ExternalModelIDs = nil
	r.Config.ModelConfig["wide"] = wide
	ctx.VSRSelectedDecision.ModelRefs = append(ctx.VSRSelectedDecision.ModelRefs, config.ModelRef{Model: "wide"})
	return first
}

func TestNativePreviewUsesRenderedInputsAndForecastNotMaximum(t *testing.T) {
	calls := 0
	r, ctx := automaticFixture(t, "hello", func(w http.ResponseWriter, req *http.Request) {
		calls++
		require.Equal(t, "/v1/chat/completions/render", req.URL.Path)
		var body map[string]any
		require.NoError(t, json.NewDecoder(req.Body).Decode(&body))
		require.Nil(t, body["max_tokens"])
		input, capacity := 100, 32768
		if body["model"] == "wide" {
			input, capacity = 3000, 65536
		}
		require.NoError(t, json.NewEncoder(w).Encode(map[string]any{"model": body["model"], "token_ids": make([]int, input), "sampling_params": map[string]any{"max_tokens": capacity - input}}))
	})
	first := addWidePreviewCandidate(r, ctx)
	for model, price := range map[string]float64{first: 1, "wide": 0.75} {
		p := addTestQuality(r.Config.ModelConfig[model], .9)
		p.Pricing = config.ModelPricing{Currency: "USD", PromptPer1M: price, CompletionPer1M: price}
		r.Config.ModelConfig[model] = p
	}
	forecast := 4096
	ctx.VSRSelectedDecision.Algorithm = &config.AlgorithmConfig{Type: config.DecisionAlgorithmMultiFactor, MultiFactor: &config.MultiFactorSelectionConfig{
		ExpectedOutputTokens: &forecast, Quality: &config.QualityEvidenceConfig{Index: testIntelligenceIndex},
		Objective: &config.MultiFactorObjectiveConfig{Strategy: config.MultiFactorObjectiveLexicographic, Priorities: []config.MultiFactorPriorityConfig{{Factor: config.MultiFactorFactorCost}}},
	}}
	before, err := json.Marshal(ctx.SemanticRequest)
	require.NoError(t, err)
	result := r.SelectModelForEval(nativePreviewInput(t, ctx))
	require.Equal(t, services.EvalSelectionSelected, result.Status, "%+v", result)
	require.Equal(t, first, result.SelectedModel, "provider input 100 vs 3000 must affect forecast cost")
	forecast = 12000
	result = r.SelectModelForEval(nativePreviewInput(t, ctx))
	require.Equal(t, services.EvalSelectionSelected, result.Status, "%+v", result)
	require.Equal(t, "wide", result.SelectedModel, "forecast changed, but native caps did not: cap is not the forecast")
	ctx.VSRSelectedDecision.Algorithm.MultiFactor.ExpectedOutputTokens = nil
	result = r.SelectModelForEval(nativePreviewInput(t, ctx))
	require.Equal(t, services.EvalSelectionUnavailable, result.Status, "missing cost forecast must not select a fallback")
	after, err := json.Marshal(ctx.SemanticRequest)
	require.NoError(t, err)
	require.Equal(t, before, after)
	require.Nil(t, r.ContextCompression, "configured overflow compression must not initialize on fitting inputs")
	require.Equal(t, 6, calls)
}

func TestNativePreviewOverflowNeverCompressesOrSwitchesRouteAction(t *testing.T) {
	for _, tt := range []struct {
		name                     string
		length                   int
		wide, compression, route bool
		status                   string
	}{
		{"partial", 40000, true, true, false, services.EvalSelectionSelected},
		{"all-needs-compression", 70000, true, true, false, services.EvalSelectionExecutionRequired},
		{"all-no-compression", 70000, true, false, false, services.EvalSelectionUnavailable},
		{"route-destination-only", 40000, true, false, true, services.EvalSelectionUnavailable},
		{"route-fits", 100, true, true, true, services.EvalSelectionSelected},
	} {
		t.Run(tt.name, func(t *testing.T) {
			calls := 0
			r, ctx := automaticFixture(t, strings.Repeat("a", tt.length), renderMock(t, &calls, 0, 0))
			first := addWidePreviewCandidate(r, ctx)
			if !tt.compression {
				ctx.VSRSelectedDecision.Plugins = ctx.VSRSelectedDecision.Plugins[1:]
			}
			if tt.route {
				ctx.VSRSelectedDecision.Action = &config.DecisionAction{Type: config.DecisionActionRoute, Destination: first}
			}
			before, err := json.Marshal(ctx.SemanticRequest)
			require.NoError(t, err)
			result := r.SelectModelForEval(nativePreviewInput(t, ctx))
			require.Equal(t, tt.status, result.Status, "%+v", result)
			if result.Status == services.EvalSelectionSelected {
				want := "wide"
				if tt.route {
					want = first
				}
				require.Equal(t, want, result.SelectedModel)
			} else {
				require.Empty(t, result.SelectedModel)
			}
			after, err := json.Marshal(ctx.SemanticRequest)
			require.NoError(t, err)
			require.Equal(t, before, after)
			require.Nil(t, r.ContextCompression)
			require.False(t, ctx.ContextCompressionApplied)
			expected := 2
			if tt.route {
				expected = 1
			}
			require.Equal(t, expected, calls)
		})
	}
}

func TestNativePreviewRenderFailureIsNotOverflowOrFallback(t *testing.T) {
	for _, tt := range []struct {
		name              string
		status, reduction int
	}{{"renderer-error", 500, 0}, {"hidden-cap", 0, 100}} {
		t.Run(tt.name, func(t *testing.T) {
			calls := 0
			r, ctx := automaticFixture(t, "hello", renderMock(t, &calls, tt.reduction, tt.status))
			addWidePreviewCandidate(r, ctx)
			result := r.SelectModelForEval(nativePreviewInput(t, ctx))
			require.Equal(t, services.EvalSelectionUnavailable, result.Status, "%+v", result)
			require.Empty(t, result.SelectedModel)
			require.Equal(t, 1, calls)
			require.Nil(t, r.ContextCompression)
		})
	}
}

func TestNativePreviewCancellationStopsRendererAndRemainingCandidates(t *testing.T) {
	var calls atomic.Int32
	started := make(chan struct{})
	stopped := make(chan struct{})
	release := make(chan struct{})
	r, ctx := automaticFixture(t, "hello", func(w http.ResponseWriter, req *http.Request) {
		calls.Add(1)
		_, err := io.Copy(io.Discard, req.Body)
		require.NoError(t, err)
		close(started)
		select {
		case <-req.Context().Done():
			close(stopped)
		case <-release:
		}
	})
	t.Cleanup(func() { close(release) })
	addWidePreviewCandidate(r, ctx)
	requestContext, cancel := context.WithCancel(t.Context())
	defer cancel()
	input := nativePreviewInput(t, ctx)
	input.Context = requestContext
	resultCh := make(chan services.EvalModelSelection, 1)
	go func() { resultCh <- r.SelectModelForEval(input) }()
	select {
	case <-started:
	case <-time.After(3 * time.Second):
		t.Fatal("renderer did not start")
	}
	cancel()
	select {
	case result := <-resultCh:
		require.Equal(t, services.EvalSelectionUnavailable, result.Status)
		require.Empty(t, result.SelectedModel)
	case <-time.After(3 * time.Second):
		t.Fatal("preview ignored cancellation")
	}
	select {
	case <-stopped:
	case <-time.After(3 * time.Second):
		t.Fatal("HTTP render did not receive cancellation")
	}
	require.EqualValues(t, 1, calls.Load())
}

func TestNativePreviewUsesConfiguredSystemReasoningAndBlockedCap(t *testing.T) {
	calls := 0
	var rendered map[string]any
	r, ctx := automaticFixture(t, "hello", func(w http.ResponseWriter, req *http.Request) {
		calls++
		require.Equal(t, "/v1/chat/completions/render", req.URL.Path)
		require.NoError(t, json.NewDecoder(req.Body).Decode(&rendered))
		require.NoError(t, json.NewEncoder(w).Encode(map[string]any{"model": rendered["model"], "token_ids": make([]int, 123), "sampling_params": map[string]any{"max_tokens": 32768 - 123}}))
	})
	first := ctx.VSRSelectedDecision.ModelRefs[0].Model
	p := r.Config.ModelConfig[first]
	p.ReasoningFamily = "reasoning"
	r.Config.ModelConfig[first] = p
	r.Config.ReasoningFamilies = map[string]config.ReasoningFamilyConfig{"reasoning": {Type: config.ReasoningFamilyTypeTopLevelReasoningEffort, Parameter: "reasoning_effort"}}
	ctx.VSRSelectedDecision.ModelRefs[0].UseReasoning = extprocBoolPtr(true)
	ctx.VSRSelectedDecision.ModelRefs[0].ReasoningEffort = "high"
	payload, err := config.NewStructuredPayload(map[string]any{"system_prompt": "Configured instruction"})
	require.NoError(t, err)
	ctx.VSRSelectedDecision.Plugins = append(ctx.VSRSelectedDecision.Plugins, config.DecisionPlugin{Type: "system_prompt", Configuration: payload})
	ctx.SemanticRequest.Sampling.MaxOutputTokens = llmprotocol.Int64(12)
	ctx.SemanticRequest.Tools = []llmprotocol.Tool{{Name: "calculator", InputSchema: json.RawMessage(`{"type":"object","properties":{"n":{"type":"integer"}}}`)}}
	ctx.SemanticRequest.ToolChoice.Mode = llmprotocol.ToolChoiceRequired
	setAutomaticBlockedParams(t, ctx, []string{"max_tokens"})
	before, err := json.Marshal(ctx.SemanticRequest)
	require.NoError(t, err)
	metricsBefore := nativePreviewObservationCounters(t)
	result := r.SelectModelForEval(nativePreviewInput(t, ctx))
	require.Equal(t, services.EvalSelectionSelected, result.Status, "%+v", result)
	require.Equal(t, "high", rendered["reasoning_effort"])
	require.Nil(t, rendered["max_tokens"])
	require.Contains(t, fmt.Sprint(rendered["messages"]), "Configured instruction")
	require.Len(t, rendered["tools"], 1)
	after, err := json.Marshal(ctx.SemanticRequest)
	require.NoError(t, err)
	require.Equal(t, before, after, "configured defaults cannot mutate prompt or nested tool schemas")
	require.Equal(t, 1, calls)
	require.Equal(t, metricsBefore, nativePreviewObservationCounters(t), "rendering must not record a reasoning decision")
	dispatch, err := r.resolveProviderDispatch(first, ctx.VSRSelectedDecision.Name, true)
	require.NoError(t, err)
	body := []byte(`{"model":"provider-model","messages":[{"role":"user","content":"hello"}]}`)
	projected, _, err := r.projectProviderRequest(body, dispatch, ctx)
	require.NoError(t, err)
	require.Equal(t, metricsBefore, nativePreviewObservationCounters(t))
	adapted, err := r.adaptProviderRequest(body, dispatch, ctx)
	require.NoError(t, err)
	require.Equal(t, projected, adapted, "dispatch must retain the exact projected provider payload")
	require.NotEqual(t, metricsBefore, nativePreviewObservationCounters(t), "actual dispatch still records reasoning observations")
}

func TestNativePreviewRejectsUnprojectedDynamicRequestsBeforeRender(t *testing.T) {
	for _, tt := range []struct {
		name, plugin string
		value        map[string]any
		mutate       func(*RequestContext)
	}{
		{name: "rag", plugin: "rag", value: map[string]any{"enabled": true}},
		{name: "memory", plugin: "memory", value: map[string]any{"enabled": true}},
		{name: "tool-selection", plugin: "tool_selection", value: map[string]any{"enabled": true}, mutate: func(ctx *RequestContext) { ctx.SemanticRequest.ToolChoice.Mode = llmprotocol.ToolChoiceAuto }},
		{name: "filtered-tools", plugin: "tools", value: map[string]any{"enabled": true, "mode": "filtered"}},
		{name: "history", mutate: func(ctx *RequestContext) {
			ctx.SemanticRequest.Messages = append([]llmprotocol.Message{{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "previous"}}}}, ctx.SemanticRequest.Messages...)
		}},
		{name: "lora", mutate: func(ctx *RequestContext) { ctx.VSRSelectedDecision.ModelRefs[0].LoRAName = "adapter" }},
		{name: "looper", mutate: func(ctx *RequestContext) {
			ctx.VSRSelectedDecision.Algorithm = &config.AlgorithmConfig{Type: config.DecisionAlgorithmConfidence}
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			calls := 0
			r, ctx := automaticFixture(t, "hello", renderMock(t, &calls, 0, 0))
			if tt.plugin != "" {
				payload, err := config.NewStructuredPayload(tt.value)
				require.NoError(t, err)
				ctx.VSRSelectedDecision.Plugins = append(ctx.VSRSelectedDecision.Plugins, config.DecisionPlugin{Type: tt.plugin, Configuration: payload})
			}
			if tt.mutate != nil {
				tt.mutate(ctx)
			}
			result := r.SelectModelForEval(nativePreviewInput(t, ctx))
			require.Equal(t, services.EvalSelectionExecutionRequired, result.Status, "%+v", result)
			require.Empty(t, result.SelectedModel)
			require.Equal(t, 0, calls)
			require.Nil(t, r.ContextCompression)
		})
	}
}

func TestNativePreviewLearningKeepsRenderedPoolAndReadOnlyState(t *testing.T) {
	selection.InitializeMetrics()
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)
	calls := 0
	r, ctx := automaticFixture(t, strings.Repeat("a", 40000), renderMock(t, &calls, 0, 0))
	first := addWidePreviewCandidate(r, ctx)
	r.Config.RouterLearning = routerLearningProtectionOnlyTestConfig(config.RouterLearningScopeConversation).RouterLearning
	r.Config.RouterLearning.Adaptation.Enabled = extprocBoolPtr(true)
	r.Config.RouterLearning.Adaptation.CandidateSet = config.RouterLearningCandidateSetGlobal
	r.Config.Decisions = []config.Decision{*ctx.VSRSelectedDecision}
	for name, p := range r.Config.ModelConfig {
		r.Config.ModelConfig[name] = addTestQuality(p, .9)
	}
	r.Config.DocumentHash = strings.Repeat("a", 64)
	live := routerLearningRequestContext("native-session", "native-conversation")
	live.VSRSelectedDecision = ctx.VSRSelectedDecision
	identity, ok := r.protectionIdentity(live, r.Config.RouterLearning.Protection)
	require.True(t, ok)
	at := time.Now()
	sessiontelemetry.RecordSessionDecision(sessiontelemetry.SessionDecisionParams{SessionID: identity.memoryKey, SelectedModel: first, SelectedCandidate: &ctx.VSRSelectedDecision.ModelRefs[0], DecisionName: ctx.VSRSelectedDecision.Name, Timestamp: at})
	sessiontelemetry.RecordTurnOutcome(identity.memoryKey, sessiontelemetry.TurnOutcome{TurnIndex: 1, Model: first, Category: sessiontelemetry.TurnProgress}, at)
	before, _ := sessiontelemetry.PeekRouterSessionSnapshot(identity.memoryKey, at)
	original := routerLearningSamplingSeedSource
	routerLearningSamplingSeedSource = func() int64 { t.Error("preview used production RNG"); return 0 }
	t.Cleanup(func() { routerLearningSamplingSeedSource = original })
	metricsBefore := nativePreviewObservationCounters(t)
	input := nativePreviewInput(t, ctx)
	seed := int64(17)
	input.PreviewContext = &services.PreviewContext{SessionID: "native-session", ConversationID: "native-conversation", SamplingSeed: &seed}
	result := r.SelectModelForEval(input)
	require.Equal(t, services.EvalSelectionSelected, result.Status, "%+v", result)
	require.Equal(t, "wide", result.SelectedModel, "Learning must not reintroduce the overflowed prior owner")
	require.NotNil(t, result.Provenance)
	require.True(t, result.Provenance.StateDependent)
	after, _ := sessiontelemetry.PeekRouterSessionSnapshot(identity.memoryKey, at)
	require.Equal(t, before, after)
	require.Nil(t, r.routerLearningRuntime)
	require.Nil(t, r.ContextCompression)
	require.Equal(t, metricsBefore, nativePreviewObservationCounters(t))
	require.Equal(t, 2, calls)

	// Existing learned data must remain intact as well as an uninitialized
	// runtime: Preview copies it, and never records a new experience or lookup.
	runtime := r.routerLearningRuntimeState()
	runtime.recordModelExperience(ctx.VSRSelectedDecision.Name, 0, "wide", routerLearningOutcomeGoodFit, 10)
	experienceBefore := runtime.experienceSnapshot(ctx.VSRSelectedDecision.Name, 0, "wide")
	lookup := lookuptable.NewMemoryStorage()
	require.NoError(t, lookup.Set(lookuptable.HandoffPenaltyKey(first, "wide"), lookuptable.Entry{Value: 0.2, Source: lookuptable.SourceManual, UpdatedAt: at}))
	r.LookupTable = lookup
	lookupBefore := lookup.All()
	result = r.SelectModelForEval(input)
	require.Equal(t, services.EvalSelectionSelected, result.Status, "%+v", result)
	require.Equal(t, "wide", result.SelectedModel)
	require.Equal(t, experienceBefore, runtime.experienceSnapshot(ctx.VSRSelectedDecision.Name, 0, "wide"))
	require.Equal(t, lookupBefore, lookup.All())
	after, _ = sessiontelemetry.PeekRouterSessionSnapshot(identity.memoryKey, at)
	require.Equal(t, before, after)
	require.Equal(t, metricsBefore, nativePreviewObservationCounters(t))
	require.Equal(t, 4, calls)
}

func nativePreviewObservationCounters(t *testing.T) string {
	t.Helper()
	families, err := prometheus.DefaultGatherer.Gather()
	require.NoError(t, err)
	var result strings.Builder
	for _, family := range families {
		if strings.HasPrefix(family.GetName(), "llm_model_selection") || strings.HasPrefix(family.GetName(), "llm_reasoning") {
			result.WriteString(family.String())
		}
	}
	return result.String()
}

func TestNativePreviewDynamicGuardUsesEffectiveToolPolicy(t *testing.T) {
	for _, mode := range []string{"blocked-choice", "stripped-history"} {
		t.Run(mode, func(t *testing.T) {
			calls := 0
			r, ctx := automaticFixture(t, "hello", renderMock(t, &calls, 0, 0))
			ctx.SemanticRequest.ToolChoice.Mode = llmprotocol.ToolChoiceAuto
			ctx.SemanticRequest.Tools = []llmprotocol.Tool{{Name: "lookup", InputSchema: json.RawMessage(`{"type":"object"}`)}}
			plugin := map[string]any{"enabled": true}
			kind := "tool_selection"
			if mode == "blocked-choice" {
				setAutomaticBlockedParams(t, ctx, []string{"tool_choice"})
			} else {
				kind = "tools"
				plugin = map[string]any{"enabled": true, "mode": "none", "strip_tool_history": true}
				ctx.SemanticRequest.Messages = append([]llmprotocol.Message{
					{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: "call", Name: "lookup", Arguments: "{}"}}}},
					{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{CallID: "call", Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "saved result"}}}}}},
				}, ctx.SemanticRequest.Messages...)
			}
			payload, err := config.NewStructuredPayload(plugin)
			require.NoError(t, err)
			ctx.VSRSelectedDecision.Plugins = append(ctx.VSRSelectedDecision.Plugins, config.DecisionPlugin{Type: kind, Configuration: payload})
			before, err := json.Marshal(ctx.SemanticRequest)
			require.NoError(t, err)
			result := r.SelectModelForEval(nativePreviewInput(t, ctx))
			require.Equal(t, services.EvalSelectionSelected, result.Status, "%+v", result)
			require.Equal(t, 1, calls)
			after, err := json.Marshal(ctx.SemanticRequest)
			require.NoError(t, err)
			require.Equal(t, before, after)
		})
	}
}

func TestNativePreviewKeepsSameNamedRecipePolicyIsolated(t *testing.T) {
	calls := 0
	var effort string
	r, ctx := automaticFixture(t, "hello", func(w http.ResponseWriter, req *http.Request) {
		calls++
		require.Equal(t, "/v1/chat/completions/render", req.URL.Path)
		var body map[string]any
		require.NoError(t, json.NewDecoder(req.Body).Decode(&body))
		effort, _ = body["reasoning_effort"].(string)
		require.NoError(t, json.NewEncoder(w).Encode(map[string]any{"model": body["model"], "token_ids": []int{1, 2}, "sampling_params": map[string]any{"max_tokens": 32766}}))
	})
	model := ctx.VSRSelectedDecision.ModelRefs[0].Model
	p := r.Config.ModelConfig[model]
	p.ReasoningFamily = "isolation"
	r.Config.ModelConfig[model] = p
	r.Config.ReasoningFamilies = map[string]config.ReasoningFamilyConfig{"isolation": {Type: config.ReasoningFamilyTypeTopLevelReasoningEffort, Parameter: "reasoning_effort"}}
	selected := *ctx.VSRSelectedDecision
	selected.ModelRefs = []config.ModelRef{{Model: model, ModelReasoningControl: config.ModelReasoningControl{UseReasoning: extprocBoolPtr(true), ReasoningEffort: "high"}}}
	foreign := selected
	foreign.ModelRefs = []config.ModelRef{{Model: model, ModelReasoningControl: config.ModelReasoningControl{UseReasoning: extprocBoolPtr(true), ReasoningEffort: "low"}}}
	r.Config.Decisions = []config.Decision{foreign}
	r.Config.Recipes = []config.RoutingRecipe{
		{Name: "foreign", Profile: config.RoutingProfile{Decisions: []config.Decision{foreign}}},
		{Name: "selected", Profile: config.RoutingProfile{CandidateRequirements: r.Config.CandidateRequirements, Decisions: []config.Decision{selected}}},
	}
	r.Config.CandidateRequirements = nil
	input := nativePreviewInput(t, ctx)
	input.Recipe = "selected"
	input.Decision = &r.Config.Recipes[1].Profile.Decisions[0]
	result := r.SelectModelForEval(input)
	require.Equal(t, services.EvalSelectionSelected, result.Status, "%+v", result)
	require.Equal(t, "high", effort)
	p.Capabilities = nil
	r.Config.ModelConfig[model] = p
	result = r.SelectModelForEval(input)
	require.Equal(t, services.EvalSelectionUnavailable, result.Status, "selected recipe must enforce its own declared capabilities")
	require.Equal(t, 1, calls, "foreign relaxed requirements must not admit a selected-recipe render")
	input.Recipe = "foreign"
	input.Decision = &r.Config.Recipes[0].Profile.Decisions[0]
	result = r.SelectModelForEval(input)
	require.Equal(t, services.EvalSelectionSelected, result.Status, "%+v", result)
	require.Equal(t, "low", effort)
	require.Equal(t, 2, calls)
	require.Equal(t, "high", r.Config.Recipes[1].Profile.Decisions[0].ModelRefs[0].ReasoningEffort)
}
