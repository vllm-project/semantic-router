package extproc

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

// Cross component modes with protection errors, not just successful proposals.
func TestSelectionPathsObserveErrorComposition(t *testing.T) {
	original := routerLearningSamplingSeedSource
	routerLearningSamplingSeedSource = func() int64 { return 424242 }
	t.Cleanup(func() { routerLearningSamplingSeedSource = original })
	for _, scope := range []string{"session", "conversation"} {
		for _, boundary := range []string{"portable", "tool_loop", "opaque"} {
			for _, mode := range []string{"apply", "observe", "bypass", "disabled"} {
				t.Run(scope+"/"+boundary+"/"+mode, func(t *testing.T) {
					sessiontelemetry.ResetRouterSessionMemoryForTesting()
					t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)
					router := &OpenAIRouter{Config: routerLearningTestConfig(scope)}
					router.routerLearningRuntimeState().recordModelExperience("audit", 0, "frontier", routerLearningOutcomeGoodFit, 1000)
					router.routerLearningRuntimeState().recordModelExperience("audit", 0, "cheap", routerLearningOutcomeUnderpowered, 1000)
					ctx := routerLearningRequestContext("audit-session", "audit-conversation")
					ctx.VSRSelectedDecision = &config.Decision{Name: "audit"}
					if mode == "disabled" {
						router.Config.RouterLearning.Protection.Enabled = extprocBoolPtr(false)
					} else {
						ctx.VSRSelectedDecision.Adaptations.Protection = &config.DecisionLearningProtectionConfig{Mode: mode}
					}
					identity, ok := router.protectionIdentity(ctx, router.Config.RouterLearning.Protection)
					require.True(t, ok)
					sessiontelemetry.RecordSessionDecision(sessiontelemetry.SessionDecisionParams{SessionID: identity.memoryKey, SelectedModel: "excluded", DecisionName: "audit", Timestamp: time.Now()})
					if boundary == "tool_loop" {
						ctx.VSRConversationFacts = classification.ConversationFacts{LastMessageToolResult: true}
					}
					if boundary == "opaque" {
						ctx.PreviousResponseID = "opaque"
					}
					refs := []config.ModelRef{{Model: "cheap"}, {
						Model: "frontier", LoRAName: "frontier-adapter",
						ModelReasoningControl: config.ModelReasoningControl{UseReasoning: extprocBoolPtr(true), ReasoningEffort: "high"},
					}}
					input := router.buildSelectionContext(refs, "audit", "continue", nil, "", nil, ctx)
					base := (&selection.SelectionResult{Score: 1, Method: selection.MethodStatic}).WithCandidate(refs[0])
					_, result, selected, applied, err := router.applyRouterLearning(input, base, &refs[0], ctx)
					if mode == "apply" && boundary != "portable" {
						require.ErrorIs(t, err, selection.ErrNoEligibleCandidates)
						require.Nil(t, selected)
						return
					}
					require.NoError(t, err)
					require.NotNil(t, selected)
					require.Equal(t, "frontier", selected.Model, "protection must not undo adaptation while observing")
					require.Equal(t, "frontier", result.SelectedModel)
					require.Equal(t, refs[1], *selected)
					require.Equal(t, refs[1], *result.SelectedCandidate)
					require.True(t, applied)
					if mode == config.DecisionAdaptationModeObserve {
						policy, ok := ctx.VSRLearningPolicies.Policy(routerLearningMethodProtection)
						require.True(t, ok)
						require.Equal(t, routerLearningActionObserve, policy.Action)
						require.Equal(t, "observe_only", policy.Reason)
						require.Equal(t, "frontier", policy.String("proposal_model"))
						require.Equal(t, "frontier", policy.String("final_model"))
					}
				})
			}
		}
	}
}

func TestSelectionPathsRescueScoreDirection(t *testing.T) {
	for _, direction := range []selection.ScoreDirection{selection.HigherIsBetter, selection.LowerIsBetter} {
		name := "higher"
		if direction == selection.LowerIsBetter {
			name = "lower"
		}
		t.Run(name, func(t *testing.T) {
			sessiontelemetry.ResetRouterSessionMemoryForTesting()
			t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)
			router := &OpenAIRouter{Config: routerLearningProtectionOnlyTestConfig("session")}
			router.routerLearningRuntimeState().recordModelExperience("audit", 0, "cheap", routerLearningOutcomeFailed, 2)
			ctx := routerLearningRequestContext("audit-session", "audit-conversation")
			ctx.VSRSelectedDecision = &config.Decision{Name: "audit"}
			refs := []config.ModelRef{{Model: "cheap"}, {Model: "frontier"}}
			sessiontelemetry.RecordSessionDecision(sessiontelemetry.SessionDecisionParams{SessionID: "audit-session", SelectedModel: "cheap", SelectedCandidate: &refs[0], DecisionName: "audit", Timestamp: time.Now()})
			input := router.buildSelectionContext(refs, "audit", "continue", nil, "", nil, ctx)
			scores := selection.CandidateScores{{Candidate: refs[0], Score: 1}, {Candidate: refs[1], Score: 2}}
			if direction == selection.LowerIsBetter {
				scores[0].Score, scores[1].Score = 2, 1
			}
			base := (&selection.SelectionResult{Score: scores[1].Score, ScoreDirection: direction, Method: selection.MethodLatencyAware}).WithCandidate(refs[1]).WithScores(scores)
			_, _, selected, _, err := router.applyRouterLearning(input, base, &refs[1], ctx)
			require.NoError(t, err)
			policy, ok := ctx.VSRLearningPolicies.Policy(routerLearningMethodProtection)
			require.True(t, ok)
			require.Equal(t, "frontier", selected.Model, "lower latency is a positive rescue advantage; policy=%+v", policy.ToMap())
			require.Equal(t, routerLearningActionRescueSwitch, policy.Action)
		})
	}
}

func TestSelectionPathsWarmScopedPreflight(t *testing.T) {
	for _, scope := range []string{"session", "conversation"} {
		t.Run(scope, func(t *testing.T) {
			sessiontelemetry.ResetRouterSessionMemoryForTesting()
			t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)
			calls := 0
			original := routerLearningSamplingSeedSource
			routerLearningSamplingSeedSource = func() int64 { calls++; return 424242 }
			t.Cleanup(func() { routerLearningSamplingSeedSource = original })
			router := &OpenAIRouter{Config: routerLearningTestConfig(scope)}
			ctx := routerLearningRequestContext("audit-session", "audit-conversation")
			ctx.VSRSelectedDecision = &config.Decision{Name: "audit"}
			refs := []config.ModelRef{{Model: "cheap"}, {Model: "frontier"}}
			identity, ok := router.protectionIdentity(ctx, router.Config.RouterLearning.Protection)
			require.True(t, ok)
			sessiontelemetry.RecordSessionDecision(sessiontelemetry.SessionDecisionParams{SessionID: identity.memoryKey, SelectedModel: "cheap", SelectedCandidate: &refs[0], DecisionName: "audit", Timestamp: time.Now()})
			sessiontelemetry.RecordSessionUsage(sessiontelemetry.SessionUsageParams{SessionID: identity.memoryKey, Model: "cheap", PromptTokens: 1000, CachedPromptTokens: 900})
			input := router.buildSelectionContext(refs, "audit", "continue", nil, "", nil, ctx)
			protected := router.protectionSelectionContext(input, ctx, identity)
			require.True(t, sessionHasWarmPreviousModel(protected.AgenticSession), "fixture must establish a warm scoped owner")
			base := (&selection.SelectionResult{Score: 1, Method: selection.MethodStatic}).WithCandidate(refs[0])
			_, _, _, _, err := router.applyRouterLearning(input, base, &refs[0], ctx)
			require.NoError(t, err)
			require.Zero(t, calls, "production preflight sampled despite warm protection-scoped session; initial previous=%q", input.AgenticSession.PreviousModel)

			// A new conversation releases conversation-scoped warmth, but not a
			// session-scoped owner. Never suppress exploration using another CID.
			ctx.Headers["x-conversation-id"] = "fresh-conversation"
			input = router.buildSelectionContext(refs, "audit", "new topic", nil, "", nil, ctx)
			_, _, _, _, err = router.applyRouterLearning(input, base, &refs[0], ctx)
			require.NoError(t, err)
			wantCalls := 0
			if scope == config.RouterLearningScopeConversation {
				wantCalls = 1
			}
			require.Equal(t, wantCalls, calls)
		})
	}
}

func TestSelectionPathsCancellationShortcuts(t *testing.T) {
	for _, path := range []string{"single", "unavailable", "successful_selector", "selector_error"} {
		t.Run(path, func(t *testing.T) {
			sessiontelemetry.ResetRouterSessionMemoryForTesting()
			t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)
			cancelled, cancel := context.WithCancel(context.Background())
			cancel()
			ctx := &RequestContext{TraceContext: cancelled}
			refs := []config.ModelRef{{Model: "cheap"}, {Model: "frontier"}}
			router := &OpenAIRouter{ModelSelector: selection.NewRegistry()}
			if path == "single" {
				refs = refs[:1]
			}
			if path == "successful_selector" {
				router.ModelSelector.Register(selection.MethodStatic, selection.NewStaticSelector(nil))
			}
			if path == "selector_error" {
				router.ModelSelector.Register(selection.MethodStatic, selectionResultSelector{err: errors.New("unavailable")})
			}
			chosen, _, err := router.selectModelFromCandidates(&selection.SelectionContext{SessionID: "audit-cancel", CandidateModels: refs}, nil, ctx)
			snapshot, exists := sessiontelemetry.GetRouterSessionSnapshot("audit-cancel", time.Now())
			t.Logf("path=%s chosen=%+v err=%v owner_exists=%v owner=%s turns=%d", path, chosen, err, exists, snapshot.CurrentModel, snapshot.TurnCount)
			require.ErrorIs(t, err, context.Canceled, "cancelled path %s selected=%+v", path, chosen)
			require.Nil(t, chosen)
			require.False(t, exists, "cancelled request wrote ownership")
		})
	}
}

func TestSelectionPathsEvalAmbiguousCandidate(t *testing.T) {
	refs := []config.ModelRef{
		{Model: "model", ModelReasoningControl: config.ModelReasoningControl{ReasoningEffort: "low"}},
		{Model: "model", ModelReasoningControl: config.ModelReasoningControl{ReasoningEffort: "high"}},
	}
	registry := selection.NewRegistry()
	// A legacy/custom selector returning a model-only identity is not an ordinary outage.
	registry.Register(selection.MethodStatic, selectionResultSelector{result: &selection.SelectionResult{SelectedModel: "model", Method: selection.MethodStatic}})
	router := &OpenAIRouter{Config: &config.RouterConfig{}, ModelSelector: registry}
	decision := &config.Decision{Name: "audit", ModelRefs: refs}
	chosen, _, err := router.selectModelFromCandidates(&selection.SelectionContext{CandidateModels: refs}, nil, &RequestContext{})
	require.ErrorIs(t, err, selection.ErrNoEligibleCandidates)
	require.Nil(t, chosen)
	preview := router.SelectModelForEval(services.EvalModelSelectionInput{Decision: decision})
	require.Equal(t, services.EvalSelectionUnavailable, preview.Status, "Eval converted runtime identity rejection into %+v", preview)

	// Ordinary invalid output still uses the documented compatibility fallback.
	registry.Register(selection.MethodStatic, selectionResultSelector{result: &selection.SelectionResult{SelectedModel: "undeclared"}})
	chosen, _, err = router.selectModelFromCandidates(&selection.SelectionContext{CandidateModels: refs}, nil, &RequestContext{})
	require.NoError(t, err)
	require.Equal(t, refs[0], *chosen)
	preview = router.SelectModelForEval(services.EvalModelSelectionInput{Decision: decision})
	require.Equal(t, services.EvalSelectionFallback, preview.Status)
}

func TestSelectionPathsLateRejectionPreservesOwner(t *testing.T) {
	for _, stage := range []string{"prepare", "encode", "cancel", "immediate", "success"} {
		t.Run(stage, func(t *testing.T) {
			sessiontelemetry.ResetRouterSessionMemoryForTesting()
			t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)
			router, decision, _ := capabilityRankingTestRouter(t, config.APIFormatResponses, 80)
			router.Config.RouterLearning = routerLearningProtectionOnlyTestConfig("conversation").RouterLearning
			request := testNeutralRequest("virtual", "hello")
			ctx := routingTestContext(llmprotocol.OpenAIResponsesV1, request)
			ctx.Headers = routerLearningRequestContext("audit-session", "audit-conversation").Headers
			ctx.SessionID = "audit-session"
			ctx.VSRSelectedDecision = decision
			identity, ok := router.protectionIdentity(ctx, router.Config.RouterLearning.Protection)
			require.True(t, ok)
			// A previous different decision allows a portable transition from B to A.
			owner := decision.ModelRefs[2]
			sessiontelemetry.RecordSessionDecision(sessiontelemetry.SessionDecisionParams{SessionID: identity.memoryKey, SelectedModel: owner.Model, SelectedCandidate: &owner, DecisionName: "previous-decision", Timestamp: time.Now()})
			before, ok := sessiontelemetry.GetRouterSessionSnapshot(identity.memoryKey, time.Now())
			require.True(t, ok)
			input := router.buildSelectionContext(decision.ModelRefs, decision.Name, "hello", decision.Algorithm, "", nil, ctx)
			chosen, _, err := router.selectModelFromCandidates(input, decision.Algorithm, ctx)
			require.NoError(t, err)
			require.Equal(t, "A", chosen.Model)
			pending, ok := sessiontelemetry.GetRouterSessionSnapshot(identity.memoryKey, time.Now())
			require.True(t, ok)
			require.Equal(t, before.CurrentCandidate, pending.CurrentCandidate, "selection must not publish ownership")
			require.Equal(t, before.TurnCount, pending.TurnCount)
			if stage == "prepare" {
				request.ImageGeneration = &llmprotocol.ImageGenerationOptions{}
				dispatch, err := router.prepareProviderDispatch(request, chosen.Model, decision.Name, false, ctx)
				require.Error(t, err)
				require.Nil(t, dispatch)
			} else {
				dispatch, err := router.prepareProviderDispatch(request, chosen.Model, decision.Name, false, ctx)
				require.NoError(t, err)
				response := router.buildProviderDispatchResponse(dispatch, ctx)
				switch stage {
				case "encode":
					request.ImageGeneration = &llmprotocol.ImageGenerationOptions{}
				case "cancel":
					cancelled, cancel := context.WithCancel(context.Background())
					cancel()
					ctx.TraceContext = cancelled
				case "immediate":
					response = router.createErrorResponse(401, "credential unavailable")
				}
				response, err = router.finalizeProviderDispatchResponse(dispatch, response, ctx)
				switch stage {
				case "success":
					require.NoError(t, err)
					require.NotNil(t, response.GetRequestBody())
					after, found := sessiontelemetry.GetRouterSessionSnapshot(identity.memoryKey, time.Now())
					require.True(t, found)
					require.Equal(t, *chosen, *after.CurrentCandidate)
					require.Equal(t, before.TurnCount+1, after.TurnCount)
					require.Equal(t, before.SwitchCount+1, after.SwitchCount)
					return
				case "immediate":
					require.NoError(t, err)
					require.EqualValues(t, 401, response.GetImmediateResponse().GetStatus().GetCode())
				default:
					require.Error(t, err)
					require.Nil(t, response)
				}
			}
			after, ok := sessiontelemetry.GetRouterSessionSnapshot(identity.memoryKey, time.Now())
			require.True(t, ok)
			require.Equal(t, before.CurrentCandidate, after.CurrentCandidate, "rejected request replaced the live owner")
			require.Equal(t, before.TurnCount, after.TurnCount)
			require.Equal(t, before.SwitchCount, after.SwitchCount)
			require.Equal(t, before.LastSeen, after.LastSeen)
		})
	}
}

func TestSelectionPathsLateFailureCannotUndoNewerDispatch(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)
	router, decision, _ := capabilityRankingTestRouter(t, config.APIFormatResponses, 80)
	router.Config.RouterLearning = routerLearningProtectionOnlyTestConfig("conversation").RouterLearning
	contexts := []*RequestContext{
		routingTestContext(llmprotocol.OpenAIResponsesV1, testNeutralRequest("virtual", "older request")),
		routingTestContext(llmprotocol.OpenAIResponsesV1, testNeutralRequest("virtual", "newer request")),
	}
	for i, ctx := range contexts {
		ctx.Headers = routerLearningRequestContext("overlap", "conversation").Headers
		ctx.SessionID = "overlap"
		ctx.VSRSelectedDecision = decision
		refs := decision.ModelRefs
		if i == 1 {
			refs = refs[3:] // New request admits C, rather than the older A proposal.
		}
		input := router.buildSelectionContext(refs, decision.Name, "hello", decision.Algorithm, "", nil, ctx)
		_, _, err := router.selectModelFromCandidates(input, decision.Algorithm, ctx)
		require.NoError(t, err)
	}
	older, newer := contexts[0], contexts[1]
	require.Equal(t, "A", older.VSRSelectedCandidate.Model)
	require.Equal(t, "C", newer.VSRSelectedCandidate.Model)
	choice := newer.VSRSelectedCandidate
	reasoning := applyReasoningModeFromSelectedModel(choice, decision.Name, 1, newer)
	response, err := router.handleEntrypointModelRouting(newer.SemanticRequest, "virtual", decision.Name, reasoning, choice.Model, newer)
	require.NoError(t, err)
	require.NotNil(t, response.GetRequestBody())
	key := protectionSessionStateKey(newer)
	before, ok := sessiontelemetry.GetRouterSessionSnapshot(key, time.Now())
	require.True(t, ok)
	require.Equal(t, *choice, *before.CurrentCandidate)
	older.SemanticRequest.ImageGeneration = &llmprotocol.ImageGenerationOptions{}
	dispatch, err := router.prepareProviderDispatch(older.SemanticRequest, "A", decision.Name, false, older)
	require.Error(t, err)
	require.Nil(t, dispatch)
	after, ok := sessiontelemetry.GetRouterSessionSnapshot(key, time.Now())
	require.True(t, ok)
	require.Equal(t, before.CurrentCandidate, after.CurrentCandidate)
	require.Equal(t, before.TurnCount, after.TurnCount)
	require.Equal(t, before.SwitchCount, after.SwitchCount)
	require.Equal(t, before.LastSeen, after.LastSeen)
}

type cancelAfterSelection struct {
	selection.Selector
	cancel context.CancelFunc
}

func (s cancelAfterSelection) Select(ctx context.Context, input *selection.SelectionContext) (*selection.SelectionResult, error) {
	result, err := s.Selector.Select(ctx, input)
	s.cancel()
	return result, err
}

func TestSelectionPathsCancellationDuringSuccessfulSelector(t *testing.T) {
	requestCtx, cancel := context.WithCancel(context.Background())
	defer cancel()
	registry := selection.NewRegistry()
	registry.Register(selection.MethodStatic, cancelAfterSelection{Selector: selection.NewStaticSelector(nil), cancel: cancel})
	router := &OpenAIRouter{ModelSelector: registry}
	ctx := &RequestContext{TraceContext: requestCtx}
	chosen, _, err := router.selectModelFromCandidates(&selection.SelectionContext{
		SessionID: "cancel-during-selection", CandidateModels: []config.ModelRef{{Model: "A"}, {Model: "B"}},
	}, nil, ctx)
	require.ErrorIs(t, err, context.Canceled)
	require.Nil(t, chosen)
	require.Nil(t, ctx.VSRSelectedCandidate)
	require.Nil(t, ctx.pendingSessionDecision)
}

func TestSelectionPathsExpiredDeadline(t *testing.T) {
	deadline, cancel := context.WithDeadline(context.Background(), time.Time{})
	defer cancel()
	router := &OpenAIRouter{}
	ctx := &RequestContext{TraceContext: deadline}
	refs := []config.ModelRef{{Model: "A"}}
	input := &selection.SelectionContext{CandidateModels: refs, SessionID: "expired"}
	chosen, _, err := router.selectModelFromCandidates(input, nil, ctx)
	require.ErrorIs(t, err, context.DeadlineExceeded)
	require.Nil(t, chosen)
	_, _, chosen, _, err = router.applyRouterLearning(input, (&selection.SelectionResult{}).WithCandidate(refs[0]), &refs[0], ctx)
	require.ErrorIs(t, err, context.DeadlineExceeded)
	require.Nil(t, chosen)
	require.Nil(t, ctx.pendingSessionDecision)
}
