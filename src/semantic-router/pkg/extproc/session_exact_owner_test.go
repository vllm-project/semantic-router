package extproc

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

// The owner is M/high, but the current quality floor only admits M/low.
// Sharing a model name must not make a different candidate the locked owner.
func TestExcludedExactOwnerFailsClosedAcrossSelection(t *testing.T) {
	for _, scope := range []string{config.RouterLearningScopeSession, config.RouterLearningScopeConversation} {
		for _, boundary := range []string{"tool_loop", "opaque", "portable"} {
			for _, mode := range []string{"apply", "observe", "bypass", "disabled"} {
				t.Run(scope+"/"+boundary+"/"+mode, func(t *testing.T) {
					router, decision, ctx, input, key := excludedExactOwnerFixture(t, scope, boundary)
					if mode == "disabled" {
						router.Config.RouterLearning.Protection.Enabled = extprocBoolPtr(false)
					} else {
						decision.Adaptations.Protection = &config.DecisionLearningProtectionConfig{Mode: mode}
					}
					before, ok := sessiontelemetry.GetRouterSessionSnapshot(key, time.Now())
					require.True(t, ok)
					selected, _, err := router.selectModelFromCandidates(input, decision.Algorithm, ctx)
					if mode != "apply" || boundary == "portable" {
						require.NoError(t, err)
						require.NotNil(t, selected)
						assert.Equal(t, "low", selected.ReasoningEffort)
						if mode == "apply" {
							policy, exists := ctx.VSRLearningPolicies.Policy(routerLearningMethodProtection)
							require.True(t, exists)
							assert.NotEqual(t, "protection_unavailable", policy.Reason, "portable release is not a protection outage")
						}
						return
					}
					require.ErrorIs(t, err, selection.ErrNoEligibleCandidates)
					assert.Nil(t, selected)
					assert.Nil(t, ctx.VSRSelectedCandidate)
					after, ok := sessiontelemetry.GetRouterSessionSnapshot(key, time.Now())
					require.True(t, ok)
					assert.Equal(t, before.CurrentCandidate, after.CurrentCandidate)
					assert.Equal(t, before.TurnCount, after.TurnCount)
					assert.Equal(t, before.SwitchCount, after.SwitchCount)
					assert.Equal(t, before.LastSeen, after.LastSeen)
					response := router.respondSelectionRejected(ctx, "virtual", err)
					require.NotNil(t, response.GetImmediateResponse())
					assert.EqualValues(t, 503, response.GetImmediateResponse().GetStatus().GetCode())
					assert.Nil(t, response.GetRequestBody(), "rejection must not produce a provider-bound continuation")
					assert.Empty(t, ctx.TargetFormat, "rejected owner must not reach dispatch")
				})
			}
		}
	}
}

// Exercise the protection caller directly, without the earlier ownership guard,
// so a guard-only fix cannot hide swallowed selector policy errors.
func TestProtectionCallerDoesNotFallbackFromExcludedExactOwner(t *testing.T) {
	for _, scope := range []string{config.RouterLearningScopeSession, config.RouterLearningScopeConversation} {
		t.Run(scope, func(t *testing.T) {
			router, _, ctx, input, key := excludedExactOwnerFixture(t, scope, "tool_loop")
			input.CandidateModels = input.CandidateModels[:1]
			base := (&selection.SelectionResult{Score: 1, Method: selection.MethodMultiFactor}).WithCandidate(input.CandidateModels[0])
			learningInput := routerLearningInput{selCtx: input, baseResult: base, selectedModelRef: base.SelectedCandidate, ctx: ctx}
			preflight := router.applyProtectionPreflight(learningInput)
			require.True(t, preflight.enabled)
			result, err := router.applyProtectionSwitch(learningInput, preflight, routerLearningDecision{})
			require.ErrorIs(t, err, selection.ErrNoEligibleCandidates)
			assert.Nil(t, result.selectedModelRef, "policy denial was converted to a base-candidate fallback")
			snapshot, ok := sessiontelemetry.GetRouterSessionSnapshot(key, time.Now())
			require.True(t, ok)
			require.NotNil(t, snapshot.CurrentCandidate)
			assert.Equal(t, "high", snapshot.CurrentCandidate.ReasoningEffort)
		})
	}
}

func TestSelectionFallbackPreservesExcludedOwnerRejection(t *testing.T) {
	router, _, ctx, input, key := excludedExactOwnerFixture(t, config.RouterLearningScopeConversation, "tool_loop")
	input.CandidateModels = input.CandidateModels[:1]
	before, ok := sessiontelemetry.GetRouterSessionSnapshot(key, time.Now())
	require.True(t, ok)
	selected, err := router.recordSelectionFallback(selection.MethodStatic, selectionFallbackUnavailable,
		input, nil, &input.CandidateModels[0], nil, ctx)
	require.ErrorIs(t, err, selection.ErrNoEligibleCandidates)
	assert.Nil(t, selected)
	after, ok := sessiontelemetry.GetRouterSessionSnapshot(key, time.Now())
	require.True(t, ok)
	assert.Equal(t, before.CurrentCandidate, after.CurrentCandidate)
	assert.Equal(t, before.TurnCount, after.TurnCount)
}

func excludedExactOwnerFixture(t *testing.T, scope, boundary string) (*OpenAIRouter, *config.Decision, *RequestContext, *selection.SelectionContext, string) {
	t.Helper()
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)
	router, decision := candidateEffortTestRouter(t, config.DecisionAlgorithmMultiFactor, "low")
	decision.Algorithm.MultiFactor.Quality.MinScore = extprocFloat64Ptr(75)
	router.Config.RouterLearning = routerLearningProtectionOnlyTestConfig(scope).RouterLearning
	request := testNeutralRequest("virtual", "continue")
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.Headers = routerLearningRequestContext("exact-owner", "conversation").Headers
	ctx.SessionID = "exact-owner"
	ctx.VSRSelectedDecision = decision
	switch boundary {
	case "tool_loop":
		ctx.VSRConversationFacts = classification.ConversationFacts{LastMessageToolResult: true}
	case "opaque":
		ctx.PreviousResponseID = "opaque-unresolved"
	}
	identity, ok := router.protectionIdentity(ctx, router.Config.RouterLearning.Protection)
	require.True(t, ok)
	key := config.RoutingNamespaceKey(ctx.Routing.RecipeName(), identity.memoryKey)
	owner := decision.ModelRefs[1]
	sessiontelemetry.RecordSessionDecision(sessiontelemetry.SessionDecisionParams{
		SessionID: key, SelectedModel: owner.Model, SelectedCandidate: &owner,
		DecisionName: decision.Name, TurnIndex: 2, Timestamp: time.Now(),
	})
	input := router.buildSelectionContext(decision.ModelRefs, decision.Name, "continue", decision.Algorithm, "", nil, ctx)
	return router, decision, ctx, input, key
}
