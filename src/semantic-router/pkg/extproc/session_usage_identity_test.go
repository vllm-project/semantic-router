package extproc

import (
	"math"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

func TestRecordSessionTurnUsesDispatchOwner(t *testing.T) {
	for _, scope := range []string{config.RouterLearningScopeSession, config.RouterLearningScopeConversation} {
		for _, protocol := range []string{"chat", "responses"} {
			t.Run(scope+"/"+protocol, func(t *testing.T) {
				sessiontelemetry.ResetForTesting()
				t.Cleanup(sessiontelemetry.ResetForTesting)
				const recipe config.RecipeName = "speed"
				const sid = "team/运行%2F"
				conversations := []string{"job/a"}
				if scope == config.RouterLearningScopeConversation {
					conversations = append(conversations, "job%2Fa")
				}
				registry := selection.NewRegistry()
				router := &OpenAIRouter{Config: routerLearningProtectionOnlyTestConfig(scope), ModelSelector: registry}
				pricing := sessiontelemetry.TurnPricing{
					Currency: "USD", PromptPer1M: 10, CachedInputPer1M: 1, CompletionPer1M: 20,
				}
				usage := responseUsageMetrics{
					promptTokens: 11, cachedPromptTokens: 4, cachedPromptTokensReported: true, completionTokens: 3,
				}
				owners := make([]string, 0, len(conversations))
				for i, cid := range conversations {
					ctx := usageIdentityRequest(protocol, sid, cid)
					ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: recipe})
					populateSessionTransitionFields(ctx)
					if ctx.SessionID != sid || extractUserID(ctx) != "authenticated-user" {
						t.Fatal("fixture must use authenticated user and explicit session identity")
					}
					ctx.VSRSelectedDecision = &config.Decision{Name: "choice"}
					model := []string{"cheap", "frontier"}[i]
					registry.Register(selection.MethodStatic, selectionResultSelector{result: &selection.SelectionResult{
						SelectedModel: model, Score: 1, Confidence: 1, Method: selection.MethodStatic,
						AllScores: map[string]float64{model: 1},
					}})
					selCtx := router.buildSelectionContext([]config.ModelRef{{Model: model}}, "choice", "", nil, "", nil, ctx)
					selected, _, err := router.selectModelFromCandidates(selCtx, nil, ctx)
					if err != nil || selected == nil || selected.Model != model {
						t.Fatalf("selection did not establish %q: selected=%+v err=%v", model, selected, err)
					}
					// Usage arrives only after a successful dispatch establishes ownership.
					if err := commitAgenticSessionDecision(ctx); err != nil {
						t.Fatal(err)
					}
					ctx.RequestModel = selected.Model
					owner := sessiontelemetry.RoutingSessionKey(recipe, sid)
					if scope == config.RouterLearningScopeConversation {
						owner = sessiontelemetry.RoutingSessionKey(recipe, sid, cid)
					}
					if protectionSessionStateKey(ctx) != owner {
						t.Fatalf("protection owner=%q, want canonical key %q", protectionSessionStateKey(ctx), owner)
					}
					before, ok := sessiontelemetry.GetRouterSessionSnapshot(owner, time.Now())
					if !ok || before.CurrentModel != model || before.TurnCount != 1 || before.CumulativePromptTokens != 0 {
						t.Fatalf("committed dispatch did not create an empty owner: %+v, found=%t", before, ok)
					}
					owners = append(owners, owner)
					recordSessionTurn(ctx, usage, pricing)
					for j, key := range owners {
						assertUsageIdentityTotals(t, key, 1, usage, pricing)
						snapshot, _ := sessiontelemetry.GetRouterSessionSnapshot(key, time.Now())
						if snapshot.CurrentModel != []string{"cheap", "frontier"}[j] || snapshot.TurnCount != 1 {
							t.Errorf("response changed decision ownership: %+v", snapshot)
						}
					}
					assertUsageIdentityTotals(t, sessiontelemetry.RoutingSessionKey(recipe, sid), i+1, usage, pricing)
					assertUsageProtocolIdentity(t, ctx, sid, cid)
					publicID := deriveSessionIDFromSemanticMessages(ctx.SemanticRequest.Messages, "authenticated-user")
					if ctx.ResponseObjectState != nil {
						publicID = ctx.ResponseObjectState.SessionTrackingID
					}
					if extra, found := sessiontelemetry.GetRouterSessionSnapshot(sessiontelemetry.RoutingSessionKey(recipe, publicID), time.Now()); found {
						t.Errorf("protocol telemetry created an unrelated routing owner: %+v", extra)
					}
				}
			})
		}
	}
}

func TestRecordSessionTurnWithoutDispatchIdentityDoesNotCreateOwner(t *testing.T) {
	sessiontelemetry.ResetForTesting()
	t.Cleanup(sessiontelemetry.ResetForTesting)
	ctx := usageIdentityRequest("chat", "", "")
	ctx.RequestModel = "cheap"
	// No routing identity was resolved: protocol telemetry must not invent one.
	recordSessionTurn(ctx, responseUsageMetrics{promptTokens: 11}, sessiontelemetry.TurnPricing{})
	derived := deriveSessionIDFromSemanticMessages(ctx.SemanticRequest.Messages, "authenticated-user")
	if snapshot, ok := sessiontelemetry.GetRouterSessionSnapshot(derived, time.Now()); ok {
		t.Fatalf("general telemetry invented a dispatch owner: %+v", snapshot)
	}
}

func usageIdentityRequest(protocol, sid, cid string) *RequestContext {
	ctx := routerLearningRequestContext(sid, cid)
	ctx.Headers[headers.AuthzUserID] = "authenticated-user"
	ctx.SemanticRequest = &llmprotocol.Request{Generation: 1, Messages: []llmprotocol.Message{
		neutralTextMessage(llmprotocol.RoleUser, "Summarize the status of this request."),
	}}
	if protocol == "responses" {
		ctx.ResponseObjectState = &ResponseObjectState{
			ConversationID: "conversation-object", SessionTrackingID: "respapi:lineage:response-parent",
			PreviousResponseID: "response-parent", GeneratedResponseID: "response-current",
		}
	}
	return ctx
}

func assertUsageIdentityTotals(t *testing.T, key string, turns int, usage responseUsageMetrics, pricing sessiontelemetry.TurnPricing) {
	t.Helper()
	snapshot, ok := sessiontelemetry.GetRouterSessionSnapshot(key, time.Now())
	if !ok || snapshot.CumulativePromptTokens != int64(turns*usage.promptTokens) ||
		snapshot.CumulativeCachedTokens != int64(turns*usage.cachedPromptTokens) ||
		snapshot.CumulativeCompletionTokens != int64(turns*usage.completionTokens) ||
		math.Abs(snapshot.CumulativeCost-float64(turns)*sessionTurnCost(usage, pricing)) > 1e-12 ||
		snapshot.LastCacheAccountingSource != "backend_reported" {
		t.Errorf("owner %q must receive each response exactly once: snapshot=%+v found=%t turns=%d", key, snapshot, ok, turns)
	}
}

func assertUsageProtocolIdentity(t *testing.T, ctx *RequestContext, sid, cid string) {
	t.Helper()
	if ctx.SessionID != sid || ctx.Headers[headers.XSessionID] != sid || ctx.Headers["x-conversation-id"] != cid {
		t.Error("usage accounting changed public routing headers")
	}
	if state := ctx.ResponseObjectState; state != nil {
		if state.ConversationID != "conversation-object" || state.SessionTrackingID != "respapi:lineage:response-parent" ||
			state.PreviousResponseID != "response-parent" || state.GeneratedResponseID != "response-current" ||
			ctx.PreviousResponseID != "response-parent" {
			t.Errorf("usage accounting changed Responses membership or lineage: %+v", state)
		}
	}
}
