package extproc

import (
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

func TestNeutralSessionRestoresPreviousModel(t *testing.T) {
	sessiontelemetry.ResetLastModelForTesting()
	t.Cleanup(sessiontelemetry.ResetLastModelForTesting)
	messages := []llmprotocol.Message{neutralTextMessage(llmprotocol.RoleUser, "remember me")}
	sessionID := deriveSessionIDFromSemanticMessages(messages, "user-prev")
	sessiontelemetry.RecordLastModel(sessionID, "model-a")
	ctx := &RequestContext{
		Headers:         map[string]string{"x-authz-user-id": "user-prev"},
		SemanticRequest: &llmprotocol.Request{Generation: 1, Messages: messages},
	}
	populateSessionTransitionFields(ctx)
	if ctx.SessionID != sessionID || ctx.PreviousModel != "model-a" {
		t.Fatalf("session=%q previous=%q", ctx.SessionID, ctx.PreviousModel)
	}
}

func TestNeutralSessionModelContinuityAcrossTurns(t *testing.T) {
	sessiontelemetry.ResetLastModelForTesting()
	t.Cleanup(sessiontelemetry.ResetLastModelForTesting)
	turn1 := &RequestContext{
		RequestID: "req-1", RequestModel: "model-a",
		Headers: map[string]string{"x-authz-user-id": "user-7"},
		SemanticRequest: &llmprotocol.Request{Generation: 1, Messages: []llmprotocol.Message{
			neutralTextMessage(llmprotocol.RoleUser, "hello there"),
		}},
	}
	populateSessionTransitionFields(turn1)
	recordSessionTurn(turn1, responseUsageMetrics{promptTokens: 10, completionTokens: 20}, sessiontelemetry.TurnPricing{})

	turn2 := &RequestContext{
		RequestID: "req-2", RequestModel: "model-b",
		Headers: map[string]string{"x-authz-user-id": "user-7"},
		SemanticRequest: &llmprotocol.Request{Generation: 1, Messages: []llmprotocol.Message{
			neutralTextMessage(llmprotocol.RoleUser, "hello there"),
			neutralTextMessage(llmprotocol.RoleAssistant, "hi"),
			neutralTextMessage(llmprotocol.RoleUser, "follow up"),
		}},
	}
	populateSessionTransitionFields(turn2)
	if turn2.SessionID != turn1.SessionID || turn2.PreviousModel != "model-a" {
		t.Fatalf("turn2 session=%q previous=%q", turn2.SessionID, turn2.PreviousModel)
	}
}

func TestRecordSessionTurnUsesAuthoritativeCacheBreakdown(t *testing.T) {
	sessiontelemetry.ResetForTesting()
	t.Cleanup(sessiontelemetry.ResetForTesting)
	ctx := &RequestContext{RequestModel: "frontier", SessionID: "pinned"}
	recordSessionTurn(ctx, responseUsageMetrics{
		promptTokens: 1000, cachedPromptTokens: 400, cachedPromptTokensReported: true,
		completionTokens: 100,
	}, sessiontelemetry.TurnPricing{
		Currency: "USD", PromptPer1M: 10, CachedInputPer1M: 1, CompletionPer1M: 20,
	})
	snapshot, ok := sessiontelemetry.GetRouterSessionSnapshot("pinned", time.Now())
	if !ok || snapshot.CumulativePromptTokens != 1000 || snapshot.CumulativeCachedTokens != 400 ||
		snapshot.CumulativeCompletionTokens != 100 || snapshot.LastCacheAccountingSource != "backend_reported" {
		t.Fatalf("snapshot=%+v found=%v", snapshot, ok)
	}
}

// TestRecordSessionTurnRecordsResponseAPILineageOnlyTurn is the end-to-end
// regression for the conversation/session decoupling: a Response API turn
// continued via previous_response_id only (no explicit conversation_id, so
// ResponseObjectState.ConversationID is empty) must still flow through to
// session telemetry via the separate, always-populated SessionTrackingID:
// both the ctx.SessionID derivation (session_transition.go) and the
// sessiontelemetry.RecordTurn gate (telemetry.go) must not key off the now
// commonly-empty ConversationID.
func TestRecordSessionTurnRecordsResponseAPILineageOnlyTurn(t *testing.T) {
	sessiontelemetry.ResetForTesting()
	t.Cleanup(sessiontelemetry.ResetForTesting)

	ctx := &RequestContext{
		RequestModel: "frontier",
		ResponseObjectState: &ResponseObjectState{
			ConversationID:      "",
			SessionTrackingID:   "respapi:lineage:resp_root",
			PreviousResponseID:  "resp_root",
			ConversationHistory: []*responseapi.StoredResponse{{Model: "frontier"}},
		},
	}
	populateSessionTransitionFields(ctx)
	if ctx.SessionID != "respapi:lineage:resp_root" {
		t.Fatalf("ctx.SessionID = %q", ctx.SessionID)
	}
	recordSessionTurn(ctx, responseUsageMetrics{promptTokens: 50, completionTokens: 10}, sessiontelemetry.TurnPricing{})

	if _, ok := sessiontelemetry.GetRouterSessionSnapshot("respapi:lineage:resp_root", time.Now()); !ok {
		t.Fatal("lineage-only Response API turn (empty ConversationID) must still be recorded under its internal tracking id")
	}
}

func TestRecordSessionTurnDoesNotEstimateAcrossModelSwitch(t *testing.T) {
	sessiontelemetry.ResetForTesting()
	t.Cleanup(sessiontelemetry.ResetForTesting)
	ctx := &RequestContext{
		RequestModel: "frontier", SessionID: "switch", PreviousModel: "small",
		HistoryTokenCount: 800, CacheWarmthEstimate: 0.9,
	}
	recordSessionTurn(ctx, responseUsageMetrics{promptTokens: 1000, completionTokens: 100}, sessiontelemetry.TurnPricing{})
	snapshot, ok := sessiontelemetry.GetRouterSessionSnapshot("switch", time.Now())
	if !ok || snapshot.CumulativeEstimatedCachedTokens != 0 || snapshot.LastCacheAccountingSource != "switch_checkout" {
		t.Fatalf("snapshot=%+v found=%v", snapshot, ok)
	}
}
