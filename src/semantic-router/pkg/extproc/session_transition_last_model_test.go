package extproc

import (
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
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
