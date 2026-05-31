package extproc

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

func TestPopulateSessionTransitionFields_ChatCompletionsPopulatesPreviousModelFromStore(t *testing.T) {
	sessiontelemetry.ResetLastModelForTesting()
	t.Cleanup(sessiontelemetry.ResetLastModelForTesting)

	msgs := []ChatCompletionMessage{{Role: "user", Content: "remember me"}}
	sid := deriveSessionIDFromMessages(msgs, "user-prev")
	sessiontelemetry.RecordLastModel(sid, "model-a")

	ctx := &RequestContext{
		Headers:                map[string]string{"x-authz-user-id": "user-prev"},
		ChatCompletionMessages: msgs,
	}
	populateSessionTransitionFields(ctx)

	require.Equal(t, sid, ctx.SessionID)
	assert.Equal(t, "model-a", ctx.PreviousModel)
}

func TestPopulateSessionTransitionFields_ChatCompletionsNoPriorModelLeavesEmpty(t *testing.T) {
	sessiontelemetry.ResetLastModelForTesting()
	t.Cleanup(sessiontelemetry.ResetLastModelForTesting)

	ctx := &RequestContext{
		Headers:                map[string]string{"x-authz-user-id": "user-fresh"},
		ChatCompletionMessages: []ChatCompletionMessage{{Role: "user", Content: "first contact"}},
	}
	populateSessionTransitionFields(ctx)

	require.NotEmpty(t, ctx.SessionID)
	assert.Empty(t, ctx.PreviousModel)
}

// TestChatCompletionsModelContinuityAcrossTurns exercises the full write→read
// path: turn 1 records its model at response time, and turn 2's request phase
// recovers it as PreviousModel — the exact gap #1753's first deliverable closes.
func TestChatCompletionsModelContinuityAcrossTurns(t *testing.T) {
	sessiontelemetry.ResetLastModelForTesting()
	t.Cleanup(sessiontelemetry.ResetLastModelForTesting)

	turn1 := &RequestContext{
		RequestID:              "req-1",
		RequestModel:           "model-a",
		Headers:                map[string]string{"x-authz-user-id": "user-7"},
		ChatCompletionMessages: []ChatCompletionMessage{{Role: "user", Content: "hello there"}},
	}
	populateSessionTransitionFields(turn1)
	require.Empty(t, turn1.PreviousModel, "first turn has no prior model")
	recordSessionTurn(turn1, responseUsageMetrics{promptTokens: 10, completionTokens: 20}, sessiontelemetry.TurnPricing{})

	// Same user + same first user message => same derived session.
	turn2 := &RequestContext{
		RequestID:    "req-2",
		RequestModel: "model-b",
		Headers:      map[string]string{"x-authz-user-id": "user-7"},
		ChatCompletionMessages: []ChatCompletionMessage{
			{Role: "user", Content: "hello there"},
			{Role: "assistant", Content: "hi"},
			{Role: "user", Content: "follow up"},
		},
	}
	populateSessionTransitionFields(turn2)

	require.Equal(t, turn1.SessionID, turn2.SessionID, "session must be stable across turns")
	assert.Equal(t, "model-a", turn2.PreviousModel)
}

func TestRecordSessionTurn_PinnedSessionWithoutUserRecordsRouterUsage(t *testing.T) {
	sessiontelemetry.ResetForTesting()
	sessiontelemetry.ResetLastModelForTesting()
	t.Cleanup(sessiontelemetry.ResetForTesting)
	t.Cleanup(sessiontelemetry.ResetLastModelForTesting)

	ctx := &RequestContext{
		RequestID:              "req-pinned-usage",
		RequestModel:           "frontier-model",
		SessionID:              "pinned-agent-session",
		Headers:                map[string]string{"x-session-id": "pinned-agent-session"},
		ChatCompletionMessages: []ChatCompletionMessage{{Role: "user", Content: "continue the task"}},
	}
	recordSessionTurn(
		ctx,
		responseUsageMetrics{
			promptTokens:               1000,
			cachedPromptTokens:         400,
			cachedPromptTokensReported: true,
			completionTokens:           100,
		},
		sessiontelemetry.TurnPricing{
			Currency:         "USD",
			PromptPer1M:      10,
			CachedInputPer1M: 1,
			CompletionPer1M:  20,
		},
	)

	snapshot, ok := sessiontelemetry.GetRouterSessionSnapshot("pinned-agent-session", time.Now())
	require.True(t, ok, "expected router-owned session memory for pinned session")
	assert.Equal(t, "frontier-model", snapshot.CurrentModel)
	assert.Equal(t, int64(1000), snapshot.CumulativePromptTokens)
	assert.Equal(t, int64(400), snapshot.CumulativeCachedTokens)
	assert.Equal(t, int64(0), snapshot.CumulativeEstimatedCachedTokens)
	assert.Equal(t, int64(100), snapshot.CumulativeCompletionTokens)
	assert.InDelta(t, 0.0084, snapshot.CumulativeCost, 0.0000001)
	assert.Equal(t, "backend_reported", snapshot.LastCacheAccountingSource)

	lastModel, ok := sessiontelemetry.GetLastModel("pinned-agent-session")
	require.True(t, ok, "expected last-model continuity for pinned session")
	assert.Equal(t, "frontier-model", lastModel)
}

func TestRecordSessionTurn_UserDerivedChatDoesNotDoubleCountRouterUsage(t *testing.T) {
	sessiontelemetry.ResetForTesting()
	t.Cleanup(sessiontelemetry.ResetForTesting)

	messages := []ChatCompletionMessage{{Role: "user", Content: "hello there"}}
	ctx := &RequestContext{
		RequestID:              "req-user-derived",
		RequestModel:           "small-model",
		Headers:                map[string]string{"x-authz-user-id": "user-7"},
		ChatCompletionMessages: messages,
	}
	populateSessionTransitionFields(ctx)
	recordSessionTurn(
		ctx,
		responseUsageMetrics{promptTokens: 200, cachedPromptTokens: 50, completionTokens: 30},
		sessiontelemetry.TurnPricing{},
	)

	snapshot, ok := sessiontelemetry.GetRouterSessionSnapshot(ctx.SessionID, time.Now())
	require.True(t, ok, "expected router-owned session memory for user-derived chat")
	assert.Equal(t, int64(200), snapshot.CumulativePromptTokens)
	assert.Equal(t, int64(50), snapshot.CumulativeCachedTokens)
	assert.Equal(t, int64(30), snapshot.CumulativeCompletionTokens)
}

func TestRecordSessionTurn_EstimatesRouterCacheReuseWhenBackendOmitsCachedTokens(t *testing.T) {
	sessiontelemetry.ResetForTesting()
	sessiontelemetry.ResetLastModelForTesting()
	t.Cleanup(sessiontelemetry.ResetForTesting)
	t.Cleanup(sessiontelemetry.ResetLastModelForTesting)

	ctx := &RequestContext{
		RequestID:              "req-router-estimate",
		RequestModel:           "frontier-model",
		SessionID:              "long-agent-session",
		PreviousModel:          "frontier-model",
		HistoryTokenCount:      800,
		CacheWarmthEstimate:    0.75,
		SessionIdleKnown:       true,
		SessionIdleSeconds:     10,
		ChatCompletionMessages: []ChatCompletionMessage{{Role: "user", Content: "continue the task"}},
	}
	recordSessionTurn(
		ctx,
		responseUsageMetrics{promptTokens: 1000, completionTokens: 100},
		sessiontelemetry.TurnPricing{
			Currency:         "USD",
			PromptPer1M:      10,
			CachedInputPer1M: 1,
			CompletionPer1M:  20,
		},
	)

	snapshot, ok := sessiontelemetry.GetRouterSessionSnapshot("long-agent-session", time.Now())
	require.True(t, ok, "expected router-owned session memory for pinned session")
	assert.Equal(t, int64(1000), snapshot.CumulativePromptTokens)
	assert.Equal(t, int64(0), snapshot.CumulativeCachedTokens)
	assert.Equal(t, int64(600), snapshot.CumulativeEstimatedCachedTokens)
	assert.InDelta(t, 0.0054, snapshot.CumulativeEstimatedCacheSavings, 0.0000001)
	assert.Equal(t, "router_estimated", snapshot.LastCacheAccountingSource)
}

func TestRecordSessionTurn_DoesNotEstimateCacheReuseAcrossPhysicalModelSwitch(t *testing.T) {
	sessiontelemetry.ResetForTesting()
	t.Cleanup(sessiontelemetry.ResetForTesting)

	ctx := &RequestContext{
		RequestID:           "req-router-switch",
		RequestModel:        "frontier-model",
		SessionID:           "switch-session",
		PreviousModel:       "small-model",
		HistoryTokenCount:   800,
		CacheWarmthEstimate: 0.90,
	}
	recordSessionTurn(
		ctx,
		responseUsageMetrics{promptTokens: 1000, completionTokens: 100},
		sessiontelemetry.TurnPricing{
			Currency:         "USD",
			PromptPer1M:      10,
			CachedInputPer1M: 1,
		},
	)

	snapshot, ok := sessiontelemetry.GetRouterSessionSnapshot("switch-session", time.Now())
	require.True(t, ok)
	assert.Equal(t, int64(0), snapshot.CumulativeEstimatedCachedTokens)
	assert.Equal(t, "switch_checkout", snapshot.LastCacheAccountingSource)
}
