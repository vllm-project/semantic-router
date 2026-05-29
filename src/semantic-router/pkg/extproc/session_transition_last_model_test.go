package extproc

import (
	"testing"

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
