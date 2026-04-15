package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

func TestPopulateSessionTransitionFields_ResponseAPI(t *testing.T) {
	ctx := &RequestContext{
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			ConversationID:       "conv-123",
			ConversationHistory: []*responseapi.StoredResponse{
				{Model: "deepseek-v3"},
				{Model: "deepseek-r1"},
			},
		},
	}

	populateSessionTransitionFields(ctx)

	assert.Equal(t, "conv-123", ctx.SessionID)
	assert.Equal(t, 2, ctx.TurnIndex)
	assert.Equal(t, "deepseek-r1", ctx.PreviousModel)
}

func TestPopulateSessionTransitionFields_ChatCompletionsFirstTurnIgnoresSystemMessages(t *testing.T) {
	ctx := &RequestContext{
		Headers: map[string]string{
			"x-authz-user-id": "user-123",
		},
		ChatCompletionMessages: []ChatCompletionMessage{
			{Role: "system", Content: "You are helpful"},
			{Role: "user", Content: "Hello"},
		},
	}

	populateSessionTransitionFields(ctx)

	require.NotEmpty(t, ctx.SessionID)
	assert.Equal(t, deriveSessionIDFromMessages(ctx.ChatCompletionMessages, "user-123"), ctx.SessionID)
	assert.Equal(t, 0, ctx.TurnIndex)
	assert.Empty(t, ctx.PreviousModel)
}

func TestPopulateSessionTransitionFields_ChatCompletionsCountsPriorAssistantTurns(t *testing.T) {
	ctx := &RequestContext{
		Headers: map[string]string{
			"x-authz-user-id": "user-123",
		},
		ChatCompletionMessages: []ChatCompletionMessage{
			{Role: "system", Content: "You are helpful"},
			{Role: "user", Content: "First question"},
			{Role: "assistant", Content: "First answer"},
			{Role: "user", Content: "Second question"},
		},
	}

	populateSessionTransitionFields(ctx)

	assert.Equal(t, 1, ctx.TurnIndex)
	assert.Empty(t, ctx.PreviousModel)
}
