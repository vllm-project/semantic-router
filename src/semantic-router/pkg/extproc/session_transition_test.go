package extproc

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

func TestPopulateSessionTransitionFields_ResponseAPI(t *testing.T) {
	ctx := &RequestContext{
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			ConversationID:       "conv-123",
			PreviousResponseID:   "resp-abc",
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
	assert.Equal(t, "resp-abc", ctx.PreviousResponseID)
}

func TestPopulateSessionTransitionFields_ResponseAPI_HistoryTokens_UsagePresent(t *testing.T) {
	ctx := &RequestContext{
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			ConversationID:       "conv-tok",
			ConversationHistory: []*responseapi.StoredResponse{
				{
					Model: "model-a",
					Usage: &responseapi.Usage{InputTokens: 100, OutputTokens: 50},
				},
				{
					Model: "model-a",
					Usage: &responseapi.Usage{InputTokens: 200, OutputTokens: 80},
				},
			},
		},
	}

	populateSessionTransitionFields(ctx)

	assert.Equal(t, 430, ctx.HistoryTokenCount)
}

func TestPopulateSessionTransitionFields_ResponseAPI_HistoryTokens_FallbackIncludesInputAndOutput(t *testing.T) {
	ctx := &RequestContext{
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			ConversationID:       "conv-fallback",
			ConversationHistory: []*responseapi.StoredResponse{
				{
					Model: "model-a",
					Input: []responseapi.InputItem{{
						Type:    responseapi.ItemTypeMessage,
						Role:    responseapi.RoleUser,
						Content: json.RawMessage(`"abcdefgh"`),
					}},
					OutputText: "abcdefgh",
				},
			},
		},
	}

	populateSessionTransitionFields(ctx)

	assert.Equal(t, 4, ctx.HistoryTokenCount)
}

func TestPopulateSessionTransitionFields_ResponseAPI_NilResponseInHistory(t *testing.T) {
	ctx := &RequestContext{
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			ConversationID:       "conv-nil",
			ConversationHistory: []*responseapi.StoredResponse{
				nil,
				{Model: "model-a", Usage: &responseapi.Usage{InputTokens: 50, OutputTokens: 20}},
			},
		},
	}

	require.NotPanics(t, func() { populateSessionTransitionFields(ctx) })
	assert.Equal(t, 70, ctx.HistoryTokenCount)
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

func TestPopulateSessionTransitionFields_XSessionIDHeaderOverridesDerivation(t *testing.T) {
	ctx := &RequestContext{
		Headers: map[string]string{
			headers.AuthzUserID: "user-123",
			headers.XSessionID:  "client-defined-session",
		},
		ChatCompletionMessages: []ChatCompletionMessage{
			{Role: "user", Content: "Hello"},
		},
	}

	populateSessionTransitionFields(ctx)

	assert.Equal(t, "client-defined-session", ctx.SessionID)
	assert.Equal(t, 0, ctx.TurnIndex)
}

func TestPopulateSessionTransitionFields_ChatCompletionsNoUserUsesStructureHash(t *testing.T) {
	ctx := &RequestContext{
		ChatCompletionMessages: []ChatCompletionMessage{
			{Role: "user", Content: "Hello without auth header"},
		},
	}

	populateSessionTransitionFields(ctx)

	require.True(t, strings.HasPrefix(ctx.SessionID, "cc-full-"))
	assert.Equal(t, 0, ctx.TurnIndex)
}

func TestPopulateSessionTransitionFields_ChatCompletionsEmptyThreadUsesRidFromRequestID(t *testing.T) {
	ctx := &RequestContext{
		RequestID: "upstream-req-abc",
		Headers: map[string]string{
			headers.RequestID: "hdr-req-xyz",
		},
	}

	populateSessionTransitionFields(ctx)

	require.True(t, strings.HasPrefix(ctx.SessionID, "rid-"))
}

func TestHistoryTokensFromChatMessages_Empty(t *testing.T) {
	assert.Equal(t, 0, historyTokensFromChatMessages(nil))
	assert.Equal(t, 0, historyTokensFromChatMessages([]ChatCompletionMessage{}))
}

func TestHistoryTokensFromChatMessages_ExcludesLastUserTurn(t *testing.T) {
	msgs := []ChatCompletionMessage{
		{Role: "system", Content: "abcd"},
		{Role: "user", Content: "abcdefgh"},
		{Role: "assistant", Content: "abcdefghij"},
		{Role: "user", Content: "current"},
	}
	assert.Equal(t, 5, historyTokensFromChatMessages(msgs))
}

func TestHistoryTokensFromChatMessages_SingleUserMessage_ZeroHistory(t *testing.T) {
	msgs := []ChatCompletionMessage{
		{Role: "user", Content: "hello world"},
	}
	assert.Equal(t, 0, historyTokensFromChatMessages(msgs))
}

func TestEstimateTokenLength(t *testing.T) {
	assert.Equal(t, 0, estimateTokenLength(""))
	assert.Equal(t, 1, estimateTokenLength("abcd"))
	assert.Equal(t, 3, estimateTokenLength("abcdefghijkl"))
}

func TestEstimateStoredResponseTokens_FallsBackToOutputItems(t *testing.T) {
	stored := &responseapi.StoredResponse{
		Input: []responseapi.InputItem{{
			Type:    responseapi.ItemTypeMessage,
			Role:    responseapi.RoleUser,
			Content: json.RawMessage(`"abcdefgh"`),
		}},
		Output: []responseapi.OutputItem{{
			Type: responseapi.ItemTypeMessage,
			Role: responseapi.RoleAssistant,
			Content: []responseapi.ContentPart{{
				Type: responseapi.ContentTypeOutputText,
				Text: "abcdefgh",
			}},
		}},
	}

	assert.Equal(t, 4, estimateStoredResponseTokens(stored))
}
