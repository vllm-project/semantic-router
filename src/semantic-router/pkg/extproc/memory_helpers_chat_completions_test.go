package extproc

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestExtractMemoryInfo_ChatCompletions_Success(t *testing.T) {
	ctx := &RequestContext{
		Headers: map[string]string{
			"x-authz-user-id": "user_alice",
		},
		ChatCompletionMessages: []ChatCompletionMessage{
			{Role: "system", Content: "You are a helpful assistant"},
			{Role: "user", Content: "Hello, my name is Alice"},
			{Role: "assistant", Content: "Hello Alice! How can I help you today?"},
			{Role: "user", Content: "What's my name?"},
		},
	}

	sessionID, userID, history, err := extractMemoryInfo(ctx)

	require.NoError(t, err)
	assert.Equal(t, "user_alice", userID)
	assert.NotEmpty(t, sessionID, "sessionID should be derived from messages")
	assert.True(t, strings.HasPrefix(sessionID, "cc-"), "Chat Completions sessionID should have 'cc-' prefix")
	require.Len(t, history, 4)
	assert.Equal(t, "system", history[0].Role)
	assert.Equal(t, "user", history[1].Role)
	assert.Equal(t, "Hello, my name is Alice", history[1].Content)
}

func TestExtractMemoryInfo_ChatCompletions_NoUserID(t *testing.T) {
	ctx := &RequestContext{
		ChatCompletionMessages: []ChatCompletionMessage{
			{Role: "user", Content: "Hello"},
		},
	}

	_, _, history, err := extractMemoryInfo(ctx)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "userID is required")
	assert.Len(t, history, 1, "should return history even on error")
}

func TestExtractMemoryInfo_ChatCompletions_AuthHeader(t *testing.T) {
	ctx := &RequestContext{
		Headers: map[string]string{
			"x-authz-user-id": "auth_user_123",
		},
		ChatCompletionMessages: []ChatCompletionMessage{
			{Role: "user", Content: "Hello"},
		},
		ChatCompletionUserID: "untrusted_user",
	}

	_, userID, _, err := extractMemoryInfo(ctx)

	require.NoError(t, err)
	assert.Equal(t, "auth_user_123", userID, "should use auth header over user field")
}

func TestExtractCurrentUserMessage_ChatCompletions(t *testing.T) {
	ctx := &RequestContext{
		ChatCompletionMessages: []ChatCompletionMessage{
			{Role: "user", Content: "First message"},
			{Role: "assistant", Content: "Response"},
			{Role: "user", Content: "Last user message"},
		},
	}

	result := extractCurrentUserMessage(ctx)

	assert.Equal(t, "Last user message", result, "should return last user message")
}

func TestConvertChatCompletionMessages(t *testing.T) {
	messages := []ChatCompletionMessage{
		{Role: "system", Content: "System prompt"},
		{Role: "user", Content: "User input"},
		{Role: "assistant", Content: "Assistant output"},
	}

	result := convertChatCompletionMessages(messages)

	require.Len(t, result, 3)
	assert.Equal(t, "system", result[0].Role)
	assert.Equal(t, "System prompt", result[0].Content)
	assert.Equal(t, "user", result[1].Role)
	assert.Equal(t, "assistant", result[2].Role)
}

func TestDeriveSessionIDFromMessages(t *testing.T) {
	messages := []ChatCompletionMessage{
		{Role: "user", Content: "Hello, I need help with my order"},
	}

	sessionID1 := deriveSessionIDFromMessages(messages, "user_123")
	sessionID2 := deriveSessionIDFromMessages(messages, "user_123")
	sessionID3 := deriveSessionIDFromMessages(messages, "user_456")

	assert.Equal(t, sessionID1, sessionID2)
	assert.NotEqual(t, sessionID1, sessionID3)
	assert.True(t, strings.HasPrefix(sessionID1, "cc-"))
}
