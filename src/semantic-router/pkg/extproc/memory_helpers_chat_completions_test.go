package extproc

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
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
	assert.NotNil(t, history[0].OfSystem, "first message should be system")
	assert.NotNil(t, history[1].OfUser, "second message should be user")
	assert.Equal(t, "Hello, my name is Alice", history[1].OfUser.Content.OfString.Value)
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
	assert.NotNil(t, result[0].OfSystem)
	assert.Equal(t, "System prompt", result[0].OfSystem.Content.OfString.Value)
	assert.NotNil(t, result[1].OfUser)
	assert.Equal(t, "User input", result[1].OfUser.Content.OfString.Value)
	assert.NotNil(t, result[2].OfAssistant)
	assert.Equal(t, "Assistant output", result[2].OfAssistant.Content.OfString.Value)
}

func TestConvertChatCompletionMessages_RoundTrip(t *testing.T) {
	messages := []ChatCompletionMessage{
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi there"},
	}

	sdkMsgs := convertChatCompletionMessages(messages)
	require.Len(t, sdkMsgs, 2)

	assert.Equal(t, "user", memory.SDKMessageRole(sdkMsgs[0]))
	assert.Equal(t, "Hello", memory.SDKMessageContent(sdkMsgs[0]))
	assert.Equal(t, "assistant", memory.SDKMessageRole(sdkMsgs[1]))
	assert.Equal(t, "Hi there", memory.SDKMessageContent(sdkMsgs[1]))
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
