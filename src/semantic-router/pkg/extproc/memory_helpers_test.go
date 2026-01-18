package extproc

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// =============================================================================
// extractAutoStore Tests
// =============================================================================

func TestExtractAutoStore_ResponseAPI_WithAutoStore(t *testing.T) {
	ctx := &RequestContext{
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			OriginalRequest: &responseapi.ResponseAPIRequest{
				MemoryConfig: &responseapi.MemoryConfig{
					Enabled:   true,
					AutoStore: true,
				},
			},
		},
	}

	result := extractAutoStore(ctx)
	assert.True(t, result, "should return true when auto_store is enabled")
}

func TestExtractAutoStore_ResponseAPI_WithoutAutoStore(t *testing.T) {
	ctx := &RequestContext{
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			OriginalRequest: &responseapi.ResponseAPIRequest{
				MemoryConfig: &responseapi.MemoryConfig{
					Enabled:   true,
					AutoStore: false,
				},
			},
		},
	}

	result := extractAutoStore(ctx)
	assert.False(t, result, "should return false when auto_store is false")
}

func TestExtractAutoStore_ResponseAPI_NoMemoryConfig(t *testing.T) {
	ctx := &RequestContext{
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			OriginalRequest:      &responseapi.ResponseAPIRequest{
				// No MemoryConfig
			},
		},
	}

	result := extractAutoStore(ctx)
	assert.False(t, result, "should return false when MemoryConfig is nil")
}

func TestExtractAutoStore_ResponseAPI_NilOriginalRequest(t *testing.T) {
	ctx := &RequestContext{
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			OriginalRequest:      nil,
		},
	}

	result := extractAutoStore(ctx)
	assert.False(t, result, "should return false when OriginalRequest is nil")
}

func TestExtractAutoStore_NonResponseAPI(t *testing.T) {
	ctx := &RequestContext{
		ResponseAPICtx: nil, // Not a Response API request
	}

	result := extractAutoStore(ctx)
	assert.False(t, result, "should return false for non-Response API requests")
}

func TestExtractAutoStore_ResponseAPI_NotSet(t *testing.T) {
	ctx := &RequestContext{
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: false, // Not a Response API request
		},
	}

	result := extractAutoStore(ctx)
	assert.False(t, result, "should return false when IsResponseAPIRequest is false")
}

// =============================================================================
// extractMemoryInfo Tests
// =============================================================================

func TestExtractMemoryInfo_WithConversationID(t *testing.T) {
	ctx := &RequestContext{
		RequestID: "req_123",
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			OriginalRequest: &responseapi.ResponseAPIRequest{
				ConversationID: "conv_456",
				MemoryContext: &responseapi.MemoryContext{
					UserID: "user_789",
				},
			},
		},
	}

	sessionID, userID, history, err := extractMemoryInfo(ctx)

	require.NoError(t, err, "should not return error when userID is provided")
	assert.Equal(t, "conv_456", sessionID, "should use ConversationID as sessionID")
	assert.Equal(t, "user_789", userID, "should extract userID from MemoryContext")
	assert.Empty(t, history, "should return empty history when ConversationHistory is nil")
}

func TestExtractMemoryInfo_WithoutConversationID_WithUserID(t *testing.T) {
	ctx := &RequestContext{
		RequestID: "req_123",
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			OriginalRequest: &responseapi.ResponseAPIRequest{
				// No ConversationID, no PreviousResponseID = new conversation
				MemoryContext: &responseapi.MemoryContext{
					UserID: "user_789",
				},
			},
		},
	}

	sessionID, userID, history, err := extractMemoryInfo(ctx)

	require.NoError(t, err, "should not return error when userID is provided")
	assert.NotEmpty(t, sessionID, "should generate ConversationID for new conversation")
	assert.True(t, strings.HasPrefix(sessionID, "conv_"), "generated ConversationID should have conv_ prefix")
	assert.Equal(t, "user_789", userID, "should extract userID from MemoryContext")
	assert.Equal(t, sessionID, ctx.ResponseAPICtx.GeneratedConversationID, "should store generated ConversationID in context")
	assert.Empty(t, history)
}

func TestExtractMemoryInfo_UserIDFromMetadata(t *testing.T) {
	ctx := &RequestContext{
		RequestID: "req_123",
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			OriginalRequest: &responseapi.ResponseAPIRequest{
				// No ConversationID, no PreviousResponseID = new conversation
				Metadata: map[string]string{
					"user_id": "user_from_metadata",
				},
			},
		},
	}

	sessionID, userID, history, err := extractMemoryInfo(ctx)

	require.NoError(t, err, "should not return error when userID is provided")
	assert.NotEmpty(t, sessionID, "should generate ConversationID for new conversation")
	assert.True(t, strings.HasPrefix(sessionID, "conv_"), "generated ConversationID should have conv_ prefix")
	assert.Equal(t, "user_from_metadata", userID, "should extract userID from metadata")
	assert.Equal(t, sessionID, ctx.ResponseAPICtx.GeneratedConversationID, "should store generated ConversationID in context")
	assert.Empty(t, history)
}

func TestExtractMemoryInfo_UserIDFromHeaders(t *testing.T) {
	ctx := &RequestContext{
		RequestID: "req_123",
		Headers: map[string]string{
			"x-user-id": "user_from_header",
		},
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			OriginalRequest:      &responseapi.ResponseAPIRequest{
				// No MemoryContext or Metadata, no PreviousResponseID = new conversation
			},
		},
	}

	sessionID, userID, history, err := extractMemoryInfo(ctx)

	require.NoError(t, err, "should not return error when userID is provided")
	assert.NotEmpty(t, sessionID, "should generate ConversationID for new conversation")
	assert.True(t, strings.HasPrefix(sessionID, "conv_"), "generated ConversationID should have conv_ prefix")
	assert.Equal(t, "user_from_header", userID, "should extract userID from header")
	assert.Equal(t, sessionID, ctx.ResponseAPICtx.GeneratedConversationID, "should store generated ConversationID in context")
	assert.Empty(t, history)
}

func TestExtractMemoryInfo_FallbackToRequestID(t *testing.T) {
	ctx := &RequestContext{
		RequestID: "req_123",
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			OriginalRequest:      &responseapi.ResponseAPIRequest{
				// No ConversationID, MemoryContext, or Metadata
			},
		},
	}

	sessionID, userID, history, err := extractMemoryInfo(ctx)

	require.Error(t, err, "should return error when userID is missing")
	assert.Contains(t, err.Error(), "userID is required", "error message should mention userID requirement")
	assert.Empty(t, sessionID, "should return empty sessionID when userID is missing (no need to calculate it)")
	assert.Empty(t, userID, "should NOT use sessionID as userID - memory would be orphaned without real userID")
	assert.Empty(t, history)
}

func TestExtractMemoryInfo_ContinuationFromChain(t *testing.T) {
	// Test case: continuation of conversation (has PreviousResponseID, no ConversationID in request)
	// Should find ConversationID from the first response in the chain
	storedResponses := []*responseapi.StoredResponse{
		{
			ID:             "resp_1",
			ConversationID: "conv_existing",
			Input: []responseapi.InputItem{
				{
					Type:    "message",
					Role:    "user",
					Content: json.RawMessage(`"Hello"`),
				},
			},
			OutputText: "Hi there!",
		},
		{
			ID:             "resp_2",
			ConversationID: "conv_existing",
			Input: []responseapi.InputItem{
				{
					Type:    "message",
					Role:    "user",
					Content: json.RawMessage(`"What's my budget?"`),
				},
			},
			OutputText: "Your budget is $10,000",
		},
	}

	ctx := &RequestContext{
		RequestID: "req_123",
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			PreviousResponseID:   "resp_2", // Has PreviousResponseID = continuation
			OriginalRequest: &responseapi.ResponseAPIRequest{
				// No ConversationID in request - should find it from chain
				MemoryContext: &responseapi.MemoryContext{
					UserID: "user_789",
				},
			},
			ConversationHistory: storedResponses,
		},
	}

	sessionID, userID, history, err := extractMemoryInfo(ctx)

	require.NoError(t, err, "should not return error when userID is provided")
	assert.Equal(t, "conv_existing", sessionID, "should find ConversationID from first response in chain")
	assert.Equal(t, "user_789", userID, "should extract userID from MemoryContext")
	assert.Empty(t, ctx.ResponseAPICtx.GeneratedConversationID, "should not generate new ConversationID for continuation")
	assert.Len(t, history, 4, "should extract history from ConversationHistory")
}

func TestExtractMemoryInfo_WithConversationHistory(t *testing.T) {
	storedResponses := []*responseapi.StoredResponse{
		{
			ID: "resp_1",
			Input: []responseapi.InputItem{
				{
					Type:    "message",
					Role:    "user",
					Content: json.RawMessage(`"Hello"`),
				},
			},
			OutputText: "Hi there!",
		},
		{
			ID: "resp_2",
			Input: []responseapi.InputItem{
				{
					Type:    "message",
					Role:    "user",
					Content: json.RawMessage(`"What's my budget?"`),
				},
			},
			OutputText: "Your budget is $10,000",
		},
	}

	ctx := &RequestContext{
		RequestID: "req_123",
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			OriginalRequest: &responseapi.ResponseAPIRequest{
				ConversationID: "conv_456",
				MemoryContext: &responseapi.MemoryContext{
					UserID: "user_789",
				},
			},
			ConversationHistory: storedResponses,
		},
	}

	sessionID, userID, history, err := extractMemoryInfo(ctx)

	require.NoError(t, err, "should not return error when userID is provided")
	assert.Equal(t, "conv_456", sessionID)
	assert.Equal(t, "user_789", userID)
	require.Len(t, history, 4, "should convert 2 stored responses to 4 messages (2 user + 2 assistant)")

	// Check first user message
	assert.Equal(t, "user", history[0].Role)
	assert.Equal(t, "Hello", history[0].Content)

	// Check first assistant message
	assert.Equal(t, "assistant", history[1].Role)
	assert.Equal(t, "Hi there!", history[1].Content)

	// Check second user message
	assert.Equal(t, "user", history[2].Role)
	assert.Equal(t, "What's my budget?", history[2].Content)

	// Check second assistant message
	assert.Equal(t, "assistant", history[3].Role)
	assert.Equal(t, "Your budget is $10,000", history[3].Content)
}

func TestExtractMemoryInfo_NonResponseAPI(t *testing.T) {
	ctx := &RequestContext{
		RequestID:      "req_123",
		ResponseAPICtx: nil, // Not a Response API request
	}

	sessionID, userID, history, err := extractMemoryInfo(ctx)

	require.Error(t, err, "should return error for non-Response API request")
	// For non-Response API, we check ConversationID requirement first (before userID)
	// because we can't track turns without ConversationID
	assert.Contains(t, err.Error(), "ConversationID required", "error message should mention ConversationID requirement for non-Response API")
	assert.Empty(t, sessionID, "should return empty sessionID for non-Response API request")
	assert.Empty(t, userID, "should return empty userID for non-Response API request")
	assert.Empty(t, history)
}

// =============================================================================
// convertStoredResponsesToMessages Tests
// =============================================================================

func TestConvertStoredResponsesToMessages_Empty(t *testing.T) {
	result := convertStoredResponsesToMessages(nil)
	assert.Empty(t, result, "should return empty slice for nil input")

	result = convertStoredResponsesToMessages([]*responseapi.StoredResponse{})
	assert.Empty(t, result, "should return empty slice for empty input")
}

func TestConvertStoredResponsesToMessages_WithOutputText(t *testing.T) {
	stored := []*responseapi.StoredResponse{
		{
			Input: []responseapi.InputItem{
				{
					Type:    "message",
					Role:    "user",
					Content: json.RawMessage(`"Hello"`),
				},
			},
			OutputText: "Hi there!",
		},
	}

	result := convertStoredResponsesToMessages(stored)

	require.Len(t, result, 2)
	assert.Equal(t, "user", result[0].Role)
	assert.Equal(t, "Hello", result[0].Content)
	assert.Equal(t, "assistant", result[1].Role)
	assert.Equal(t, "Hi there!", result[1].Content)
}

func TestConvertStoredResponsesToMessages_WithoutOutputText_WithOutputItems(t *testing.T) {
	stored := []*responseapi.StoredResponse{
		{
			Input: []responseapi.InputItem{
				{
					Type:    "message",
					Role:    "user",
					Content: json.RawMessage(`"What's the weather?"`),
				},
			},
			Output: []responseapi.OutputItem{
				{
					Type: "message",
					Role: "assistant",
					Content: []responseapi.ContentPart{
						{
							Type: "output_text",
							Text: "It's sunny today!",
						},
					},
				},
			},
		},
	}

	result := convertStoredResponsesToMessages(stored)

	require.Len(t, result, 2)
	assert.Equal(t, "user", result[0].Role)
	assert.Equal(t, "What's the weather?", result[0].Content)
	assert.Equal(t, "assistant", result[1].Role)
	assert.Equal(t, "It's sunny today!", result[1].Content)
}

func TestConvertStoredResponsesToMessages_MultipleResponses(t *testing.T) {
	stored := []*responseapi.StoredResponse{
		{
			Input: []responseapi.InputItem{
				{
					Type:    "message",
					Role:    "user",
					Content: json.RawMessage(`"First message"`),
				},
			},
			OutputText: "First response",
		},
		{
			Input: []responseapi.InputItem{
				{
					Type:    "message",
					Role:    "user",
					Content: json.RawMessage(`"Second message"`),
				},
			},
			OutputText: "Second response",
		},
	}

	result := convertStoredResponsesToMessages(stored)

	require.Len(t, result, 4)
	assert.Equal(t, "First message", result[0].Content)
	assert.Equal(t, "First response", result[1].Content)
	assert.Equal(t, "Second message", result[2].Content)
	assert.Equal(t, "Second response", result[3].Content)
}

func TestConvertStoredResponsesToMessages_SkipsNonMessageTypes(t *testing.T) {
	stored := []*responseapi.StoredResponse{
		{
			Input: []responseapi.InputItem{
				{
					Type:    "tool_call", // Not a message
					Role:    "user",
					Content: json.RawMessage(`"tool data"`),
				},
				{
					Type:    "message",
					Role:    "user",
					Content: json.RawMessage(`"Hello"`),
				},
			},
			OutputText: "Hi!",
		},
	}

	result := convertStoredResponsesToMessages(stored)

	require.Len(t, result, 2, "should skip non-message input items")
	assert.Equal(t, "Hello", result[0].Content)
	assert.Equal(t, "Hi!", result[1].Content)
}

func TestConvertStoredResponsesToMessages_EmptyContent(t *testing.T) {
	stored := []*responseapi.StoredResponse{
		{
			Input: []responseapi.InputItem{
				{
					Type:    "message",
					Role:    "user",
					Content: json.RawMessage(`""`), // Empty content
				},
			},
			OutputText: "Response",
		},
	}

	result := convertStoredResponsesToMessages(stored)

	require.Len(t, result, 1, "should skip empty content")
	assert.Equal(t, "assistant", result[0].Role)
	assert.Equal(t, "Response", result[0].Content)
}

func TestConvertStoredResponsesToMessages_DefaultRole(t *testing.T) {
	stored := []*responseapi.StoredResponse{
		{
			Input: []responseapi.InputItem{
				{
					Type:    "message",
					Role:    "", // Empty role
					Content: json.RawMessage(`"Hello"`),
				},
			},
			Output: []responseapi.OutputItem{
				{
					Type: "message",
					Role: "", // Empty role
					Content: []responseapi.ContentPart{
						{
							Type: "output_text",
							Text: "Hi!",
						},
					},
				},
			},
		},
	}

	result := convertStoredResponsesToMessages(stored)

	require.Len(t, result, 2)
	assert.Equal(t, "user", result[0].Role, "should default to 'user' for input items")
	assert.Equal(t, "assistant", result[1].Role, "should default to 'assistant' for output items")
}

// =============================================================================
// extractContentFromInputItem Tests
// =============================================================================

func TestExtractContentFromInputItem_StringContent(t *testing.T) {
	item := responseapi.InputItem{
		Content: json.RawMessage(`"Hello world"`),
	}

	result := extractContentFromInputItem(item)
	assert.Equal(t, "Hello world", result)
}

func TestExtractContentFromInputItem_ArrayContent(t *testing.T) {
	item := responseapi.InputItem{
		Content: json.RawMessage(`[{"type": "output_text", "text": "Hello"}]`),
	}

	result := extractContentFromInputItem(item)
	assert.Equal(t, "Hello", result)
}

func TestExtractContentFromInputItem_EmptyContent(t *testing.T) {
	item := responseapi.InputItem{
		Content: json.RawMessage(`""`),
	}

	result := extractContentFromInputItem(item)
	assert.Equal(t, "", result)
}

func TestExtractContentFromInputItem_NilContent(t *testing.T) {
	item := responseapi.InputItem{
		Content: nil,
	}

	result := extractContentFromInputItem(item)
	assert.Equal(t, "", result)
}

func TestExtractContentFromInputItem_InvalidJSON(t *testing.T) {
	item := responseapi.InputItem{
		Content: json.RawMessage(`invalid json`),
	}

	result := extractContentFromInputItem(item)
	assert.Equal(t, "", result, "should return empty string for invalid JSON")
}

// =============================================================================
// extractContentFromOutputItem Tests
// =============================================================================

func TestExtractContentFromOutputItem_WithText(t *testing.T) {
	item := responseapi.OutputItem{
		Content: []responseapi.ContentPart{
			{
				Type: "output_text",
				Text: "Hello world",
			},
		},
	}

	result := extractContentFromOutputItem(item)
	assert.Equal(t, "Hello world", result)
}

func TestExtractContentFromOutputItem_MultipleParts(t *testing.T) {
	item := responseapi.OutputItem{
		Content: []responseapi.ContentPart{
			{
				Type: "output_text",
				Text: "Hello ",
			},
			{
				Type: "output_text",
				Text: "world!",
			},
		},
	}

	result := extractContentFromOutputItem(item)
	assert.Equal(t, "Hello world!", result)
}

func TestExtractContentFromOutputItem_SkipsNonTextParts(t *testing.T) {
	item := responseapi.OutputItem{
		Content: []responseapi.ContentPart{
			{
				Type: "image",
				Text: "image_url",
			},
			{
				Type: "output_text",
				Text: "Hello",
			},
		},
	}

	result := extractContentFromOutputItem(item)
	assert.Equal(t, "Hello", result, "should only extract output_text parts")
}

func TestExtractContentFromOutputItem_EmptyContent(t *testing.T) {
	item := responseapi.OutputItem{
		Content: []responseapi.ContentPart{},
	}

	result := extractContentFromOutputItem(item)
	assert.Equal(t, "", result)
}

func TestExtractContentFromOutputItem_NilContent(t *testing.T) {
	item := responseapi.OutputItem{
		Content: nil,
	}

	result := extractContentFromOutputItem(item)
	assert.Equal(t, "", result)
}

// =============================================================================
// extractTextFromContentParts Tests
// =============================================================================

func TestExtractTextFromContentParts_SingleTextPart(t *testing.T) {
	parts := []responseapi.ContentPart{
		{
			Type: "output_text",
			Text: "Hello",
		},
	}

	result := extractTextFromContentParts(parts)
	assert.Equal(t, "Hello", result)
}

func TestExtractTextFromContentParts_MultipleTextParts(t *testing.T) {
	parts := []responseapi.ContentPart{
		{
			Type: "output_text",
			Text: "Hello ",
		},
		{
			Type: "output_text",
			Text: "world",
		},
		{
			Type: "output_text",
			Text: "!",
		},
	}

	result := extractTextFromContentParts(parts)
	assert.Equal(t, "Hello world!", result)
}

func TestExtractTextFromContentParts_SkipsNonTextParts(t *testing.T) {
	parts := []responseapi.ContentPart{
		{
			Type: "image",
			Text: "image_url",
		},
		{
			Type: "output_text",
			Text: "Hello",
		},
		{
			Type: "tool_call",
			Text: "tool_data",
		},
	}

	result := extractTextFromContentParts(parts)
	assert.Equal(t, "Hello", result, "should only extract output_text parts")
}

func TestExtractTextFromContentParts_EmptyText(t *testing.T) {
	parts := []responseapi.ContentPart{
		{
			Type: "output_text",
			Text: "",
		},
		{
			Type: "output_text",
			Text: "Hello",
		},
	}

	result := extractTextFromContentParts(parts)
	assert.Equal(t, "Hello", result, "should skip empty text")
}

func TestExtractTextFromContentParts_EmptySlice(t *testing.T) {
	result := extractTextFromContentParts(nil)
	assert.Equal(t, "", result)

	result = extractTextFromContentParts([]responseapi.ContentPart{})
	assert.Equal(t, "", result)
}

// =============================================================================
// Integration Tests
// =============================================================================

func TestExtractMemoryInfo_RealisticScenario(t *testing.T) {
	// Simulate a realistic conversation with multiple turns
	storedResponses := []*responseapi.StoredResponse{
		{
			ID: "resp_1",
			Input: []responseapi.InputItem{
				{
					Type:    "message",
					Role:    "user",
					Content: json.RawMessage(`"I want to plan a trip to Hawaii"`),
				},
			},
			OutputText: "Great choice! Hawaii is beautiful. What's your budget?",
		},
		{
			ID: "resp_2",
			Input: []responseapi.InputItem{
				{
					Type:    "message",
					Role:    "user",
					Content: json.RawMessage(`"My budget is $10,000"`),
				},
			},
			OutputText: "Perfect! With $10,000 you can have an amazing trip to Hawaii.",
		},
		{
			ID: "resp_3",
			Input: []responseapi.InputItem{
				{
					Type:    "message",
					Role:    "user",
					Content: json.RawMessage(`"What did I say my budget was?"`),
				},
			},
			OutputText: "You said your budget is $10,000 for the Hawaii trip.",
		},
	}

	ctx := &RequestContext{
		RequestID: "req_123",
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			OriginalRequest: &responseapi.ResponseAPIRequest{
				ConversationID: "conv_hawaii_trip",
				MemoryContext: &responseapi.MemoryContext{
					UserID: "user_alice",
				},
			},
			ConversationHistory: storedResponses,
		},
	}

	sessionID, userID, history, err := extractMemoryInfo(ctx)

	require.NoError(t, err, "should not return error when userID is provided")
	assert.Equal(t, "conv_hawaii_trip", sessionID)
	assert.Equal(t, "user_alice", userID)
	require.Len(t, history, 6, "should have 6 messages (3 user + 3 assistant)")

	// Verify conversation flow
	assert.Equal(t, "I want to plan a trip to Hawaii", history[0].Content)
	assert.Equal(t, "Great choice! Hawaii is beautiful. What's your budget?", history[1].Content)
	assert.Equal(t, "My budget is $10,000", history[2].Content)
	assert.Equal(t, "Perfect! With $10,000 you can have an amazing trip to Hawaii.", history[3].Content)
	assert.Equal(t, "What did I say my budget was?", history[4].Content)
	assert.Equal(t, "You said your budget is $10,000 for the Hawaii trip.", history[5].Content)
}
