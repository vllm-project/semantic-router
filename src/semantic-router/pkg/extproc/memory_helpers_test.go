package extproc

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// =============================================================================
// extractAutoStore Tests
// =============================================================================

// Note: Request-level MemoryConfig was removed for OpenAI API compatibility.
// Memory configuration is now controlled server-side via plugin config.
// See: extractAutoStore_PerDecisionPlugin tests below.

func TestExtractAutoStore_ResponseAPI_NoPluginConfig(t *testing.T) {
	// Without plugin config, auto_store defaults to false
	ctx := &RequestContext{
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			OriginalRequest:      &responseapi.ResponseAPIRequest{},
		},
	}

	result := extractAutoStore(ctx)
	assert.False(t, result, "should return false when no memory plugin config")
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
// extractAutoStore - Per-Decision Plugin Config Tests
// =============================================================================

func TestExtractAutoStore_PerDecisionPlugin_Enabled(t *testing.T) {
	autoStoreTrue := true
	ctx := &RequestContext{
		VSRSelectedDecision: &config.Decision{
			Name: "customer_support",
			Plugins: []config.DecisionPlugin{
				{
					Type: "memory",
					Configuration: map[string]interface{}{
						"enabled":    true,
						"auto_store": true,
					},
				},
			},
		},
		VSRSelectedDecisionName: "customer_support",
	}
	// Verify plugin config is parsed correctly
	memCfg := ctx.VSRSelectedDecision.GetMemoryConfig()
	require.NotNil(t, memCfg)
	require.NotNil(t, memCfg.AutoStore)
	assert.Equal(t, autoStoreTrue, *memCfg.AutoStore)

	result := extractAutoStore(ctx)
	assert.True(t, result, "should return true from per-decision plugin config")
}

func TestExtractAutoStore_PerDecisionPlugin_Disabled(t *testing.T) {
	ctx := &RequestContext{
		VSRSelectedDecision: &config.Decision{
			Name: "coding_task",
			Plugins: []config.DecisionPlugin{
				{
					Type: "memory",
					Configuration: map[string]interface{}{
						"enabled":    true,
						"auto_store": false, // Explicitly disabled
					},
				},
			},
		},
		VSRSelectedDecisionName: "coding_task",
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			OriginalRequest:      &responseapi.ResponseAPIRequest{},
		},
	}

	result := extractAutoStore(ctx)
	assert.False(t, result, "per-decision plugin config should control auto_store")
}

func TestExtractAutoStore_PerDecisionPlugin_NotSet_DefaultsFalse(t *testing.T) {
	// When plugin doesn't set auto_store, it defaults to false (server controls)
	ctx := &RequestContext{
		VSRSelectedDecision: &config.Decision{
			Name: "general",
			Plugins: []config.DecisionPlugin{
				{
					Type: "memory",
					Configuration: map[string]interface{}{
						"enabled": true,
						// auto_store NOT set - defaults to false
					},
				},
			},
		},
		VSRSelectedDecisionName: "general",
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			OriginalRequest:      &responseapi.ResponseAPIRequest{},
		},
	}

	result := extractAutoStore(ctx)
	assert.False(t, result, "should return false when plugin doesn't set auto_store")
}

func TestExtractAutoStore_NoMemoryPlugin_ReturnsFalse(t *testing.T) {
	// Without memory plugin, auto_store is always false
	ctx := &RequestContext{
		VSRSelectedDecision: &config.Decision{
			Name: "no_memory_decision",
			Plugins: []config.DecisionPlugin{
				{Type: "pii", Configuration: map[string]interface{}{"enabled": true}},
				// No memory plugin
			},
		},
		VSRSelectedDecisionName: "no_memory_decision",
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			OriginalRequest:      &responseapi.ResponseAPIRequest{},
		},
	}

	result := extractAutoStore(ctx)
	assert.False(t, result, "should return false when no memory plugin configured")
}

// =============================================================================
// extractMemoryInfo Tests
// =============================================================================

func TestExtractMemoryInfo_WithConversationID(t *testing.T) {
	// ConversationID is now set during TranslateRequest, not extractMemoryInfo.
	// Tests must set ConversationID in ResponseAPIContext.
	ctx := &RequestContext{
		RequestID: "req_123",
		Headers: map[string]string{
			headers.AuthzUserID: "user_789",
		},
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			ConversationID:       "conv_456", // Set by TranslateRequest
			OriginalRequest: &responseapi.ResponseAPIRequest{
				ConversationID: "conv_456",
			},
		},
	}

	sessionID, userID, history, err := extractMemoryInfo(ctx)

	require.NoError(t, err, "should not return error when userID is provided")
	assert.Equal(t, "conv_456", sessionID, "should use ConversationID from context")
	assert.Equal(t, "user_789", userID, "should extract userID from auth header")
	assert.Empty(t, history, "should return empty history when ConversationHistory is nil")
}

func TestExtractMemoryInfo_WithoutConversationID_WithUserID(t *testing.T) {
	// When no ConversationID in request, TranslateRequest generates one.
	// extractMemoryInfo just reads it from context.
	ctx := &RequestContext{
		RequestID: "req_123",
		Headers: map[string]string{
			headers.AuthzUserID: "user_789",
		},
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			ConversationID:       "conv_generated_by_translate", // Set by TranslateRequest
			OriginalRequest:      &responseapi.ResponseAPIRequest{},
		},
	}

	sessionID, userID, history, err := extractMemoryInfo(ctx)

	require.NoError(t, err, "should not return error when userID is provided")
	assert.Equal(t, "conv_generated_by_translate", sessionID, "should use ConversationID from context")
	assert.Equal(t, "user_789", userID, "should extract userID from auth header")
	assert.Empty(t, history)
}

// =============================================================================
// extractUserID Tests
// =============================================================================
// NOTE: extractUserID tests have been moved to:
// - user_id_test.go (common tests for both dev and prod builds)
// - user_id_dev_test.go (dev-only tests for metadata fallback)
// - user_id_prod_test.go (prod-only tests verifying no metadata fallback)

// =============================================================================
// extractMemoryInfo Tests
// =============================================================================

func TestExtractMemoryInfo_AuthHeaderUserID(t *testing.T) {
	// Auth header (x-authz-user-id) provides trusted user ID
	ctx := &RequestContext{
		RequestID: "req_123",
		Headers: map[string]string{
			headers.AuthzUserID: "user_from_auth",
		},
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			ConversationID:       "conv_from_translate",
			OriginalRequest: &responseapi.ResponseAPIRequest{
				Metadata: map[string]string{
					"user_id": "user_from_metadata", // Should be ignored
				},
			},
		},
	}

	sessionID, userID, history, err := extractMemoryInfo(ctx)

	require.NoError(t, err)
	assert.Equal(t, "conv_from_translate", sessionID)
	assert.Equal(t, "user_from_auth", userID, "should use auth header over metadata")
	assert.Empty(t, history)
}

func TestExtractMemoryInfo_MissingUserID(t *testing.T) {
	// Even with ConversationID set, userID is required for memory storage
	ctx := &RequestContext{
		RequestID: "req_123",
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			ConversationID:       "conv_123", // Set by TranslateRequest
			OriginalRequest:      &responseapi.ResponseAPIRequest{
				// No MemoryContext or Metadata with user_id
			},
		},
	}

	sessionID, userID, history, err := extractMemoryInfo(ctx)

	require.Error(t, err, "should return error when userID is missing")
	assert.Contains(t, err.Error(), "userID is required", "error message should mention userID requirement")
	assert.Empty(t, sessionID, "should return empty sessionID when userID is missing")
	assert.Empty(t, userID, "should NOT use sessionID as userID - memory would be orphaned without real userID")
	assert.Empty(t, history)
}

func TestExtractMemoryInfo_ContinuationFromChain(t *testing.T) {
	// Test case: continuation of conversation (has PreviousResponseID)
	// TranslateRequest finds ConversationID from the chain and sets it in context
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
		Headers: map[string]string{
			headers.AuthzUserID: "user_789",
		},
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			ConversationID:       "conv_existing", // Set by TranslateRequest from chain
			PreviousResponseID:   "resp_2",        // Has PreviousResponseID = continuation
			OriginalRequest:      &responseapi.ResponseAPIRequest{},
			ConversationHistory:  storedResponses,
		},
	}

	sessionID, userID, history, err := extractMemoryInfo(ctx)

	require.NoError(t, err, "should not return error when userID is provided")
	assert.Equal(t, "conv_existing", sessionID, "should use ConversationID from context (found from chain)")
	assert.Equal(t, "user_789", userID, "should extract userID from auth header")
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
		Headers: map[string]string{
			headers.AuthzUserID: "user_789",
		},
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			ConversationID:       "conv_456", // Set by TranslateRequest
			OriginalRequest: &responseapi.ResponseAPIRequest{
				ConversationID: "conv_456",
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

func TestExtractMemoryInfo_NoHistoryAvailable(t *testing.T) {
	ctx := &RequestContext{
		RequestID:              "req_123",
		ResponseAPICtx:         nil, // Not a Response API request
		ChatCompletionMessages: nil, // No Chat Completions messages either
	}

	sessionID, userID, history, err := extractMemoryInfo(ctx)

	require.Error(t, err, "should return error when no history available")
	// Now supports both Response API and Chat Completions
	assert.Contains(t, err.Error(), "no conversation history available", "error should indicate no history available")
	assert.Empty(t, sessionID)
	assert.Empty(t, userID)
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
	assert.Empty(t, result)
}

func TestExtractContentFromInputItem_NilContent(t *testing.T) {
	item := responseapi.InputItem{
		Content: nil,
	}

	result := extractContentFromInputItem(item)
	assert.Empty(t, result)
}

func TestExtractContentFromInputItem_InvalidJSON(t *testing.T) {
	item := responseapi.InputItem{
		Content: json.RawMessage(`invalid json`),
	}

	result := extractContentFromInputItem(item)
	assert.Empty(t, result, "should return empty string for invalid JSON")
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
	assert.Empty(t, result)
}

func TestExtractContentFromOutputItem_NilContent(t *testing.T) {
	item := responseapi.OutputItem{
		Content: nil,
	}

	result := extractContentFromOutputItem(item)
	assert.Empty(t, result)
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
	assert.Empty(t, result)

	result = extractTextFromContentParts([]responseapi.ContentPart{})
	assert.Empty(t, result)
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
		Headers: map[string]string{
			headers.AuthzUserID: "user_alice",
		},
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			ConversationID:       "conv_hawaii_trip", // Set by TranslateRequest
			OriginalRequest: &responseapi.ResponseAPIRequest{
				ConversationID: "conv_hawaii_trip",
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

// ============================================================================
// Chat Completions Memory Extraction Tests
// ============================================================================

func TestExtractMemoryInfo_ChatCompletions_Success(t *testing.T) {
	ctx := &RequestContext{
		Headers: map[string]string{
			"x-authz-user-id": "user_alice", // Production: user_id from trusted auth header
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
		// No ChatCompletionUserID set
	}

	_, _, history, err := extractMemoryInfo(ctx)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "userID is required")
	assert.Len(t, history, 1, "should return history even on error")
}

func TestExtractMemoryInfo_ChatCompletions_AuthHeader(t *testing.T) {
	// Auth header (x-authz-user-id) takes precedence over Chat Completions "user" field
	ctx := &RequestContext{
		Headers: map[string]string{
			"x-authz-user-id": "auth_user_123", // Trusted header from authz framework
		},
		ChatCompletionMessages: []ChatCompletionMessage{
			{Role: "user", Content: "Hello"},
		},
		ChatCompletionUserID: "untrusted_user", // Should be ignored
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

	// Same messages + same user = same session ID
	assert.Equal(t, sessionID1, sessionID2)

	// Same messages + different user = different session ID
	assert.NotEqual(t, sessionID1, sessionID3)

	// All session IDs should have the prefix
	assert.True(t, strings.HasPrefix(sessionID1, "cc-"))
}
