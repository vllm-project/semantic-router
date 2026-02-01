package extproc

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// extractAutoStore checks if auto_store is enabled.
// Priority: per-decision plugin config > request-level memory_config.auto_store.
// Only supported for Response API (Chat Completions is stateless).
func extractAutoStore(ctx *RequestContext) bool {
	if ctx.VSRSelectedDecision != nil {
		memoryPluginConfig := ctx.VSRSelectedDecision.GetMemoryConfig()
		if memoryPluginConfig != nil && memoryPluginConfig.AutoStore != nil {
			logging.Infof("extractAutoStore: Using per-decision plugin config, AutoStore=%v (decision: %s)",
				*memoryPluginConfig.AutoStore, ctx.VSRSelectedDecisionName)
			return *memoryPluginConfig.AutoStore
		}
	}

	// Chat Completions API is stateless by design and doesn't support auto_store.
	// The router doesn't manage conversation history for Chat Completions requests,
	// so there's no history to extract memories from. Only Response API supports
	// auto_store because it maintains conversation history.
	return false
}

// extractMemoryInfo extracts sessionID, userID, and history from the request context.
//
// Returns an error if userID is not available, because memory would be orphaned
// (unretrievable) without a valid userID. Memory retrieval filters by userID first,
// so memories stored without userID cannot be retrieved later.
//
// userID is required and must be provided via:
//   - metadata["user_id"] in Response API request (OpenAI API spec-compliant)
func extractMemoryInfo(ctx *RequestContext) (sessionID string, userID string, history []memory.Message, err error) {
	// First check if this is a Response API request
	// Non-Response API requests cannot track turns without ConversationID
	if ctx.ResponseAPICtx == nil || !ctx.ResponseAPICtx.IsResponseAPIRequest {
		return "", "", nil, fmt.Errorf("ConversationID required for memory extraction - cannot track turns without it. Please use Response API (/v1/responses) with conversation_id or previous_response_id")
	}

	// Extract userID (required for memory extraction)
	// userID is provided via metadata.user_id (OpenAI API spec-compliant)
	if ctx.ResponseAPICtx.OriginalRequest != nil {
		if ctx.ResponseAPICtx.OriginalRequest.Metadata != nil {
			if uid, ok := ctx.ResponseAPICtx.OriginalRequest.Metadata["user_id"]; ok {
				userID = uid
			}
		}
	}

	// Require userID - without it, memory would be orphaned (unretrievable)
	// because memory retrieval filters by userID first
	// Check this early to avoid unnecessary sessionID calculation
	if userID == "" {
		// Extract history for error context (even though we'll return error)
		if ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.ConversationHistory != nil {
			history = convertStoredResponsesToMessages(ctx.ResponseAPICtx.ConversationHistory)
		}
		return "", "", history, fmt.Errorf("userID is required for memory extraction but not provided. Please set metadata[\"user_id\"] in the request")
	}

	// Extract sessionID (ConversationID) for turnCounts tracking.
	// ConversationID is determined early during TranslateRequest and stored in context.
	// This ensures consistent tracking across the entire request lifecycle.
	if ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.IsResponseAPIRequest {
		sessionID = ctx.ResponseAPICtx.ConversationID
	}

	// Safety check: ConversationID should always be set by TranslateRequest
	if sessionID == "" {
		return "", "", nil, fmt.Errorf("ConversationID not set in context - this should not happen")
	}

	// Extract history from ResponseAPIContext
	if ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.ConversationHistory != nil {
		history = convertStoredResponsesToMessages(ctx.ResponseAPICtx.ConversationHistory)
	}

	return sessionID, userID, history, nil
}

// convertStoredResponsesToMessages converts StoredResponse[] to Message[].
// It extracts user input and assistant output from each stored response.
func convertStoredResponsesToMessages(storedResponses []*responseapi.StoredResponse) []memory.Message {
	var messages []memory.Message

	for _, stored := range storedResponses {
		// Add input items as user messages
		for _, inputItem := range stored.Input {
			if inputItem.Type == "message" {
				// Extract content from InputItem
				content := extractContentFromInputItem(inputItem)
				if content != "" {
					role := inputItem.Role
					if role == "" {
						role = "user" // Default to user
					}
					messages = append(messages, memory.Message{
						Role:    role,
						Content: content,
					})
				}
			}
		}

		// Add output items as assistant messages
		// First, try to use OutputText if available (simpler)
		if stored.OutputText != "" {
			messages = append(messages, memory.Message{
				Role:    "assistant",
				Content: stored.OutputText,
			})
		} else {
			// Fallback: extract from Output items
			for _, outputItem := range stored.Output {
				if outputItem.Type == "message" {
					content := extractContentFromOutputItem(outputItem)
					if content != "" {
						role := outputItem.Role
						if role == "" {
							role = "assistant" // Default to assistant
						}
						messages = append(messages, memory.Message{
							Role:    role,
							Content: content,
						})
					}
				}
			}
		}
	}

	return messages
}

// extractContentFromInputItem extracts text content from an InputItem.
func extractContentFromInputItem(item responseapi.InputItem) string {
	if len(item.Content) == 0 {
		return ""
	}

	// Try parsing as string first
	var contentStr string
	if err := json.Unmarshal(item.Content, &contentStr); err == nil {
		return contentStr
	}

	// Try parsing as array of ContentPart
	var parts []responseapi.ContentPart
	if err := json.Unmarshal(item.Content, &parts); err == nil {
		return extractTextFromContentParts(parts)
	}

	return ""
}

// extractContentFromOutputItem extracts text content from an OutputItem.
func extractContentFromOutputItem(item responseapi.OutputItem) string {
	if len(item.Content) == 0 {
		return ""
	}

	return extractTextFromContentParts(item.Content)
}

// extractTextFromContentParts extracts text from ContentPart array.
func extractTextFromContentParts(parts []responseapi.ContentPart) string {
	var text strings.Builder
	for _, part := range parts {
		if part.Type == "output_text" && part.Text != "" {
			text.WriteString(part.Text)
		}
	}
	return text.String()
}

// extractCurrentUserMessage extracts the current user message from the request context.
// This is used to include the current turn in memory extraction.
func extractCurrentUserMessage(ctx *RequestContext) string {
	if ctx.ResponseAPICtx == nil || ctx.ResponseAPICtx.OriginalRequest == nil {
		return ""
	}

	// Response API: input is json.RawMessage, try to parse as string
	input := ctx.ResponseAPICtx.OriginalRequest.Input
	if len(input) == 0 {
		return ""
	}

	// Try parsing as a simple string first
	var inputStr string
	if err := json.Unmarshal(input, &inputStr); err == nil {
		return inputStr
	}

	// Fallback: return raw JSON as string (for complex input types)
	return string(input)
}

// extractAssistantResponseText extracts the assistant's response text from the LLM response body.
// Supports OpenAI Chat Completions format.
func extractAssistantResponseText(responseBody []byte) string {
	if len(responseBody) == 0 {
		return ""
	}

	// Try to parse as OpenAI Chat Completions response
	var chatResp struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
			Delta struct {
				Content string `json:"content"`
			} `json:"delta"`
		} `json:"choices"`
	}

	if err := json.Unmarshal(responseBody, &chatResp); err != nil {
		logging.Debugf("extractAssistantResponseText: failed to parse response: %v", err)
		return ""
	}

	if len(chatResp.Choices) == 0 {
		return ""
	}

	// Try message.content first, then delta.content (for streaming)
	content := chatResp.Choices[0].Message.Content
	if content == "" {
		content = chatResp.Choices[0].Delta.Content
	}

	return content
}
