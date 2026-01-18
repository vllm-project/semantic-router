package extproc

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// extractAutoStore checks if auto_store is enabled in the request.
//
// auto_store must be explicitly set in the request body under memory_config.auto_store.
// This is an opt-in feature - if not provided, memory extraction will not run.
//
// Example request with auto_store enabled:
//
//	{
//	  "model": "qwen3",
//	  "input": "What's my budget?",
//	  "memory_config": {
//	    "enabled": true,
//	    "auto_store": true  // ← Must be set to true to enable automatic memory extraction
//	  },
//	  "memory_context": {
//	    "user_id": "user_123"
//	  }
//	}
//
// Note: Only supported for Response API requests (/v1/responses).
// Chat Completions API is stateless by design and doesn't support auto_store
// because the router doesn't manage conversation history for Chat Completions.
func extractAutoStore(ctx *RequestContext) bool {
	// Check Response API request for memory_config.auto_store
	if ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.IsResponseAPIRequest {
		if ctx.ResponseAPICtx.OriginalRequest != nil {
			if ctx.ResponseAPICtx.OriginalRequest.MemoryConfig != nil {
				return ctx.ResponseAPICtx.OriginalRequest.MemoryConfig.AutoStore
			}
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
// userID is required and must be provided via one of:
//   - memory_context.user_id in Response API request (preferred)
//   - metadata["user_id"] in Response API request
//   - x-user-id header
func extractMemoryInfo(ctx *RequestContext) (sessionID string, userID string, history []memory.Message, err error) {
	// First check if this is a Response API request
	// Non-Response API requests cannot track turns without ConversationID
	if ctx.ResponseAPICtx == nil || !ctx.ResponseAPICtx.IsResponseAPIRequest {
		return "", "", nil, fmt.Errorf("ConversationID required for memory extraction - cannot track turns without it. Please use Response API (/v1/responses) with conversation_id or previous_response_id")
	}

	// Extract userID (required for memory extraction)
	if ctx.ResponseAPICtx.OriginalRequest != nil {
		// First, check MemoryContext (preferred method according to POC)
		if ctx.ResponseAPICtx.OriginalRequest.MemoryContext != nil {
			userID = ctx.ResponseAPICtx.OriginalRequest.MemoryContext.UserID
		}

		// Fallback: check metadata for user_id
		if userID == "" && ctx.ResponseAPICtx.OriginalRequest.Metadata != nil {
			if uid, ok := ctx.ResponseAPICtx.OriginalRequest.Metadata["user_id"]; ok {
				userID = uid
			}
		}
	}

	// Fallback: check headers
	if userID == "" {
		if uid, ok := ctx.Headers["x-user-id"]; ok {
			userID = uid
		}
	}

	// Require userID - without it, memory would be orphaned (unretrievable)
	// because memory retrieval filters by userID first
	// Check this early to avoid unnecessary sessionID calculation
	if userID == "" {
		// Extract history for error context (even though we'll return error)
		var history []memory.Message
		if ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.ConversationHistory != nil {
			history = convertStoredResponsesToMessages(ctx.ResponseAPICtx.ConversationHistory)
		}
		return "", "", history, fmt.Errorf("userID is required for memory extraction but not provided. Please provide memory_context.user_id in the request, or set metadata[\"user_id\"], or include x-user-id header")
	}

	// Extract sessionID (ConversationID) for turnCounts tracking
	// Memory extraction requires ConversationID to track turns correctly per conversation.
	// Without ConversationID, turnCounts would leak across conversations for the same user.
	if ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.IsResponseAPIRequest {
		// Case 1: Request provided ConversationID
		if ctx.ResponseAPICtx.OriginalRequest != nil && ctx.ResponseAPICtx.OriginalRequest.ConversationID != "" {
			sessionID = ctx.ResponseAPICtx.OriginalRequest.ConversationID
		} else if ctx.ResponseAPICtx.ConversationHistory != nil && len(ctx.ResponseAPICtx.ConversationHistory) > 0 {
			// Case 2: No ConversationID in request, but has PreviousResponseID (continuation)
			// Find ConversationID from the first response in the chain
			firstResponse := ctx.ResponseAPICtx.ConversationHistory[0]
			if firstResponse.ConversationID != "" {
				sessionID = firstResponse.ConversationID
			}
		}

		// Case 3: No ConversationID and no PreviousResponseID (new conversation)
		// Generate a new ConversationID
		if sessionID == "" {
			sessionID = responseapi.GenerateConversationID()
			// Store in context so we can use it in the response
			if ctx.ResponseAPICtx != nil {
				ctx.ResponseAPICtx.GeneratedConversationID = sessionID
			}
		}
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
