package extproc

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// extractAutoStore checks if auto_store is enabled.
// Priority: per-decision plugin config > global config.
// Supported for both Response API and Chat Completions.
func extractAutoStore(ctx *RequestContext) bool {
	if ctx.VSRSelectedDecision != nil {
		memoryPluginConfig := ctx.VSRSelectedDecision.GetMemoryConfig()
		if memoryPluginConfig != nil && memoryPluginConfig.AutoStore != nil {
			logging.Infof("extractAutoStore: Using per-decision plugin config, AutoStore=%v (decision: %s)",
				*memoryPluginConfig.AutoStore, ctx.VSRSelectedDecisionName)
			return *memoryPluginConfig.AutoStore
		}
	}

	// Default: auto_store disabled unless explicitly enabled via plugin config
	return false
}

// extractUserID is defined in user_id_dev.go (with metadata fallback) and
// user_id_prod.go (auth header only) using build tags.

// extractMemoryInfo extracts sessionID, userID, and history from the request context.
// Supports both Response API and Chat Completions API.
//
// Returns an error if userID is not available, because memory would be orphaned
// (unretrievable) without a valid userID. Memory retrieval filters by userID first,
// so memories stored without userID cannot be retrieved later.
//
// userID extraction priority:
//  1. Auth header (x-authz-user-id) injected by external auth service (Authorino, Envoy Gateway JWT, etc.)
//  2. metadata["user_id"] in Response API request (fallback for development/testing)
//  3. Chat Completions "user" field (fallback for development/testing)
func extractMemoryInfo(ctx *RequestContext) (sessionID string, userID string, history []memory.Message, err error) {
	// Determine API type
	isResponseAPI := ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.IsResponseAPIRequest
	isChatCompletions := len(ctx.ChatCompletionMessages) > 0

	if !isResponseAPI && !isChatCompletions {
		return "", "", nil, fmt.Errorf("no conversation history available for memory extraction. " +
			"Use Response API (/v1/responses) or Chat Completions (/v1/chat/completions) with messages")
	}

	// Extract userID with priority: auth header > metadata/user field fallback
	userID = extractUserID(ctx)

	// Require userID - without it, memory would be orphaned (unretrievable)
	if userID == "" {
		// Extract history for error context
		if isResponseAPI && ctx.ResponseAPICtx.ConversationHistory != nil {
			history = convertStoredResponsesToMessages(ctx.ResponseAPICtx.ConversationHistory)
		} else if isChatCompletions {
			history = convertChatCompletionMessages(ctx.ChatCompletionMessages)
		}
		return "", "", history, fmt.Errorf("userID is required for memory extraction but not provided. " +
			"Please set metadata[\"user_id\"] (Response API), \"user\" field (Chat Completions), " +
			"or configure auth_user_id_header in memory config")
	}

	// Extract sessionID and history based on API type
	if isResponseAPI {
		sessionID = ctx.ResponseAPICtx.ConversationID
		if sessionID == "" {
			return "", "", nil, fmt.Errorf("ConversationID not set in Response API context")
		}
		if ctx.ResponseAPICtx.ConversationHistory != nil {
			history = convertStoredResponsesToMessages(ctx.ResponseAPICtx.ConversationHistory)
		}
	} else if isChatCompletions {
		// For Chat Completions, derive sessionID from conversation hash
		sessionID = deriveSessionIDFromMessages(ctx.ChatCompletionMessages, userID)
		history = convertChatCompletionMessages(ctx.ChatCompletionMessages)
	}

	return sessionID, userID, history, nil
}

// deriveSessionIDFromMessages creates a session ID from Chat Completions messages.
// Uses a hash of the first few messages + userID to group related conversations.
func deriveSessionIDFromMessages(messages []ChatCompletionMessage, userID string) string {
	// Use first message content + userID to create a stable session ID
	// This allows tracking turns within the same "conversation topic"
	var builder strings.Builder
	builder.WriteString(userID)
	builder.WriteString(":")

	// Include first user message to identify the conversation topic
	for _, msg := range messages {
		if msg.Role == "user" {
			// Truncate to first 100 chars to keep hash stable for long messages
			content := msg.Content
			if len(content) > 100 {
				content = content[:100]
			}
			builder.WriteString(content)
			break
		}
	}

	// Create SHA256 hash and take first 16 chars
	hash := sha256.Sum256([]byte(builder.String()))
	return "cc-" + hex.EncodeToString(hash[:])[:16]
}

// convertChatCompletionMessages converts ChatCompletionMessage[] to memory.Message[].
func convertChatCompletionMessages(messages []ChatCompletionMessage) []memory.Message {
	result := make([]memory.Message, 0, len(messages))
	for _, msg := range messages {
		result = append(result, memory.Message{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}
	return result
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
// Supports both Response API and Chat Completions.
func extractCurrentUserMessage(ctx *RequestContext) string {
	// Response API: extract from OriginalRequest.Input
	if ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.OriginalRequest != nil {
		input := ctx.ResponseAPICtx.OriginalRequest.Input
		if len(input) > 0 {
			// Try parsing as a simple string first
			var inputStr string
			if err := json.Unmarshal(input, &inputStr); err == nil {
				return inputStr
			}
			// Fallback: return raw JSON as string
			return string(input)
		}
	}

	// Chat Completions: extract last user message
	if len(ctx.ChatCompletionMessages) > 0 {
		// Find the last user message (current turn)
		for i := len(ctx.ChatCompletionMessages) - 1; i >= 0; i-- {
			if ctx.ChatCompletionMessages[i].Role == "user" {
				return ctx.ChatCompletionMessages[i].Content
			}
		}
	}

	return ""
}

// extractAssistantResponseText extracts the assistant's response text from the
// LLM response body using openai-go typed parsing. vLLM separates reasoning
// into the reasoning_content field, so ChatCompletion.Choices[0].Message.Content
// is already clean. StripThinkTags is applied as a safety net for backends that
// may still embed <think> blocks in the content field.
func extractAssistantResponseText(responseBody []byte) string {
	if len(responseBody) == 0 {
		return ""
	}

	var completion openai.ChatCompletion
	if err := json.Unmarshal(responseBody, &completion); err != nil {
		logging.Debugf("extractAssistantResponseText: failed to parse response: %v", err)
		return ""
	}

	if len(completion.Choices) == 0 {
		return ""
	}

	// Content is clean when vLLM separates reasoning; safety-net strip for others
	return memory.StripThinkTags(completion.Choices[0].Message.Content)
}
