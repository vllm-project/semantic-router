package extproc

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// extractAutoStore checks if auto_store is enabled from per-decision plugin config.
// Response API request overrides are resolved separately so request-level
// false values can still disable router-level fallback.
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

// extractRequestAutoStore is defined in dev/prod build-tagged files.

// extractMemoryInfo extracts sessionID, userID, and history from the request context.
// It consumes only neutral conversation state; object-store history is an
// optional prefix for Responses continuations.
//
// Returns an error if userID is not available, because memory would be orphaned
// (unretrievable) without a valid userID. Memory retrieval filters by userID first,
// so memories stored without userID cannot be retrieved later.
//
// userID is read only from the Router-authenticated TenantContext. Request
// headers, metadata, and protocol-level user fields are never identity sources.
func extractMemoryInfo(ctx *RequestContext) (sessionID string, userID string, history []llmprotocol.Message, err error) {
	if ctx == nil || ctx.SemanticRequest == nil || len(ctx.SemanticRequest.Messages) == 0 {
		return "", "", nil, fmt.Errorf("no conversation history available for memory extraction. " +
			"provide at least one conversation message")
	}

	// Authenticated memory is always scoped by authenticated identity.
	userID = extractUserID(ctx)

	// Require userID - without it, memory would be orphaned (unretrievable)
	if userID == "" {
		if state := ctx.ResponseObjectState; state != nil && state.ConversationHistory != nil {
			history = convertStoredResponsesToMessages(state.ConversationHistory)
		}
		history = append(history, cloneSemanticMessages(ctx.SemanticRequest.Messages)...)
		return "", "", history, fmt.Errorf(
			"userID is required for memory extraction but the authenticated tenant has no user identity",
		)
	}

	sessionID = ctx.SessionID
	if sessionID == "" {
		sessionID = deriveSessionIDFromSemanticMessages(ctx.SemanticRequest.Messages, userID)
	}
	if state := ctx.ResponseObjectState; state != nil && state.ConversationHistory != nil {
		history = convertStoredResponsesToMessages(state.ConversationHistory)
	}
	history = append(history, cloneSemanticMessages(ctx.SemanticRequest.Messages)...)

	return sessionID, userID, history, nil
}

// deriveSessionIDFromMessages creates a session ID from Chat Completions messages.
// Uses a hash of the first few messages + userID to group related conversations.
func deriveSessionIDFromSemanticMessages(messages []llmprotocol.Message, userID string) string {
	// Use first message content + userID to create a stable session ID
	// This allows tracking turns within the same "conversation topic"
	var builder strings.Builder
	builder.WriteString(userID)
	builder.WriteString(":")

	// Include first user message to identify the conversation topic
	for _, msg := range messages {
		if msg.Role == llmprotocol.RoleUser {
			// Truncate to first 100 chars to keep hash stable for long messages
			content := semanticText(msg.Content)
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

const sessionMessageFingerprintMaxRunes = 100

// deriveSessionIDFromMessagesStructure builds a session ID from the full message
// list (roles + truncated content). Used when no stable user ID is available.
// Prefix "cc-full-" distinguishes IDs from deriveSessionIDFromMessages (cc-).
func deriveSessionIDFromSemanticStructure(messages []llmprotocol.Message) string {
	if len(messages) == 0 {
		return ""
	}
	var b strings.Builder
	for i := range messages {
		if i > 0 {
			b.WriteByte('|')
		}
		b.WriteString(string(messages[i].Role))
		b.WriteByte(':')
		content := semanticText(messages[i].Content)
		if len(content) > sessionMessageFingerprintMaxRunes {
			content = content[:sessionMessageFingerprintMaxRunes]
		}
		b.WriteString(content)
	}
	hash := sha256.Sum256([]byte(b.String()))
	return "cc-full-" + hex.EncodeToString(hash[:])[:16]
}

// deriveSessionIDFromRequestID returns a deterministic pseudo-session from
// RequestID (or x-request-id header) when no message-based session exists.
func deriveSessionIDFromRequestID(ctx *RequestContext) string {
	if ctx == nil {
		return ""
	}
	rid := strings.TrimSpace(ctx.RequestID)
	if rid == "" {
		rid = strings.TrimSpace(headerValueCI(ctx, headers.RequestID))
	}
	if rid == "" {
		return ""
	}
	hash := sha256.Sum256([]byte("rid:" + rid))
	return "rid-" + hex.EncodeToString(hash[:])[:16]
}

func cloneSemanticMessages(messages []llmprotocol.Message) []llmprotocol.Message {
	result := make([]llmprotocol.Message, len(messages))
	for index := range messages {
		result[index] = messages[index]
		result[index].Content = append([]llmprotocol.Content(nil), messages[index].Content...)
	}
	return result
}

// convertStoredResponsesToMessages converts retained response objects into the
// same neutral conversation contract used by live requests.
func convertStoredResponsesToMessages(storedResponses []*responseapi.StoredResponse) []llmprotocol.Message {
	var messages []llmprotocol.Message
	for _, stored := range storedResponses {
		messages = appendInputMessages(messages, stored.Input)
		messages = appendOutputMessages(messages, stored.OutputText, stored.Output)
	}
	return messages
}

// appendInputMessages appends user-side messages extracted from Response API InputItems.
func appendInputMessages(messages []llmprotocol.Message, items []responseapi.InputItem) []llmprotocol.Message {
	for _, item := range items {
		if item.Type != "message" {
			continue
		}
		content := extractContentFromInputItem(item)
		if content == "" {
			continue
		}
		role := item.Role
		if role == "" {
			role = "user"
		}
		messages = append(messages, neutralMemoryMessage(role, content))
	}
	return messages
}

// appendOutputMessages appends assistant-side messages. It prefers OutputText
// when available and falls back to extracting from individual OutputItems.
func appendOutputMessages(messages []llmprotocol.Message, outputText string, items []responseapi.OutputItem) []llmprotocol.Message {
	if outputText != "" {
		return append(messages, neutralMemoryMessage("assistant", outputText))
	}
	for _, item := range items {
		if item.Type != "message" {
			continue
		}
		content := extractContentFromOutputItem(item)
		if content == "" {
			continue
		}
		role := item.Role
		if role == "" {
			role = "assistant"
		}
		messages = append(messages, neutralMemoryMessage(role, content))
	}
	return messages
}

func neutralMemoryMessage(role string, text string) llmprotocol.Message {
	semanticRole := llmprotocol.Role(role)
	switch semanticRole {
	case llmprotocol.RoleSystem, llmprotocol.RoleDeveloper, llmprotocol.RoleUser,
		llmprotocol.RoleAssistant, llmprotocol.RoleTool:
	default:
		semanticRole = llmprotocol.RoleUser
	}
	return llmprotocol.Message{
		Role:    semanticRole,
		Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}},
	}
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
	if ctx.SemanticRequest != nil {
		for index := len(ctx.SemanticRequest.Messages) - 1; index >= 0; index-- {
			message := ctx.SemanticRequest.Messages[index]
			if message.Role == llmprotocol.RoleUser {
				return semanticText(message.Content)
			}
		}
	}

	return ""
}

func semanticText(contents []llmprotocol.Content) string {
	parts := make([]string, 0, len(contents))
	for _, content := range contents {
		if content.Kind == llmprotocol.ContentText && content.Text != "" {
			parts = append(parts, content.Text)
		}
	}
	return strings.Join(parts, " ")
}

func extractSemanticAssistantResponseText(response *llmprotocol.Response) string {
	return memory.StripThinkTags(semanticAssistantContent(response))
}
