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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

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
		return "", "", nil, fmt.Errorf(
			"userID is required for memory extraction but the authenticated tenant has no user identity",
		)
	}

	sessionID = ctx.SessionID
	if sessionID == "" {
		sessionID = deriveSessionIDFromSemanticMessages(ctx.SemanticRequest.Messages, userID)
	}
	if state := ctx.ResponseObjectState; state != nil && !state.ProviderContextApplied && state.ConversationHistory != nil {
		history = convertStoredResponsesToMessages(state.ConversationHistory)
	}
	history = append(history, cloneSemanticMessages(ctx.SemanticRequest.Messages)...)

	return sessionID, userID, history, nil
}

// historyBeforeCurrentTurn is the conversation before its last user message,
// the current turn, which the extractor adds to a session window itself.
func historyBeforeCurrentTurn(history []llmprotocol.Message) []llmprotocol.Message {
	for index := len(history) - 1; index >= 0; index-- {
		if history[index].Role == llmprotocol.RoleUser {
			return history[:index]
		}
	}
	return history
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

// conversationStartRunes bounds how much of the instructions and of the first
// user message a conversation-start fingerprint reads, so that text a client
// keeps further into a long system prompt, such as a date or a file listing,
// cannot change it.
const conversationStartRunes = 512

// deriveConversationStartID fingerprints what every request of one
// conversation repeats: the leading text of the system and developer
// instructions and of the first user message, scoped by the authenticated
// tenant when the gateway names one. Later turns only append messages, so
// they keep the fingerprint; two conversations that start the same way share
// it. Prefix "cc-start-" distinguishes IDs from
// deriveSessionIDFromSemanticMessages (cc-).
func deriveConversationStartID(request *llmprotocol.Request, tenant string) string {
	if request == nil {
		return ""
	}
	instructions := make([]string, 0, len(request.Instructions))
	for _, block := range request.Instructions {
		if text := semanticText(block.Content); text != "" {
			instructions = append(instructions, text)
		}
	}
	firstUser := ""
	for _, message := range request.Messages {
		if message.Role == llmprotocol.RoleUser {
			firstUser = semanticText(message.Content)
			break
		}
	}
	if len(instructions) == 0 && firstUser == "" {
		return ""
	}
	var encoded strings.Builder
	for _, part := range []string{
		tenant,
		leadingRunes(strings.Join(instructions, "\n"), conversationStartRunes),
		leadingRunes(firstUser, conversationStartRunes),
	} {
		fmt.Fprintf(&encoded, "%d:%s\n", len(part), part)
	}
	hash := sha256.Sum256([]byte(encoded.String()))
	return "cc-start-" + hex.EncodeToString(hash[:])[:16]
}

func leadingRunes(text string, limit int) string {
	count := 0
	for index := range text {
		if count == limit {
			return text[:index]
		}
		count++
	}
	return text
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

// convertStoredResponsesToMessages converts retained response objects into the
// same neutral conversation contract used by live requests.
func convertStoredResponsesToMessages(storedResponses []*responseapi.StoredResponse) []llmprotocol.Message {
	var messages []llmprotocol.Message
	for _, stored := range storedResponses {
		if stored == nil {
			continue
		}
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
// Supports both Response API and Chat Completions. When routing_input changed
// the request, it is the routing text, so recalled memories and client
// metadata are not stored as the user's words.
func extractCurrentUserMessage(ctx *RequestContext) string {
	if ctx.RoutingInput.Changed() {
		return ctx.UserContent
	}
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
