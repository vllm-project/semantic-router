package cache

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ChatMessage represents a message in the OpenAI chat format with role and content.
// Content is json.RawMessage to support both plain strings and multimodal arrays.
type ChatMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// extractTextContent returns the text portion of a message's content field.
// Handles both plain strings and OpenAI multimodal content arrays.
func extractTextContent(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	// Try plain string first
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	// Try array of content parts (multimodal)
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text,omitempty"`
	}
	if err := json.Unmarshal(raw, &parts); err == nil {
		var result string
		for _, p := range parts {
			if p.Type == "text" && p.Text != "" {
				result += p.Text
			}
		}
		return result
	}
	return ""
}

// OpenAIRequest represents the structure of an OpenAI API request
type OpenAIRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	Stream      bool          `json:"stream,omitempty"`
	Temperature float32       `json:"temperature,omitempty"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	Tools       []interface{} `json:"tools,omitempty"`
	TopP        float32       `json:"top_p,omitempty"`
}

// BuildContextAwareCacheQuery builds a cache key that encodes conversation context.
//
// When contextWindowTurns is 0 the behaviour is identical to ExtractQueryFromOpenAIRequest:
// the cache key is simply the last user message.
//
// When contextWindowTurns > 0 the key also includes the system prompt and up to
// contextWindowTurns prior user+assistant pairs prepended to the current user message.
// This prevents false cache hits when the same question appears in different
// conversation contexts (e.g., "How do I fix it?" after unrelated prior turns).
//
// Returns:
//
//	model           – the model name from the request
//	lastUserQuery   – the raw last user message (used for logging / metrics)
//	contextQuery    – the embedding text: either lastUserQuery or the context-enriched form
func BuildContextAwareCacheQuery(requestBody []byte, contextWindowTurns int) (model, lastUserQuery, contextQuery string, err error) {
	var req OpenAIRequest
	if err = json.Unmarshal(requestBody, &req); err != nil {
		return "", "", "", fmt.Errorf("invalid request body: %w", err)
	}

	// Extract the raw last user message (unchanged path used by logging and metrics).
	for _, msg := range req.Messages {
		if msg.Role == "user" {
			if t := extractTextContent(msg.Content); t != "" {
				lastUserQuery = t
			}
		}
	}

	if contextWindowTurns <= 0 {
		// Context-aware mode disabled: behave like the original function.
		return req.Model, lastUserQuery, lastUserQuery, nil
	}

	contextQuery = buildContextWindow(req.Messages, contextWindowTurns)
	if contextQuery == "" {
		// Fallback: no messages produced a key; use the plain query.
		contextQuery = lastUserQuery
	}
	return req.Model, lastUserQuery, contextQuery, nil
}

// maxContextSegmentChars is the per-role-segment character cap applied when
// building a context-aware cache key.  It bounds the total embedding input
// size regardless of conversation length.
const maxContextSegmentChars = 200

// buildContextWindow assembles the multi-turn context string used as the
// embedding text for a context-aware cache key.
//
// Format (roles separated by newlines):
//
//	[system]: <truncated system prompt>
//	[user]:   <prior user turn 1>
//	[assistant]: <prior assistant turn 1>
//	…
//	[user]:   <current user message>
//
// contextWindowTurns controls how many prior user+assistant pairs are included.
// Each segment is truncated to maxContextSegmentChars characters.
func buildContextWindow(messages []ChatMessage, contextWindowTurns int) string {
	// Locate the last user message (the "current" query).
	lastUserIdx := -1
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			lastUserIdx = i
			break
		}
	}
	if lastUserIdx < 0 {
		return ""
	}

	currentUserText := truncateCacheSegment(extractTextContent(messages[lastUserIdx].Content))

	// Collect system text and prior non-system messages (before lastUserIdx).
	var systemText string
	var priorMsgs []ChatMessage
	for _, msg := range messages[:lastUserIdx] {
		if msg.Role == "system" {
			// Use the last system message if multiple exist.
			systemText = truncateCacheSegment(extractTextContent(msg.Content))
		} else {
			priorMsgs = append(priorMsgs, msg)
		}
	}

	// Keep only the most recent contextWindowTurns pairs (2 messages per turn).
	maxPrior := contextWindowTurns * 2
	if len(priorMsgs) > maxPrior {
		priorMsgs = priorMsgs[len(priorMsgs)-maxPrior:]
	}

	var sb strings.Builder
	if systemText != "" {
		sb.WriteString("[system]: ")
		sb.WriteString(systemText)
		sb.WriteByte('\n')
	}
	for _, msg := range priorMsgs {
		text := truncateCacheSegment(extractTextContent(msg.Content))
		if text == "" {
			continue
		}
		sb.WriteByte('[')
		sb.WriteString(msg.Role)
		sb.WriteString("]: ")
		sb.WriteString(text)
		sb.WriteByte('\n')
	}
	sb.WriteString("[user]: ")
	sb.WriteString(currentUserText)
	return sb.String()
}

// truncateCacheSegment caps a string to maxContextSegmentChars characters.
func truncateCacheSegment(s string) string {
	if len(s) <= maxContextSegmentChars {
		return s
	}
	return s[:maxContextSegmentChars]
}

// ExtractQueryFromOpenAIRequest parses an OpenAI request and extracts the user query
func ExtractQueryFromOpenAIRequest(requestBody []byte) (string, string, error) {
	var req OpenAIRequest
	if err := json.Unmarshal(requestBody, &req); err != nil {
		return "", "", fmt.Errorf("invalid request body: %w", err)
	}

	var userMessages []string
	for _, msg := range req.Messages {
		if msg.Role == "user" {
			text := extractTextContent(msg.Content)
			if text != "" {
				userMessages = append(userMessages, text)
			}
		}
	}

	query := ""
	if len(userMessages) > 0 {
		query = userMessages[len(userMessages)-1]
	}

	return req.Model, query, nil
}
