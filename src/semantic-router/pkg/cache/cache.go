package cache

import (
	"encoding/json"
	"fmt"
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
