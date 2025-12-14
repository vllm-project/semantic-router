package cache

import (
	"encoding/json"
	"fmt"
)

// ContentPart in multimodal messages
type ContentPart struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL struct {
		URL string `json:"url"`
	} `json:"image_url,omitempty"`
}

type FlexibleContent struct {
	value interface{}
}

// UnmarshalJSON handles both string and array content
func (fc *FlexibleContent) UnmarshalJSON(data []byte) error {
	var str string
	if err := json.Unmarshal(data, &str); err == nil {
		fc.value = str
		return nil
	}

	var arr []ContentPart
	if err := json.Unmarshal(data, &arr); err == nil {
		fc.value = arr
		return nil
	}

	fc.value = data
	return nil
}

// ExtractText extracts text content from flexible content
func (fc *FlexibleContent) ExtractText() string {
	if fc.value == nil {
		return ""
	}

	switch v := fc.value.(type) {
	case string:
		return v
	case []ContentPart:
		// Extract text from content parts
		var texts []string
		for _, part := range v {
			if part.Type == "text" && part.Text != "" {
				texts = append(texts, part.Text)
			}
		}
		if len(texts) > 0 {
			return texts[0]
		}
		return ""
	default:
		return ""
	}
}

// ChatMessage represents a message in the OpenAI chat format with role and content
type ChatMessage struct {
	Role    string          `json:"role"`
	Content FlexibleContent `json:"content"`
}

// OpenAIRequest represents the structure of an OpenAI API request
type OpenAIRequest struct {
	Model    string        `json:"model"`
	Messages []ChatMessage `json:"messages"`
}

// ExtractQueryFromOpenAIRequest parses an OpenAI request and extracts the user query
func ExtractQueryFromOpenAIRequest(requestBody []byte) (string, string, error) {
	var req OpenAIRequest
	if err := json.Unmarshal(requestBody, &req); err != nil {
		return "", "", fmt.Errorf("invalid request body: %w", err)
	}

	// Find user messages in the conversation
	var userMessages []string
	for _, msg := range req.Messages {
		if msg.Role == "user" {
			text := msg.Content.ExtractText()
			if text != "" {
				userMessages = append(userMessages, text)
			}
		}
	}

	// Use the most recent user message as the query
	query := ""
	if len(userMessages) > 0 {
		query = userMessages[len(userMessages)-1]
	}

	return req.Model, query, nil
}
