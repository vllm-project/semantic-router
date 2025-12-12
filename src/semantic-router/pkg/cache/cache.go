package cache

import (
	"encoding/json"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/consts"
)

// ChatMessage represents a message in the OpenAI chat format with role and content
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
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
		if msg.Role == consts.USER {
			userMessages = append(userMessages, msg.Content)
		}
	}

	// Use the most recent user message as the query
	query := ""
	if len(userMessages) > 0 {
		query = userMessages[len(userMessages)-1]
	}

	return req.Model, query, nil
}
