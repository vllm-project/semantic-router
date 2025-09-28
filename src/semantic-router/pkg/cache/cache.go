// Copyright 2025 The vLLM Semantic Router Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package cache

import (
	"encoding/json"
	"fmt"
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
		if msg.Role == "user" {
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
