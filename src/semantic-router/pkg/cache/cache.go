package cache

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"strings"
)

const (
	cacheQueryEmbeddingWeight float32 = 0.75
	cacheUserNamespaceWeight  float32 = 0.66
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

func normalizeOptionalUserID(userID ...string) string {
	if len(userID) == 0 {
		return ""
	}
	return strings.TrimSpace(userID[0])
}

func scopeEmbeddingToUser(embedding []float32, userID string) []float32 {
	normalizedUserID := strings.TrimSpace(userID)
	if normalizedUserID == "" || len(embedding) == 0 {
		return embedding
	}

	normalizedEmbedding := normalizeEmbedding(embedding)
	namespaceVector := buildUserNamespaceVector(normalizedUserID, len(embedding))
	scoped := make([]float32, len(embedding))
	for i := range normalizedEmbedding {
		scoped[i] = normalizedEmbedding[i]*cacheQueryEmbeddingWeight +
			namespaceVector[i]*cacheUserNamespaceWeight
	}

	return normalizeEmbedding(scoped)
}

func normalizeEmbedding(embedding []float32) []float32 {
	normalized := make([]float32, len(embedding))
	copy(normalized, embedding)

	var magnitude float64
	for _, value := range normalized {
		magnitude += float64(value * value)
	}
	if magnitude == 0 {
		return normalized
	}

	scale := float32(1 / math.Sqrt(magnitude))
	for i := range normalized {
		normalized[i] *= scale
	}
	return normalized
}

func buildUserNamespaceVector(userID string, dimension int) []float32 {
	namespace := make([]float32, dimension)
	if dimension == 0 {
		return namespace
	}

	filled := 0
	blockIndex := 0
	for filled < dimension {
		digest := sha256.Sum256([]byte(fmt.Sprintf("%s:%d", userID, blockIndex)))
		for _, raw := range digest {
			if filled >= dimension {
				break
			}
			namespace[filled] = (float32(int(raw)) - 127.5) / 127.5
			filled++
		}
		blockIndex++
	}

	return normalizeEmbedding(namespace)
}
