package cache

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
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

// scopeNamespaceRepeat controls how many times the user namespace token is
// repeated in the scoped prefix. Repeating the token amplifies its weight in
// the embedding so that queries from different users land in distinct regions
// of the vector space, even when the original question text is identical.
const scopeNamespaceRepeat = 3

// ScopeQueryToUser adds a deterministic user namespace to the cache query.
// If userID is empty, the original query is returned unchanged for backward compatibility.
func ScopeQueryToUser(query string, userID string) string {
	normalizedUserID := strings.TrimSpace(userID)
	if normalizedUserID == "" || query == "" {
		return query
	}

	namespace := userScopeNamespace(normalizedUserID)
	tokens := make([]string, scopeNamespaceRepeat)
	for i := range tokens {
		tokens[i] = namespace
	}
	// Single-line prefix so log/trace consumers never see raw newlines from user text.
	return fmt.Sprintf("cache-scope %s %s", strings.Join(tokens, " "), query)
}

var (
	userScopeSecretOnce   sync.Once
	cachedUserScopeSecret string
)

func userScopeNamespace(userID string) string {
	// Use a keyed construction (HMAC-SHA256) to avoid exposing a plain hash
	// of the userID, which is vulnerable to offline guessing attacks when
	// user IDs are predictable (e.g., emails, incremental IDs).
	//
	// The secret key is expected to be provided via environment variable
	// USER_SCOPE_NAMESPACE_SECRET so that it is not hard-coded in source
	// and can be managed per deployment.
	userScopeSecretOnce.Do(func() {
		cachedUserScopeSecret = os.Getenv("USER_SCOPE_NAMESPACE_SECRET")
	})
	secret := cachedUserScopeSecret
	if secret != "" {
		mac := hmac.New(sha256.New, []byte(secret))
		_, _ = mac.Write([]byte(userID))
		sum := mac.Sum(nil)
		return fmt.Sprintf("%x", sum[:8])
	}

	// Fallback to the previous behavior if no secret is configured, to avoid
	// breaking existing deployments. For production use, configuring a secret
	// is strongly recommended.
	digest := sha256.Sum256([]byte(userID))
	return fmt.Sprintf("%x", digest[:8])
}
