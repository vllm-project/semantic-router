package cache

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/openai/openai-go"
)

// extractUserContent returns the text portion of a user message's content.
// Handles both plain string and multimodal content array variants via the
// official openai-go SDK union type.
func extractUserContent(content openai.ChatCompletionUserMessageParamContentUnion) string {
	if content.OfString.Value != "" {
		return content.OfString.Value
	}
	var result string
	for _, part := range content.OfArrayOfContentParts {
		if part.OfText != nil {
			result += part.OfText.Text
		}
	}
	return result
}

// ExtractQueryFromOpenAIRequest parses an OpenAI request using the official
// openai-go SDK types and extracts the user query (last user message text)
// along with the model name.
func ExtractQueryFromOpenAIRequest(requestBody []byte) (string, string, error) {
	var req openai.ChatCompletionNewParams
	if err := json.Unmarshal(requestBody, &req); err != nil {
		return "", "", fmt.Errorf("invalid request body: %w", err)
	}

	var userMessages []string
	for _, msg := range req.Messages {
		if msg.OfUser != nil {
			text := extractUserContent(msg.OfUser.Content)
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
