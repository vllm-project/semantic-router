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
	return fmt.Sprintf("%s%s %s", cacheScopePrefix, strings.Join(tokens, " "), query)
}

// cacheScopePrefix is the marker ScopeQueryToUser writes before the repeated
// user namespace tokens. Backends use it to recover the namespace for a hard
// scope check (see CacheScopeNamespaceOf / SameCacheScope).
const cacheScopePrefix = "cache-scope "

// CacheScopeNamespaceOf returns the user-scope namespace embedded in a scoped
// query, or "" if the query is unscoped (the global/anonymous scope).
// ScopeQueryToUser formats scoped queries as
// "cache-scope <ns> <ns> <ns> <query>", so the namespace is the first
// whitespace-delimited token after the prefix.
func CacheScopeNamespaceOf(query string) string {
	if !strings.HasPrefix(query, cacheScopePrefix) {
		return ""
	}
	rest := query[len(cacheScopePrefix):]
	if i := strings.IndexByte(rest, ' '); i >= 0 {
		return rest[:i]
	}
	return rest
}

// SameCacheScope reports whether two queries belong to the same user scope.
// It is a HARD equality check on the namespace, independent of embedding
// similarity: a cache backend must require it before returning a hit so one
// user can never receive another user's cached response (the embedding prefix
// alone does not separate users reliably for long queries). Two unscoped
// queries share the empty global scope; a scoped query never matches an
// unscoped one.
//
// This is a convenience helper for backends that compare two raw queries
// (e.g. Redis/Milvus/Qdrant/Valkey, when they adopt the gate). The in-memory
// search path intentionally does NOT use it: it recovers the requester's
// namespace once via CacheScopeNamespaceOf and compares that against each
// candidate, instead of re-parsing the full query on both sides per entry.
func SameCacheScope(a, b string) bool {
	return CacheScopeNamespaceOf(a) == CacheScopeNamespaceOf(b)
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

const defaultEmbeddingModel = "bert"

// normalizeEmbeddingModel normalizes the embedding model name by trimming whitespace and converting to lowercase.
// If the resulting model name is empty, it returns "bert" as the default embedding model.
func normalizeEmbeddingModel(model string) string {
	normalized := strings.ToLower(strings.TrimSpace(model))
	if normalized == "" {
		return defaultEmbeddingModel
	}
	return normalized
}

func semanticCacheEmbeddingDimension(configured int, embeddingModel string) int {
	if configured > 0 {
		return configured
	}

	switch normalizeEmbeddingModel(embeddingModel) {
	case "qwen3":
		return 1024
	case "gemma":
		return 768
	case "mmbert":
		return 768
	case "multimodal":
		return 384
	default:
		return 384
	}
}
