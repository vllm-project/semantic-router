package extproc

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"sync"
	"time"
	"unicode"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

var (
	// Shared HTTP client for external API requests with connection pooling
	externalAPIClient     *http.Client
	externalAPIClientOnce sync.Once
)

const defaultExternalAPIMaxResponseBodyBytes int64 = 16 << 20

// getExternalAPIClient returns a shared HTTP client for external API requests
func getExternalAPIClient() *http.Client {
	externalAPIClientOnce.Do(func() {
		externalAPIClient = &http.Client{
			Timeout: 30 * time.Second,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 10,
				IdleConnTimeout:     90 * time.Second,
			},
		}
	})
	return externalAPIClient
}

// validateHeaderName validates a header name to prevent header injection
// Header names must contain only printable ASCII characters and cannot contain
// newlines, carriage returns, or colons (which are used as separators)
func validateHeaderName(name string) error {
	if name == "" {
		return fmt.Errorf("header name cannot be empty")
	}

	for _, r := range name {
		if r < 32 || r > 126 {
			return fmt.Errorf("header name contains invalid character: %q", r)
		}
		if r == ':' || r == '\n' || r == '\r' {
			return fmt.Errorf("header name contains forbidden character: %q", r)
		}
	}

	return nil
}

// validateHeaderValue validates and sanitizes a header value to prevent header injection
// Removes newlines, carriage returns, and other control characters
func validateHeaderValue(value string) (string, error) {
	if value == "" {
		return "", nil
	}

	// Remove control characters (including newlines and carriage returns)
	var sanitized strings.Builder
	for _, r := range value {
		if unicode.IsPrint(r) || r == '\t' {
			sanitized.WriteRune(r)
		} else if r == '\n' || r == '\r' {
			// Replace newlines with spaces to prevent header injection
			sanitized.WriteRune(' ')
		}
		// Other control characters are silently removed
	}

	result := strings.TrimSpace(sanitized.String())
	if len(result) == 0 && len(value) > 0 {
		return "", fmt.Errorf("header value contains only invalid characters")
	}

	return result, nil
}

// retrieveFromExternalAPI retrieves context from external API backend
func (r *OpenAIRouter) retrieveFromExternalAPI(traceCtx context.Context, ctx *RequestContext, ragConfig *config.RAGPluginConfig) (string, error) {
	apiConfig, err := ragConfig.ExternalAPIBackendConfig()
	if err != nil {
		return "", fmt.Errorf("invalid external API RAG config: %w", err)
	}

	// Build request based on format
	var requestBody []byte
	var buildErr error

	switch apiConfig.RequestFormat {
	case "pinecone":
		requestBody, buildErr = r.buildPineconeRequest(ctx, ragConfig)
	case "weaviate":
		requestBody, buildErr = r.buildWeaviateRequest(ctx, ragConfig)
	case "elasticsearch":
		requestBody, buildErr = r.buildElasticsearchRequest(ctx, ragConfig)
	case "custom":
		requestBody, buildErr = r.buildCustomRequest(ctx, ragConfig, apiConfig.RequestTemplate)
	default:
		return "", fmt.Errorf("unsupported request format: %s", apiConfig.RequestFormat)
	}

	if buildErr != nil {
		return "", fmt.Errorf("failed to build request: %w", buildErr)
	}

	// Create HTTP request
	req, err := http.NewRequestWithContext(traceCtx, "POST", apiConfig.Endpoint, bytes.NewBuffer(requestBody))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	if apiConfig.APIKey != "" {
		authHeader := apiConfig.AuthHeader
		if authHeader == "" {
			authHeader = "Authorization"
		}

		// Validate header name
		if validateErr := validateHeaderName(authHeader); validateErr != nil {
			return "", fmt.Errorf("invalid auth header name: %w", validateErr)
		}

		// Validate and sanitize API key to prevent header injection
		sanitizedAPIKey, validateErr := validateHeaderValue(apiConfig.APIKey)
		if validateErr != nil {
			logging.Errorf("Failed to sanitize API key: %v", validateErr)
			return "", fmt.Errorf("invalid API key format")
		}

		req.Header.Set(authHeader, fmt.Sprintf("Bearer %s", sanitizedAPIKey))
	}

	// Add custom headers with validation
	for k, v := range apiConfig.Headers {
		// Validate header name
		if validateErr := validateHeaderName(k); validateErr != nil {
			logging.Warnf("Skipping invalid header name: %s (error: %v)", k, validateErr)
			continue
		}

		// Validate and sanitize header value
		sanitizedValue, validateErr := validateHeaderValue(v)
		if validateErr != nil {
			logging.Warnf("Skipping invalid header value for %s: %v", k, validateErr)
			continue
		}

		req.Header.Set(k, sanitizedValue)
	}

	// Use shared HTTP client with connection pooling
	client := getExternalAPIClient()

	// Override timeout if specified in config
	if apiConfig.TimeoutSeconds != nil {
		timeout := time.Duration(*apiConfig.TimeoutSeconds) * time.Second
		reqCtx, cancel := context.WithTimeout(traceCtx, timeout)
		defer cancel()
		req = req.WithContext(reqCtx)
	}

	// Execute request
	start := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("API request failed: %w", err)
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	latency := time.Since(start).Seconds()
	ctx.RAGRetrievalLatency = latency

	if resp.StatusCode != http.StatusOK {
		// Limit error response body size to prevent memory exhaustion
		const maxErrorBodySize = 1024 * 10 // 10KB limit
		limitedReader := io.LimitReader(resp.Body, maxErrorBodySize)
		body, _ := io.ReadAll(limitedReader)
		return "", fmt.Errorf("API returned status %d: %s", resp.StatusCode, string(body))
	}

	maxResponseBodyBytes := externalAPIResponseBodyLimit(apiConfig)
	responseBody, exceeded, readErr := readExternalAPIResponseBody(resp.Body, maxResponseBodyBytes)
	if readErr != nil {
		return "", fmt.Errorf("failed to read external API response: %w", readErr)
	}
	if exceeded {
		return "", fmt.Errorf("external API successful response exceeded configured limit of %d bytes", maxResponseBodyBytes)
	}

	// Parse the complete bounded response. json.Unmarshal rejects trailing
	// non-whitespace data and, unlike decoding a limited stream, cannot accept a
	// valid truncated prefix.
	var apiResponse map[string]interface{}
	if decodeErr := json.Unmarshal(responseBody, &apiResponse); decodeErr != nil {
		return "", fmt.Errorf("failed to parse response: %w", decodeErr)
	}

	// Extract context based on format
	context, err := r.extractContextFromResponse(apiResponse, apiConfig.RequestFormat)
	if err != nil {
		return "", fmt.Errorf("failed to extract context: %w", err)
	}

	logging.Infof("Retrieved context from external API (latency: %.3fs, format: %s)", latency, apiConfig.RequestFormat)
	return context, nil
}

func externalAPIResponseBodyLimit(apiConfig *config.ExternalAPIRAGConfig) int64 {
	if apiConfig.MaxResponseBodyBytes != nil {
		return *apiConfig.MaxResponseBodyBytes
	}
	return defaultExternalAPIMaxResponseBodyBytes
}

func readExternalAPIResponseBody(reader io.Reader, maxBytes int64) ([]byte, bool, error) {
	readLimit := maxBytes
	if maxBytes < math.MaxInt64 {
		readLimit++
	}
	body, err := io.ReadAll(io.LimitReader(reader, readLimit))
	if err != nil {
		return nil, false, err
	}
	if int64(len(body)) > maxBytes {
		return nil, true, nil
	}
	return body, false, nil
}

// buildPineconeRequest builds a Pinecone query request
func (r *OpenAIRouter) buildPineconeRequest(ctx *RequestContext, ragConfig *config.RAGPluginConfig) ([]byte, error) {
	// Generate embedding
	queryEmbedding, err := candle_binding.GetEmbedding(ctx.UserContent, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding: %w", err)
	}

	topK := 5
	if ragConfig.TopK != nil {
		topK = *ragConfig.TopK
	}

	request := map[string]interface{}{
		"vector":          queryEmbedding,
		"topK":            topK,
		"includeMetadata": true,
		"filter":          map[string]interface{}{},
	}

	return json.Marshal(request)
}

// buildWeaviateRequest builds a Weaviate query request
func (r *OpenAIRouter) buildWeaviateRequest(ctx *RequestContext, ragConfig *config.RAGPluginConfig) ([]byte, error) {
	// Generate embedding
	queryEmbedding, err := candle_binding.GetEmbedding(ctx.UserContent, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding: %w", err)
	}

	topK := 5
	if ragConfig.TopK != nil {
		topK = *ragConfig.TopK
	}

	// Properly JSON-encode the vector for GraphQL query
	embeddingJSON, err := json.Marshal(queryEmbedding)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal embedding: %w", err)
	}

	// Build GraphQL query with properly encoded vector
	query := fmt.Sprintf(`
		{
			Get {
				Document(
					nearVector: {
						vector: %s
					}
					limit: %d
				) {
					content
					_additional {
						distance
					}
				}
			}
		}`, string(embeddingJSON), topK)

	request := map[string]interface{}{
		"query": query,
	}

	return json.Marshal(request)
}

// buildElasticsearchRequest builds an Elasticsearch query request
func (r *OpenAIRouter) buildElasticsearchRequest(ctx *RequestContext, ragConfig *config.RAGPluginConfig) ([]byte, error) {
	topK := 5
	if ragConfig.TopK != nil {
		topK = *ragConfig.TopK
	}

	request := map[string]interface{}{
		"query": map[string]interface{}{
			"match": map[string]interface{}{
				"content": ctx.UserContent,
			},
		},
		"size": topK,
	}

	return json.Marshal(request)
}

// buildCustomRequest builds a custom request using template
func (r *OpenAIRouter) buildCustomRequest(ctx *RequestContext, ragConfig *config.RAGPluginConfig, template string) ([]byte, error) {
	if template == "" {
		return nil, fmt.Errorf("request template is required for custom format")
	}

	topK := 5
	if ragConfig.TopK != nil {
		topK = *ragConfig.TopK
	}
	threshold := 0.7
	if ragConfig.SimilarityThreshold != nil {
		threshold = float64(*ragConfig.SimilarityThreshold)
	}

	// Bare placeholders such as `"top_k": ${top_k}` are not valid JSON.
	// Quote only exact placeholder tokens that occur outside JSON strings, then
	// decode the template before inserting any user-controlled value. This keeps
	// the configured object/array shape authoritative while retaining the two
	// placeholder syntaxes supported by earlier configurations.
	normalized := quoteBareCustomRequestPlaceholders(template)
	document, err := decodeCustomRequestTemplate(normalized)
	if err != nil {
		return nil, fmt.Errorf("invalid custom request template JSON: %w", err)
	}

	typedDocument, err := substituteCustomRequestPlaceholders(document, ctx.UserContent, topK, threshold)
	if err != nil {
		return nil, err
	}

	requestBody, err := json.Marshal(typedDocument)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal custom request template: %w", err)
	}
	return requestBody, nil
}

func decodeCustomRequestTemplate(template string) (interface{}, error) {
	decoder := json.NewDecoder(strings.NewReader(template))
	decoder.UseNumber()

	var document interface{}
	if err := decoder.Decode(&document); err != nil {
		return nil, err
	}

	var trailing interface{}
	if err := decoder.Decode(&trailing); err != io.EOF {
		if err == nil {
			return nil, fmt.Errorf("template contains multiple JSON values")
		}
		return nil, fmt.Errorf("invalid trailing data: %w", err)
	}
	return document, nil
}

var customRequestPlaceholders = []string{
	"{{.Query}}",
	"{{.TopK}}",
	"{{.Threshold}}",
	"${user_content}",
	"${top_k}",
	"${threshold}",
}

func quoteBareCustomRequestPlaceholders(template string) string {
	var result strings.Builder
	result.Grow(len(template))

	inString := false
	escaped := false
	for i := 0; i < len(template); {
		if inString {
			ch := template[i]
			result.WriteByte(ch)
			i++
			if escaped {
				escaped = false
				continue
			}
			switch ch {
			case '\\':
				escaped = true
			case '"':
				inString = false
			}
			continue
		}

		if template[i] == '"' {
			inString = true
			result.WriteByte(template[i])
			i++
			continue
		}

		if placeholder := customRequestPlaceholderAt(template, i); placeholder != "" {
			encoded, _ := json.Marshal(placeholder)
			result.Write(encoded)
			i += len(placeholder)
			continue
		}

		result.WriteByte(template[i])
		i++
	}

	return result.String()
}

func customRequestPlaceholderAt(value string, offset int) string {
	for _, placeholder := range customRequestPlaceholders {
		if strings.HasPrefix(value[offset:], placeholder) {
			return placeholder
		}
	}
	return ""
}

func containsCustomRequestPlaceholder(value string) bool {
	for _, placeholder := range customRequestPlaceholders {
		if strings.Contains(value, placeholder) {
			return true
		}
	}
	return false
}

func substituteCustomRequestPlaceholders(value interface{}, query string, topK int, threshold float64) (interface{}, error) {
	switch typed := value.(type) {
	case map[string]interface{}:
		result := make(map[string]interface{}, len(typed))
		for key, child := range typed {
			if containsCustomRequestPlaceholder(key) {
				return nil, fmt.Errorf("custom request template placeholders are not allowed in object keys: %q", key)
			}
			replaced, err := substituteCustomRequestPlaceholders(child, query, topK, threshold)
			if err != nil {
				return nil, err
			}
			result[key] = replaced
		}
		return result, nil
	case []interface{}:
		result := make([]interface{}, len(typed))
		for i, child := range typed {
			replaced, err := substituteCustomRequestPlaceholders(child, query, topK, threshold)
			if err != nil {
				return nil, err
			}
			result[i] = replaced
		}
		return result, nil
	case string:
		if replacement, ok := exactCustomRequestPlaceholder(typed, query, topK, threshold); ok {
			return replacement, nil
		}
		return interpolateCustomRequestPlaceholders(typed, query, topK, threshold), nil
	default:
		return value, nil
	}
}

func exactCustomRequestPlaceholder(value, query string, topK int, threshold float64) (interface{}, bool) {
	switch value {
	case "{{.Query}}", "${user_content}":
		return query, true
	case "{{.TopK}}", "${top_k}":
		return topK, true
	case "{{.Threshold}}", "${threshold}":
		return json.Number(fmt.Sprintf("%.3f", threshold)), true
	default:
		return nil, false
	}
}

func interpolateCustomRequestPlaceholders(value, query string, topK int, threshold float64) string {
	var result strings.Builder
	result.Grow(len(value))

	for i := 0; i < len(value); {
		placeholder := customRequestPlaceholderAt(value, i)
		if placeholder == "" {
			result.WriteByte(value[i])
			i++
			continue
		}

		switch placeholder {
		case "{{.Query}}", "${user_content}":
			result.WriteString(query)
		case "{{.TopK}}", "${top_k}":
			result.WriteString(fmt.Sprintf("%d", topK))
		case "{{.Threshold}}", "${threshold}":
			result.WriteString(fmt.Sprintf("%.3f", threshold))
		}
		i += len(placeholder)
	}

	return result.String()
}

// extractContextFromResponse extracts context from API response based on format
func (r *OpenAIRouter) extractContextFromResponse(response map[string]interface{}, format string) (string, error) {
	switch format {
	case "pinecone":
		return r.extractPineconeContext(response)
	case "weaviate":
		return r.extractWeaviateContext(response)
	case "elasticsearch":
		return r.extractElasticsearchContext(response)
	case "custom":
		// For custom format, assume response has a "content" or "text" field
		if content, ok := response["content"].(string); ok {
			return content, nil
		}
		if text, ok := response["text"].(string); ok {
			return text, nil
		}
		// Try to extract from results array
		if results, ok := response["results"].([]interface{}); ok {
			var parts []string
			for _, result := range results {
				if resultMap, ok := result.(map[string]interface{}); ok {
					if content, ok := resultMap["content"].(string); ok {
						parts = append(parts, content)
					} else if text, ok := resultMap["text"].(string); ok {
						parts = append(parts, text)
					}
				}
			}
			return strings.Join(parts, "\n\n---\n\n"), nil
		}
		return "", fmt.Errorf("unable to extract context from custom response format")
	default:
		return "", fmt.Errorf("unknown response format: %s", format)
	}
}

// extractPineconeContext extracts context from Pinecone response
func (r *OpenAIRouter) extractPineconeContext(response map[string]interface{}) (string, error) {
	matches, ok := response["matches"].([]interface{})
	if !ok {
		return "", fmt.Errorf("no matches in Pinecone response")
	}

	var parts []string
	for _, match := range matches {
		if matchMap, ok := match.(map[string]interface{}); ok {
			if metadata, ok := matchMap["metadata"].(map[string]interface{}); ok {
				if content, ok := metadata["content"].(string); ok {
					parts = append(parts, content)
				} else if text, ok := metadata["text"].(string); ok {
					parts = append(parts, text)
				}
			}
		}
	}

	if len(parts) == 0 {
		return "", fmt.Errorf("no content found in Pinecone matches")
	}

	return strings.Join(parts, "\n\n---\n\n"), nil
}

// extractWeaviateContext extracts context from Weaviate response
func (r *OpenAIRouter) extractWeaviateContext(response map[string]interface{}) (string, error) {
	data, ok := response["data"].(map[string]interface{})
	if !ok {
		return "", fmt.Errorf("no data in Weaviate response")
	}

	get, ok := data["Get"].(map[string]interface{})
	if !ok {
		return "", fmt.Errorf("no Get in Weaviate response")
	}

	document, ok := get["Document"].([]interface{})
	if !ok {
		return "", fmt.Errorf("no Document in Weaviate response")
	}

	var parts []string
	for _, doc := range document {
		if docMap, ok := doc.(map[string]interface{}); ok {
			if content, ok := docMap["content"].(string); ok {
				parts = append(parts, content)
			}
		}
	}

	if len(parts) == 0 {
		return "", fmt.Errorf("no content found in Weaviate documents")
	}

	return strings.Join(parts, "\n\n---\n\n"), nil
}

// extractElasticsearchContext extracts context from Elasticsearch response
func (r *OpenAIRouter) extractElasticsearchContext(response map[string]interface{}) (string, error) {
	hits, ok := response["hits"].(map[string]interface{})
	if !ok {
		return "", fmt.Errorf("no hits in Elasticsearch response")
	}

	hitsArray, ok := hits["hits"].([]interface{})
	if !ok {
		return "", fmt.Errorf("no hits array in Elasticsearch response")
	}

	var parts []string
	for _, hit := range hitsArray {
		if hitMap, ok := hit.(map[string]interface{}); ok {
			if source, ok := hitMap["_source"].(map[string]interface{}); ok {
				if content, ok := source["content"].(string); ok {
					parts = append(parts, content)
				} else if text, ok := source["text"].(string); ok {
					parts = append(parts, text)
				}
			}
		}
	}

	if len(parts) == 0 {
		return "", fmt.Errorf("no content found in Elasticsearch hits")
	}

	return strings.Join(parts, "\n\n---\n\n"), nil
}
