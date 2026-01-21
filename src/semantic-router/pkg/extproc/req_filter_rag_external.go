package extproc

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
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
	apiConfig, ok := ragConfig.BackendConfig.(*config.ExternalAPIRAGConfig)
	if !ok {
		return "", fmt.Errorf("invalid external API RAG config")
	}

	// Build request based on format
	var requestBody []byte
	var err error

	switch apiConfig.RequestFormat {
	case "pinecone":
		requestBody, err = r.buildPineconeRequest(ctx, ragConfig)
	case "weaviate":
		requestBody, err = r.buildWeaviateRequest(ctx, ragConfig)
	case "elasticsearch":
		requestBody, err = r.buildElasticsearchRequest(ctx, ragConfig)
	case "custom":
		requestBody, err = r.buildCustomRequest(ctx, ragConfig, apiConfig.RequestTemplate)
	default:
		return "", fmt.Errorf("unsupported request format: %s", apiConfig.RequestFormat)
	}

	if err != nil {
		return "", fmt.Errorf("failed to build request: %w", err)
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
	defer resp.Body.Close()

	latency := time.Since(start).Seconds()
	ctx.RAGRetrievalLatency = latency

	if resp.StatusCode != http.StatusOK {
		// Limit error response body size to prevent memory exhaustion
		const maxErrorBodySize = 1024 * 10 // 10KB limit
		limitedReader := io.LimitReader(resp.Body, maxErrorBodySize)
		body, _ := io.ReadAll(limitedReader)
		return "", fmt.Errorf("API returned status %d: %s", resp.StatusCode, string(body))
	}

	// Parse response
	var apiResponse map[string]interface{}
	if decodeErr := json.NewDecoder(resp.Body).Decode(&apiResponse); decodeErr != nil {
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

	// Simple template substitution (can be enhanced with proper template engine)
	query := ctx.UserContent
	topK := 5
	if ragConfig.TopK != nil {
		topK = *ragConfig.TopK
	}
	threshold := 0.7
	if ragConfig.SimilarityThreshold != nil {
		threshold = float64(*ragConfig.SimilarityThreshold)
	}

	// Replace template variables with basic validation
	// Note: This is a simple substitution. For production use, consider using
	// a proper template engine (e.g., text/template) with escaping.
	// Basic validation: check if user content contains template markers (could indicate injection attempt)
	if strings.Contains(query, "{{") || strings.Contains(query, "}}") || strings.Contains(query, "${") {
		logging.Warnf("User content contains template markers, potential injection attempt")
		// Escape the markers to prevent injection
		query = strings.ReplaceAll(query, "{{", "\\{\\{")
		query = strings.ReplaceAll(query, "}}", "\\}\\}")
		query = strings.ReplaceAll(query, "${", "\\${")
	}

	replaced := strings.ReplaceAll(template, "{{.Query}}", query)
	replaced = strings.ReplaceAll(replaced, "{{.TopK}}", fmt.Sprintf("%d", topK))
	replaced = strings.ReplaceAll(replaced, "{{.Threshold}}", fmt.Sprintf("%.3f", threshold))
	replaced = strings.ReplaceAll(replaced, "${user_content}", query)
	replaced = strings.ReplaceAll(replaced, "${top_k}", fmt.Sprintf("%d", topK))
	replaced = strings.ReplaceAll(replaced, "${threshold}", fmt.Sprintf("%.3f", threshold))

	return []byte(replaced), nil
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
