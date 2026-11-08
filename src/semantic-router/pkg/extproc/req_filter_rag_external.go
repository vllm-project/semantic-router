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

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/match"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	httputil "github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/http"
)

const defaultExternalRAGMaxResponseBytes int64 = 4 * 1024 * 1024

var (
	externalAPITransport     *http.Transport
	externalAPITransportOnce sync.Once
)

func externalAPIClient(timeout time.Duration) *http.Client {
	externalAPITransportOnce.Do(func() {
		externalAPITransport = &http.Transport{
			MaxIdleConns:        100,
			MaxIdleConnsPerHost: 10,
			IdleConnTimeout:     90 * time.Second,
		}
	})
	return &http.Client{Timeout: timeout, Transport: externalAPITransport}
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

// recallFromExternalAPI recalls documents from an external search API. The
// service ranks them; a document it scores reports that score as its
// similarity (Weaviate's as one less its distance), and one it does not
// score counts as similarity 1, the service having applied the threshold it
// was sent.
func (r *OpenAIRouter) recallFromExternalAPI(traceCtx context.Context, ctx *RequestContext, source *config.ContextSource, query string, recall config.MatchRecall) ([]match.Scored, error) {
	apiConfig, err := source.ExternalAPIBackendConfig()
	if err != nil {
		return nil, fmt.Errorf("invalid external API RAG config: %w", err)
	}
	var requestBody []byte
	switch apiConfig.RequestFormat {
	case "pinecone", "weaviate":
		requestBody, err = r.buildVectorSearchRequest(traceCtx, ctx, query, recall.TopK, apiConfig)
	case "elasticsearch":
		requestBody, err = buildElasticsearchRequest(query, recall.TopK)
	case "custom":
		requestBody, err = buildCustomRequest(query, recall, apiConfig.RequestTemplate)
	default:
		return nil, fmt.Errorf("unsupported request format: %s", apiConfig.RequestFormat)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to build request: %w", err)
	}
	apiResponse, latency, err := r.sendExternalRAGRequest(traceCtx, apiConfig, requestBody)
	if err != nil {
		return nil, err
	}
	ctx.RAGRetrievalLatency = latency
	texts, similarities, err := extractDocumentsFromResponse(apiResponse, apiConfig.RequestFormat)
	if err != nil {
		return nil, fmt.Errorf("failed to extract context: %w", err)
	}
	logging.Infof("Recalled %d documents from external API (latency: %.3fs, format: %s)", len(texts), latency, apiConfig.RequestFormat)
	return scoredDocuments(texts, similarities), nil
}

// buildVectorSearchRequest embeds the query for the formats that search by vector.
func (r *OpenAIRouter) buildVectorSearchRequest(traceCtx context.Context, ctx *RequestContext, query string, topK int, apiConfig *config.ExternalAPIRAGConfig) ([]byte, error) {
	queryEmbedding, err := r.ragQueryEmbedding(traceCtx, query, ctx, apiConfig.Embedding)
	if err != nil {
		return nil, err
	}
	if apiConfig.RequestFormat == "pinecone" {
		return buildPineconeRequest(queryEmbedding, topK)
	}
	return buildWeaviateRequest(queryEmbedding, topK)
}

// sendExternalRAGRequest posts one request body and decodes the response,
// returning how long the call took.
func (r *OpenAIRouter) sendExternalRAGRequest(traceCtx context.Context, apiConfig *config.ExternalAPIRAGConfig, requestBody []byte) (map[string]interface{}, float64, error) {
	req, err := http.NewRequestWithContext(traceCtx, "POST", apiConfig.Endpoint, bytes.NewBuffer(requestBody))
	if err != nil {
		return nil, 0, fmt.Errorf("failed to create request: %w", err)
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
			return nil, 0, fmt.Errorf("invalid auth header name: %w", validateErr)
		}

		// Validate and sanitize API key to prevent header injection
		sanitizedAPIKey, validateErr := validateHeaderValue(apiConfig.APIKey)
		if validateErr != nil {
			logging.Errorf("Failed to sanitize API key: %v", validateErr)
			return nil, 0, fmt.Errorf("invalid API key format")
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

	client := externalAPIClient(apiConfig.GetTimeout())

	// Execute request
	start := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		return nil, 0, fmt.Errorf("API request failed: %w", err)
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	latency := time.Since(start).Seconds()

	if resp.StatusCode != http.StatusOK {
		// Limit error response body size to prevent memory exhaustion
		const maxErrorBodySize = 1024 * 10 // 10KB limit
		limitedReader := io.LimitReader(resp.Body, maxErrorBodySize)
		body, _ := io.ReadAll(limitedReader)
		return nil, latency, fmt.Errorf("API returned status %d: %s", resp.StatusCode, string(body))
	}

	apiResponse, decodeErr := decodeExternalRAGResponse(resp.Body, apiConfig.MaxResponseBytes)
	if decodeErr != nil {
		return nil, latency, fmt.Errorf("failed to parse response: %w", decodeErr)
	}
	return apiResponse, latency, nil
}

func decodeExternalRAGResponse(body io.Reader, maxResponseBytes int64) (map[string]interface{}, error) {
	if maxResponseBytes == 0 {
		maxResponseBytes = defaultExternalRAGMaxResponseBytes
	}
	responseBody, err := httputil.ReadLimitedBody(body, maxResponseBytes)
	if err != nil {
		return nil, fmt.Errorf("read external RAG response: %w", err)
	}

	var apiResponse map[string]interface{}
	if err := json.NewDecoder(bytes.NewReader(responseBody)).Decode(&apiResponse); err != nil {
		return nil, err
	}
	return apiResponse, nil
}

// buildPineconeRequest builds a Pinecone query request
func buildPineconeRequest(queryEmbedding []float32, topK int) ([]byte, error) {
	request := map[string]interface{}{
		"vector":          queryEmbedding,
		"topK":            topK,
		"includeMetadata": true,
		"filter":          map[string]interface{}{},
	}
	return json.Marshal(request)
}

// buildWeaviateRequest builds a Weaviate query request
func buildWeaviateRequest(queryEmbedding []float32, topK int) ([]byte, error) {
	// Properly JSON-encode the vector for GraphQL query
	embeddingJSON, err := json.Marshal(queryEmbedding)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal embedding: %w", err)
	}
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
	return json.Marshal(map[string]interface{}{"query": query})
}

// buildElasticsearchRequest builds an Elasticsearch query request
func buildElasticsearchRequest(query string, topK int) ([]byte, error) {
	request := map[string]interface{}{
		"query": map[string]interface{}{
			"match": map[string]interface{}{
				"content": query,
			},
		},
		"size": topK,
	}
	return json.Marshal(request)
}

// buildCustomRequest builds a custom request from its template: the query,
// top_k and threshold (recall's min_similarity, else 0) substitute their
// placeholders.
func buildCustomRequest(query string, recall config.MatchRecall, template string) ([]byte, error) {
	if template == "" {
		return nil, fmt.Errorf("request template is required for custom format")
	}
	threshold := 0.0
	if recall.MinSimilarity != nil {
		threshold = *recall.MinSimilarity
	}
	// This is a simple substitution: user content that carries template
	// markers is escaped so it cannot inject placeholders.
	if strings.Contains(query, "{{") || strings.Contains(query, "}}") || strings.Contains(query, "${") {
		logging.Warnf("User content contains template markers, potential injection attempt")
		query = strings.ReplaceAll(query, "{{", "\\{\\{")
		query = strings.ReplaceAll(query, "}}", "\\}\\}")
		query = strings.ReplaceAll(query, "${", "\\${")
	}
	replaced := strings.ReplaceAll(template, "{{.Query}}", query)
	replaced = strings.ReplaceAll(replaced, "{{.TopK}}", fmt.Sprintf("%d", recall.TopK))
	replaced = strings.ReplaceAll(replaced, "{{.Threshold}}", fmt.Sprintf("%.3f", threshold))
	replaced = strings.ReplaceAll(replaced, "${user_content}", query)
	replaced = strings.ReplaceAll(replaced, "${top_k}", fmt.Sprintf("%d", recall.TopK))
	replaced = strings.ReplaceAll(replaced, "${threshold}", fmt.Sprintf("%.3f", threshold))
	return []byte(replaced), nil
}

// extractDocumentsFromResponse extracts the documents an API response
// returned and their similarities.
func extractDocumentsFromResponse(response map[string]interface{}, format string) ([]string, []float64, error) {
	switch format {
	case "pinecone":
		return extractPineconeDocuments(response)
	case "weaviate":
		return extractWeaviateDocuments(response)
	case "elasticsearch":
		return extractElasticsearchDocuments(response)
	case "custom":
		return extractCustomDocuments(response)
	default:
		return nil, nil, fmt.Errorf("unknown response format: %s", format)
	}
}

// documentScore reads a document's score field, 1 without one.
func documentScore(fields map[string]interface{}, key string) float64 {
	if score, ok := fields[key].(float64); ok {
		return score
	}
	return 1
}

// extractCustomDocuments reads a content or text field, else a results
// array of them, with each result's score.
func extractCustomDocuments(response map[string]interface{}) ([]string, []float64, error) {
	for _, field := range []string{"content", "text"} {
		if content, ok := response[field].(string); ok {
			return []string{content}, []float64{documentScore(response, "score")}, nil
		}
	}
	results, ok := response["results"].([]interface{})
	if !ok {
		return nil, nil, fmt.Errorf("unable to extract context from custom response format")
	}
	var texts []string
	var similarities []float64
	for _, result := range results {
		resultMap, ok := result.(map[string]interface{})
		if !ok {
			continue
		}
		for _, field := range []string{"content", "text"} {
			if content, ok := resultMap[field].(string); ok {
				texts, similarities = append(texts, content), append(similarities, documentScore(resultMap, "score"))
				break
			}
		}
	}
	return texts, similarities, nil
}

// extractPineconeDocuments extracts the matches' metadata content and scores.
func extractPineconeDocuments(response map[string]interface{}) ([]string, []float64, error) {
	matches, ok := response["matches"].([]interface{})
	if !ok {
		return nil, nil, fmt.Errorf("no matches in Pinecone response")
	}
	var texts []string
	var similarities []float64
	for _, match := range matches {
		matchMap, ok := match.(map[string]interface{})
		if !ok {
			continue
		}
		metadata, ok := matchMap["metadata"].(map[string]interface{})
		if !ok {
			continue
		}
		for _, field := range []string{"content", "text"} {
			if content, ok := metadata[field].(string); ok {
				texts, similarities = append(texts, content), append(similarities, documentScore(matchMap, "score"))
				break
			}
		}
	}
	return texts, similarities, nil
}

// extractWeaviateDocuments extracts the documents' content; a document's
// similarity is one less its cosine distance.
func extractWeaviateDocuments(response map[string]interface{}) ([]string, []float64, error) {
	data, ok := response["data"].(map[string]interface{})
	if !ok {
		return nil, nil, fmt.Errorf("no data in Weaviate response")
	}
	get, ok := data["Get"].(map[string]interface{})
	if !ok {
		return nil, nil, fmt.Errorf("no Get in Weaviate response")
	}
	document, ok := get["Document"].([]interface{})
	if !ok {
		return nil, nil, fmt.Errorf("no Document in Weaviate response")
	}
	var texts []string
	var similarities []float64
	for _, doc := range document {
		docMap, ok := doc.(map[string]interface{})
		if !ok {
			continue
		}
		content, ok := docMap["content"].(string)
		if !ok {
			continue
		}
		similarity := 1.0
		if additional, ok := docMap["_additional"].(map[string]interface{}); ok {
			if distance, ok := additional["distance"].(float64); ok {
				similarity = 1 - distance
			}
		}
		texts, similarities = append(texts, content), append(similarities, similarity)
	}
	return texts, similarities, nil
}

// extractElasticsearchDocuments extracts the hits' source content and scores.
func extractElasticsearchDocuments(response map[string]interface{}) ([]string, []float64, error) {
	hits, ok := response["hits"].(map[string]interface{})
	if !ok {
		return nil, nil, fmt.Errorf("no hits in Elasticsearch response")
	}
	hitsArray, ok := hits["hits"].([]interface{})
	if !ok {
		return nil, nil, fmt.Errorf("no hits array in Elasticsearch response")
	}
	var texts []string
	var similarities []float64
	for _, hit := range hitsArray {
		hitMap, ok := hit.(map[string]interface{})
		if !ok {
			continue
		}
		source, ok := hitMap["_source"].(map[string]interface{})
		if !ok {
			continue
		}
		for _, field := range []string{"content", "text"} {
			if content, ok := source[field].(string); ok {
				texts, similarities = append(texts, content), append(similarities, documentScore(hitMap, "_score"))
				break
			}
		}
	}
	return texts, similarities, nil
}
