/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package vectorstore

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

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// LlamaStackBackendConfig holds configuration for the Llama Stack backend.
type LlamaStackBackendConfig struct {
	// Endpoint is the base URL of the Llama Stack server (e.g. "http://localhost:8321").
	Endpoint string

	// AuthToken is an optional bearer token for authentication.
	AuthToken string

	// EmbeddingModel is the embedding model registered in Llama Stack.
	// Llama Stack uses this model to embed chunks and queries internally.
	// If empty, Llama Stack uses its configured default model.
	EmbeddingModel string

	// EmbeddingDimension is the embedding dimension to pass when creating
	// vector stores. If 0, Llama Stack auto-detects from the model.
	EmbeddingDimension int

	// RequestTimeoutSeconds is the HTTP timeout. Default: 30.
	RequestTimeoutSeconds int
}

// LlamaStackBackend implements VectorStoreBackend by delegating to a
// locally-running Llama Stack instance via its REST API.
//
// Design note: Llama Stack's OpenAI-compatible API generates its own vector
// store IDs (e.g. "vs_abc123") rather than using the name we pass during
// creation. This backend maintains an in-memory name→ID mapping (storeIDs)
// that is populated on CreateCollection and lazily refreshed via the list API
// when a name isn't cached (e.g. after a process restart).
//
// For InsertChunks, we send the pre-computed embeddings from EmbeddedChunk
// to the vector-io/insert API. For Search, we use the query text passed via
// the filter map ("_query_text" key) since Llama Stack's search endpoint
// operates on text queries, not embedding vectors.
type LlamaStackBackend struct {
	endpoint       string
	authToken      string
	embeddingModel string
	embeddingDim   int
	httpClient     *http.Client

	// storeIDs maps our vectorStoreID (name) to Llama Stack's generated ID.
	storeIDs map[string]string
	mu       sync.RWMutex
}

// NewLlamaStackBackend creates a new Llama Stack vector store backend.
func NewLlamaStackBackend(cfg LlamaStackBackendConfig) (*LlamaStackBackend, error) {
	if cfg.Endpoint == "" {
		return nil, fmt.Errorf("llama stack endpoint is required")
	}

	// Strip trailing slash to avoid double-slash in URLs.
	endpoint := strings.TrimRight(cfg.Endpoint, "/")

	timeout := cfg.RequestTimeoutSeconds
	if timeout <= 0 {
		timeout = 30
	}

	return &LlamaStackBackend{
		endpoint:       endpoint,
		authToken:      cfg.AuthToken,
		embeddingModel: cfg.EmbeddingModel,
		embeddingDim:   cfg.EmbeddingDimension,
		httpClient: &http.Client{
			Timeout: time.Duration(timeout) * time.Second,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 10,
				IdleConnTimeout:     90 * time.Second,
			},
		},
		storeIDs: make(map[string]string),
	}, nil
}

// CreateCollection creates a new vector store in Llama Stack.
// Maps to: POST /v1/vector_stores
//
// The response contains a generated ID (e.g. "vs_abc123") which is cached
// in the storeIDs map for use by all subsequent operations.
func (l *LlamaStackBackend) CreateCollection(ctx context.Context, vectorStoreID string, dimension int) error {
	body := map[string]interface{}{
		"name": vectorStoreID,
	}

	if l.embeddingModel != "" {
		body["embedding_model"] = l.embeddingModel
	}
	dim := dimension
	if dim <= 0 {
		dim = l.embeddingDim
	}
	if dim > 0 {
		body["embedding_dimension"] = dim
	}

	resp, err := l.doRequest(ctx, http.MethodPost, "/v1/vector_stores", body)
	if err != nil {
		return fmt.Errorf("failed to create vector store %s in Llama Stack: %w", vectorStoreID, err)
	}

	// Parse the generated ID from the response and cache the name→ID mapping.
	var result struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return fmt.Errorf("failed to parse create response for vector store %s: %w", vectorStoreID, err)
	}
	if result.ID == "" {
		return fmt.Errorf("llama stack returned empty ID for vector store %s", vectorStoreID)
	}

	l.mu.Lock()
	l.storeIDs[vectorStoreID] = result.ID
	l.mu.Unlock()

	logging.Infof("Llama Stack vector store created: id=%s, name=%s", result.ID, vectorStoreID)
	return nil
}

// DeleteCollection deletes a vector store from Llama Stack.
// Maps to: DELETE /v1/vector_stores/{vector_store_id}
func (l *LlamaStackBackend) DeleteCollection(ctx context.Context, vectorStoreID string) error {
	storeID, err := l.resolveStoreID(ctx, vectorStoreID)
	if err != nil {
		// If the store doesn't exist, treat as a no-op for idempotency.
		if strings.Contains(err.Error(), "not found") {
			return nil
		}
		return fmt.Errorf("failed to resolve vector store %s: %w", vectorStoreID, err)
	}

	_, err = l.doRequest(ctx, http.MethodDelete, "/v1/vector_stores/"+storeID, nil)
	if err != nil {
		return fmt.Errorf("failed to delete vector store %s from Llama Stack: %w", vectorStoreID, err)
	}

	// Remove from cache after successful deletion.
	l.mu.Lock()
	delete(l.storeIDs, vectorStoreID)
	l.mu.Unlock()

	return nil
}

// CollectionExists checks whether a vector store is registered in Llama Stack.
// Maps to: GET /v1/vector_stores/{vector_store_id}
func (l *LlamaStackBackend) CollectionExists(ctx context.Context, vectorStoreID string) (bool, error) {
	storeID, err := l.resolveStoreID(ctx, vectorStoreID)
	if err != nil {
		if strings.Contains(err.Error(), "not found") {
			return false, nil
		}
		return false, fmt.Errorf("failed to check vector store %s in Llama Stack: %w", vectorStoreID, err)
	}

	_, err = l.doRequest(ctx, http.MethodGet, "/v1/vector_stores/"+storeID, nil)
	if err != nil {
		if strings.Contains(err.Error(), "status 404") || strings.Contains(err.Error(), "status 400") {
			return false, nil
		}
		return false, fmt.Errorf("failed to check vector store %s in Llama Stack: %w", vectorStoreID, err)
	}
	return true, nil
}

// InsertChunks inserts pre-embedded chunks into a Llama Stack vector store.
// Maps to: POST /v1/vector-io/insert
//
// Each chunk must include a pre-computed embedding (EmbeddedChunk.Embedding).
func (l *LlamaStackBackend) InsertChunks(ctx context.Context, vectorStoreID string, chunks []EmbeddedChunk) error {
	if len(chunks) == 0 {
		return nil
	}

	storeID, err := l.resolveStoreID(ctx, vectorStoreID)
	if err != nil {
		return fmt.Errorf("failed to resolve vector store %s for insert: %w", vectorStoreID, err)
	}

	// Build chunks in Llama Stack v0.5 format: content, chunk_id,
	// chunk_metadata, embedding, embedding_model, embedding_dimension.
	lsChunks := make([]map[string]interface{}, len(chunks))
	for i, c := range chunks {
		embDim := len(c.Embedding)
		if embDim == 0 {
			embDim = l.embeddingDim
		}

		// Convert []float32 to []float64 for JSON serialization consistency.
		embedding := make([]float64, len(c.Embedding))
		for j, v := range c.Embedding {
			embedding[j] = float64(v)
		}

		lsChunks[i] = map[string]interface{}{
			"content":  c.Content,
			"chunk_id": c.ID,
			"chunk_metadata": map[string]interface{}{
				"document_id": c.ID,
				"source":      c.Filename,
			},
			"metadata": map[string]interface{}{
				"file_id":     c.FileID,
				"filename":    c.Filename,
				"chunk_index": c.ChunkIndex,
			},
			"embedding":           embedding,
			"embedding_model":     l.embeddingModel,
			"embedding_dimension": embDim,
		}
	}

	body := map[string]interface{}{
		"vector_store_id": storeID,
		"chunks":          lsChunks,
	}

	_, err = l.doRequest(ctx, http.MethodPost, "/v1/vector-io/insert", body)
	if err != nil {
		return fmt.Errorf("failed to insert %d chunks into vector store %s in Llama Stack: %w",
			len(chunks), vectorStoreID, err)
	}

	logging.Debugf("Inserted %d chunks into Llama Stack vector store %s (id=%s)", len(chunks), vectorStoreID, storeID)
	return nil
}

// DeleteByFileID removes all chunks belonging to a file from the vector store.
// Maps to: DELETE /v1/vector_stores/{vector_store_id}/files/{file_id}
func (l *LlamaStackBackend) DeleteByFileID(ctx context.Context, vectorStoreID string, fileID string) error {
	storeID, err := l.resolveStoreID(ctx, vectorStoreID)
	if err != nil {
		if strings.Contains(err.Error(), "not found") {
			logging.Warnf("Vector store %s not found in Llama Stack, nothing to delete", vectorStoreID)
			return nil
		}
		return fmt.Errorf("failed to resolve vector store %s for file deletion: %w", vectorStoreID, err)
	}

	path := fmt.Sprintf("/v1/vector_stores/%s/files/%s", storeID, fileID)
	_, err = l.doRequest(ctx, http.MethodDelete, path, nil)
	if err != nil {
		if strings.Contains(err.Error(), "status 404") {
			logging.Warnf("File %s not found in Llama Stack vector store %s (may already be deleted)", fileID, vectorStoreID)
			return nil
		}
		return fmt.Errorf("failed to delete file %s from vector store %s in Llama Stack: %w",
			fileID, vectorStoreID, err)
	}
	return nil
}

// Search performs a text-based similarity search in a Llama Stack vector store.
// Maps to: POST /v1/vector_stores/{vector_store_id}/search
//
// Llama Stack searches by text query, not by embedding vector. The query text
// must be passed via filter["_query_text"]; the queryEmbedding param is ignored.
func (l *LlamaStackBackend) Search(
	ctx context.Context, vectorStoreID string, queryEmbedding []float32,
	topK int, threshold float32, filter map[string]interface{},
) ([]SearchResult, error) {
	queryText, ok := filter["_query_text"].(string)
	if !ok || queryText == "" {
		return nil, fmt.Errorf(
			"llama_stack backend requires '_query_text' in the filter map for search; " +
				"Llama Stack searches by text query, not by embedding vector")
	}

	storeID, err := l.resolveStoreID(ctx, vectorStoreID)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve vector store %s for search: %w", vectorStoreID, err)
	}

	body := map[string]interface{}{
		"query":           queryText,
		"max_num_results": topK,
	}

	if fid, ok := filter["file_id"].(string); ok && fid != "" {
		body["filters"] = map[string]interface{}{
			"type":  "eq",
			"key":   "file_id",
			"value": fid,
		}
	}

	path := fmt.Sprintf("/v1/vector_stores/%s/search", storeID)
	resp, err := l.doRequest(ctx, http.MethodPost, path, body)
	if err != nil {
		return nil, fmt.Errorf("failed to search vector store %s in Llama Stack: %w", vectorStoreID, err)
	}

	var searchResp struct {
		Data []struct {
			Content  []map[string]interface{} `json:"content"`
			FileID   string                   `json:"file_id"`
			Filename string                   `json:"filename"`
			Score    float64                  `json:"score"`
		} `json:"data"`
	}
	if jsonErr := json.Unmarshal(resp, &searchResp); jsonErr != nil {
		return nil, fmt.Errorf("failed to parse Llama Stack search response: %w", jsonErr)
	}

	var results []SearchResult
	for _, hit := range searchResp.Data {
		if float32(hit.Score) < threshold {
			continue
		}

		// Llama Stack returns content as [{type: "text", text: "..."}].
		content := ""
		for _, c := range hit.Content {
			if c["type"] == "text" {
				if text, ok := c["text"].(string); ok {
					content = text
					break
				}
			}
		}

		results = append(results, SearchResult{
			FileID:   hit.FileID,
			Filename: hit.Filename,
			Content:  content,
			Score:    hit.Score,
		})
	}

	return results, nil
}

// Close releases HTTP client resources.
func (l *LlamaStackBackend) Close() error {
	l.httpClient.CloseIdleConnections()
	return nil
}

// resolveStoreID maps a vectorStoreID (name) to Llama Stack's generated ID.
// Checks the in-memory cache first, then falls back to listing all stores
// via the API (handles process restarts where the cache is empty).
func (l *LlamaStackBackend) resolveStoreID(ctx context.Context, name string) (string, error) {
	l.mu.RLock()
	if id, ok := l.storeIDs[name]; ok {
		l.mu.RUnlock()
		return id, nil
	}
	l.mu.RUnlock()

	// Cache miss — list all vector stores and find by name.
	resp, err := l.doRequest(ctx, http.MethodGet, "/v1/vector_stores", nil)
	if err != nil {
		return "", fmt.Errorf("failed to list vector stores while resolving %q: %w", name, err)
	}

	var listResp struct {
		Data []struct {
			ID        string `json:"id"`
			Name      string `json:"name"`
			CreatedAt int64  `json:"created_at"`
		} `json:"data"`
	}
	if err := json.Unmarshal(resp, &listResp); err != nil {
		return "", fmt.Errorf("failed to parse vector store list: %w", err)
	}

	// Find the most recently created store with the matching name.
	var bestID string
	var bestTime int64
	for _, s := range listResp.Data {
		if s.Name == name && s.CreatedAt >= bestTime {
			bestID = s.ID
			bestTime = s.CreatedAt
		}
	}

	if bestID == "" {
		return "", fmt.Errorf("vector store %q not found in Llama Stack", name)
	}

	l.mu.Lock()
	l.storeIDs[name] = bestID
	l.mu.Unlock()

	return bestID, nil
}

// doRequest is the shared HTTP helper for all Llama Stack API calls.
func (l *LlamaStackBackend) doRequest(
	ctx context.Context, method, path string, body interface{},
) ([]byte, error) {
	var reqBody io.Reader
	if body != nil {
		jsonBytes, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request body: %w", err)
		}
		reqBody = bytes.NewReader(jsonBytes)
	}

	url := l.endpoint + path
	req, err := http.NewRequestWithContext(ctx, method, url, reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if l.authToken != "" {
		req.Header.Set("Authorization", "Bearer "+l.authToken)
	}

	resp, err := l.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HTTP request to Llama Stack failed (%s %s): %w", method, path, err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read Llama Stack response body: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		errMsg := string(respBody)
		if len(errMsg) > 500 {
			errMsg = errMsg[:500] + "..."
		}
		return nil, fmt.Errorf("llama stack API error: %s %s returned status %d: %s",
			method, path, resp.StatusCode, errMsg)
	}

	return respBody, nil
}
