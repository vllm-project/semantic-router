package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// VectorStoreClient handles interactions with OpenAI Vector Store API
type VectorStoreClient struct {
	httpClient *http.Client
	baseURL    string
	apiKey     string
}

// NewVectorStoreClient creates a new OpenAI Vector Store client
func NewVectorStoreClient(baseURL string, apiKey string) *VectorStoreClient {
	return &VectorStoreClient{
		httpClient: &http.Client{
			Timeout: 60 * time.Second,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 10,
				IdleConnTimeout:     90 * time.Second,
			},
		},
		baseURL: baseURL,
		apiKey:  apiKey,
	}
}

// VectorStore represents an OpenAI vector store
type VectorStore struct {
	ID         string `json:"id"`
	Object     string `json:"object"`
	CreatedAt  int64  `json:"created_at"`
	Name       string `json:"name"`
	FileCounts struct {
		InProgress int `json:"in_progress"`
		Completed  int `json:"completed"`
		Failed     int `json:"failed"`
		Cancelled  int `json:"cancelled"`
		Total      int `json:"total"`
	} `json:"file_counts"`
	Status       string `json:"status,omitempty"`
	ExpiresAfter *struct {
		Anchor string `json:"anchor"`
		Days   int    `json:"days"`
	} `json:"expires_after,omitempty"`
	ExpiresAt    *int64                 `json:"expires_at,omitempty"`
	LastActiveAt *int64                 `json:"last_active_at,omitempty"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
}

// VectorStoreListResponse represents the response from listing vector stores
type VectorStoreListResponse struct {
	Object  string        `json:"object"`
	Data    []VectorStore `json:"data"`
	FirstID string        `json:"first_id,omitempty"`
	LastID  string        `json:"last_id,omitempty"`
	HasMore bool          `json:"has_more"`
}

// CreateVectorStoreRequest represents the request to create a vector store
type CreateVectorStoreRequest struct {
	Name         string   `json:"name,omitempty"`
	FileIDs      []string `json:"file_ids,omitempty"`
	ExpiresAfter *struct {
		Anchor string `json:"anchor"`
		Days   int    `json:"days"`
	} `json:"expires_after,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// CreateVectorStore creates a new vector store
func (c *VectorStoreClient) CreateVectorStore(ctx context.Context, req *CreateVectorStoreRequest) (*VectorStore, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/v1/vector_stores", bytes.NewBuffer(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to create vector store: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return nil, fmt.Errorf("create vector store failed with status %d: %s", resp.StatusCode, string(body))
	}

	var vectorStore VectorStore
	if err := json.NewDecoder(resp.Body).Decode(&vectorStore); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	logging.Infof("Created vector store %s (ID: %s)", req.Name, vectorStore.ID)
	return &vectorStore, nil
}

// ListVectorStores lists all vector stores
func (c *VectorStoreClient) ListVectorStores(ctx context.Context, limit int, order string, after string, before string) (*VectorStoreListResponse, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", c.baseURL+"/v1/vector_stores", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	// Add query parameters
	q := req.URL.Query()
	if limit > 0 {
		q.Set("limit", fmt.Sprintf("%d", limit))
	}
	if order != "" {
		q.Set("order", order)
	}
	if after != "" {
		q.Set("after", after)
	}
	if before != "" {
		q.Set("before", before)
	}
	req.URL.RawQuery = q.Encode()

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to list vector stores: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return nil, fmt.Errorf("list vector stores failed with status %d: %s", resp.StatusCode, string(body))
	}

	var listResp VectorStoreListResponse
	if err := json.NewDecoder(resp.Body).Decode(&listResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &listResp, nil
}

// GetVectorStore retrieves information about a specific vector store
func (c *VectorStoreClient) GetVectorStore(ctx context.Context, vectorStoreID string) (*VectorStore, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", fmt.Sprintf("%s/v1/vector_stores/%s", c.baseURL, vectorStoreID), nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to get vector store: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return nil, fmt.Errorf("get vector store failed with status %d: %s", resp.StatusCode, string(body))
	}

	var vectorStore VectorStore
	if err := json.NewDecoder(resp.Body).Decode(&vectorStore); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &vectorStore, nil
}

// UpdateVectorStoreRequest represents the request to update a vector store
type UpdateVectorStoreRequest struct {
	Name         string `json:"name,omitempty"`
	ExpiresAfter *struct {
		Anchor string `json:"anchor"`
		Days   int    `json:"days"`
	} `json:"expires_after,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// UpdateVectorStore updates a vector store
func (c *VectorStoreClient) UpdateVectorStore(ctx context.Context, vectorStoreID string, req *UpdateVectorStoreRequest) (*VectorStore, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", fmt.Sprintf("%s/v1/vector_stores/%s", c.baseURL, vectorStoreID), bytes.NewBuffer(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to update vector store: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return nil, fmt.Errorf("update vector store failed with status %d: %s", resp.StatusCode, string(body))
	}

	var vectorStore VectorStore
	if err := json.NewDecoder(resp.Body).Decode(&vectorStore); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &vectorStore, nil
}

// DeleteVectorStore deletes a vector store
func (c *VectorStoreClient) DeleteVectorStore(ctx context.Context, vectorStoreID string) error {
	req, err := http.NewRequestWithContext(ctx, "DELETE", fmt.Sprintf("%s/v1/vector_stores/%s", c.baseURL, vectorStoreID), nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to delete vector store: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return fmt.Errorf("delete vector store failed with status %d: %s", resp.StatusCode, string(body))
	}

	logging.Infof("Deleted vector store %s", vectorStoreID)
	return nil
}

// VectorStoreFile represents a file in a vector store
type VectorStoreFile struct {
	ID            string `json:"id"`
	Object        string `json:"object"`
	CreatedAt     int64  `json:"created_at"`
	VectorStoreID string `json:"vector_store_id"`
	Status        string `json:"status"`
	LastError     *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"last_error,omitempty"`
}

// VectorStoreFileListResponse represents the response from listing files in a vector store
type VectorStoreFileListResponse struct {
	Object  string            `json:"object"`
	Data    []VectorStoreFile `json:"data"`
	FirstID string            `json:"first_id,omitempty"`
	LastID  string            `json:"last_id,omitempty"`
	HasMore bool              `json:"has_more"`
}

// CreateVectorStoreFileRequest represents the request to attach a file to a vector store
type CreateVectorStoreFileRequest struct {
	FileID string `json:"file_id"`
}

// CreateVectorStoreFile attaches a file to a vector store
func (c *VectorStoreClient) CreateVectorStoreFile(ctx context.Context, vectorStoreID string, fileID string) (*VectorStoreFile, error) {
	reqBody := CreateVectorStoreFileRequest{FileID: fileID}
	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", fmt.Sprintf("%s/v1/vector_stores/%s/files", c.baseURL, vectorStoreID), bytes.NewBuffer(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to attach file to vector store: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return nil, fmt.Errorf("attach file failed with status %d: %s", resp.StatusCode, string(body))
	}

	var vectorStoreFile VectorStoreFile
	if err := json.NewDecoder(resp.Body).Decode(&vectorStoreFile); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	logging.Infof("Attached file %s to vector store %s", fileID, vectorStoreID)
	return &vectorStoreFile, nil
}

// ListVectorStoreFiles lists files in a vector store
func (c *VectorStoreClient) ListVectorStoreFiles(ctx context.Context, vectorStoreID string, limit int, order string, after string, before string, filter string) (*VectorStoreFileListResponse, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", fmt.Sprintf("%s/v1/vector_stores/%s/files", c.baseURL, vectorStoreID), nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	// Add query parameters
	q := req.URL.Query()
	if limit > 0 {
		q.Set("limit", fmt.Sprintf("%d", limit))
	}
	if order != "" {
		q.Set("order", order)
	}
	if after != "" {
		q.Set("after", after)
	}
	if before != "" {
		q.Set("before", before)
	}
	if filter != "" {
		q.Set("filter", filter)
	}
	req.URL.RawQuery = q.Encode()

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to list vector store files: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return nil, fmt.Errorf("list vector store files failed with status %d: %s", resp.StatusCode, string(body))
	}

	var listResp VectorStoreFileListResponse
	if err := json.NewDecoder(resp.Body).Decode(&listResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &listResp, nil
}

// DeleteVectorStoreFile detaches a file from a vector store
func (c *VectorStoreClient) DeleteVectorStoreFile(ctx context.Context, vectorStoreID string, fileID string) error {
	req, err := http.NewRequestWithContext(ctx, "DELETE", fmt.Sprintf("%s/v1/vector_stores/%s/files/%s", c.baseURL, vectorStoreID, fileID), nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to detach file from vector store: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return fmt.Errorf("detach file failed with status %d: %s", resp.StatusCode, string(body))
	}

	logging.Infof("Detached file %s from vector store %s", fileID, vectorStoreID)
	return nil
}

// VectorStoreSearchResult represents a search result from vector store
type VectorStoreSearchResult struct {
	Content  string  `json:"content"`
	Filename string  `json:"filename"`
	Score    float64 `json:"score,omitempty"`
}

// VectorStoreSearchResponse represents the response from vector store search
type VectorStoreSearchResponse struct {
	Object string                    `json:"object"`
	Data   []VectorStoreSearchResult `json:"data"`
}

// SearchVectorStore performs a search on a vector store
// This is the standalone search API (not through file_search tool)
func (c *VectorStoreClient) SearchVectorStore(ctx context.Context, vectorStoreID string, query string, limit int, filter map[string]interface{}) (*VectorStoreSearchResponse, error) {
	// Build request body
	reqBody := map[string]interface{}{
		"query": query,
	}
	if limit > 0 {
		reqBody["limit"] = limit
	}
	if filter != nil {
		reqBody["filter"] = filter
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", fmt.Sprintf("%s/v1/vector_stores/%s/search", c.baseURL, vectorStoreID), bytes.NewBuffer(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to search vector store: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return nil, fmt.Errorf("vector store search failed with status %d: %s", resp.StatusCode, string(body))
	}

	var searchResp VectorStoreSearchResponse
	if err := json.NewDecoder(resp.Body).Decode(&searchResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &searchResp, nil
}
