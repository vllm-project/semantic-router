package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// FileStoreClient handles interactions with OpenAI File Store API
type FileStoreClient struct {
	httpClient *http.Client
	baseURL    string
	apiKey     string
}

// NewFileStoreClient creates a new OpenAI File Store client
func NewFileStoreClient(baseURL string, apiKey string) *FileStoreClient {
	return &FileStoreClient{
		httpClient: &http.Client{
			Timeout: 300 * time.Second, // File uploads can take time
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

// File represents an OpenAI file
type File struct {
	ID            string `json:"id"`
	Object        string `json:"object"`
	Bytes         int64  `json:"bytes"`
	CreatedAt     int64  `json:"created_at"`
	Filename      string `json:"filename"`
	Purpose       string `json:"purpose"`
	Status        string `json:"status,omitempty"`
	StatusDetails string `json:"status_details,omitempty"`
}

// FileListResponse represents the response from listing files
type FileListResponse struct {
	Object  string `json:"object"`
	Data    []File `json:"data"`
	FirstID string `json:"first_id,omitempty"`
	LastID  string `json:"last_id,omitempty"`
	HasMore bool   `json:"has_more"`
}

// UploadFile uploads a file to OpenAI's file store
// purpose should be "assistants" for vector store usage
func (c *FileStoreClient) UploadFile(ctx context.Context, fileReader io.Reader, filename string, purpose string) (*File, error) {
	// Create multipart form
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	// Add purpose field
	if err := writer.WriteField("purpose", purpose); err != nil {
		return nil, fmt.Errorf("failed to write purpose field: %w", err)
	}

	// Add file field
	fileWriter, createErr := writer.CreateFormFile("file", filename)
	if createErr != nil {
		return nil, fmt.Errorf("failed to create form file: %w", createErr)
	}

	if _, copyErr := io.Copy(fileWriter, fileReader); copyErr != nil {
		return nil, fmt.Errorf("failed to copy file content: %w", copyErr)
	}

	if err := writer.Close(); err != nil {
		return nil, fmt.Errorf("failed to close multipart writer: %w", err)
	}

	// Create request
	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/v1/files", &buf)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", writer.FormDataContentType())
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	// Execute request
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to upload file: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return nil, fmt.Errorf("file upload failed with status %d: %s", resp.StatusCode, string(body))
	}

	var file File
	if err := json.NewDecoder(resp.Body).Decode(&file); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	logging.Infof("Uploaded file %s (ID: %s, size: %d bytes)", filename, file.ID, file.Bytes)
	return &file, nil
}

// ListFiles lists all files in the file store
func (c *FileStoreClient) ListFiles(ctx context.Context, purpose string) (*FileListResponse, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", c.baseURL+"/v1/files", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	// Add purpose query parameter if specified
	if purpose != "" {
		q := req.URL.Query()
		q.Set("purpose", purpose)
		req.URL.RawQuery = q.Encode()
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to list files: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return nil, fmt.Errorf("list files failed with status %d: %s", resp.StatusCode, string(body))
	}

	var fileList FileListResponse
	if err := json.NewDecoder(resp.Body).Decode(&fileList); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &fileList, nil
}

// GetFile retrieves information about a specific file
func (c *FileStoreClient) GetFile(ctx context.Context, fileID string) (*File, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", fmt.Sprintf("%s/v1/files/%s", c.baseURL, fileID), nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to get file: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return nil, fmt.Errorf("get file failed with status %d: %s", resp.StatusCode, string(body))
	}

	var file File
	if err := json.NewDecoder(resp.Body).Decode(&file); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &file, nil
}

// DeleteFile deletes a file from the file store
func (c *FileStoreClient) DeleteFile(ctx context.Context, fileID string) error {
	req, err := http.NewRequestWithContext(ctx, "DELETE", fmt.Sprintf("%s/v1/files/%s", c.baseURL, fileID), nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to delete file: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return fmt.Errorf("delete file failed with status %d: %s", resp.StatusCode, string(body))
	}

	logging.Infof("Deleted file %s", fileID)
	return nil
}

// DownloadFile downloads file content
func (c *FileStoreClient) DownloadFile(ctx context.Context, fileID string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", fmt.Sprintf("%s/v1/files/%s/content", c.baseURL, fileID), nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to download file: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return nil, fmt.Errorf("download file failed with status %d: %s", resp.StatusCode, string(body))
	}

	content, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read file content: %w", err)
	}

	return content, nil
}
