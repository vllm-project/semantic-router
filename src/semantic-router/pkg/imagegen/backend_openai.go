package imagegen

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// OpenAIBackend implements the Backend interface for OpenAI image generation
type OpenAIBackend struct {
	baseURL    string
	apiKey     string
	model      string
	quality    string
	style      string
	httpClient *http.Client
}

// NewOpenAIBackend creates a new OpenAI image generation backend
func NewOpenAIBackend(cfg *config.ImageGenPluginConfig) (Backend, error) {
	openaiConfig, ok := cfg.BackendConfig.(*config.OpenAIImageGenConfig)
	if !ok {
		return nil, fmt.Errorf("invalid backend_config for openai, expected OpenAIImageGenConfig")
	}

	if openaiConfig.APIKey == "" {
		return nil, fmt.Errorf("api_key is required for openai backend")
	}

	baseURL := openaiConfig.BaseURL
	if baseURL == "" {
		baseURL = "https://api.openai.com"
	}

	timeout := time.Duration(cfg.TimeoutSeconds) * time.Second
	if timeout == 0 {
		timeout = 60 * time.Second
	}

	model := openaiConfig.Model
	if model == "" {
		model = "gpt-image-1"
	}

	return &OpenAIBackend{
		baseURL: baseURL,
		apiKey:  openaiConfig.APIKey,
		model:   model,
		quality: openaiConfig.Quality,
		style:   openaiConfig.Style,
		httpClient: &http.Client{
			Timeout: timeout,
		},
	}, nil
}

// Name returns the backend name
func (b *OpenAIBackend) Name() string {
	return "openai"
}

// GenerateImage generates an image using OpenAI's image generation API
func (b *OpenAIBackend) GenerateImage(ctx context.Context, req *GenerateRequest) (*GenerateResponse, error) {
	// Build OpenAI request
	openaiReq := openAIImageRequest{
		Model:  b.model,
		Prompt: req.Prompt,
		N:      1,
		Size:   fmt.Sprintf("%dx%d", getIntOrDefault(req.Width, 1024), getIntOrDefault(req.Height, 1024)),
	}

	if req.Model != "" {
		openaiReq.Model = req.Model
	}

	if req.Quality != "" {
		openaiReq.Quality = req.Quality
	} else if b.quality != "" {
		openaiReq.Quality = b.quality
	}

	if req.Style != "" {
		openaiReq.Style = req.Style
	} else if b.style != "" {
		openaiReq.Style = b.style
	}

	// Serialize request
	body, err := json.Marshal(openaiReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", b.baseURL+"/v1/images/generations", bytes.NewBuffer(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+b.apiKey)

	// Execute request
	resp, err := b.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	// Read response body
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	// Check for errors
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("request failed with status %d: %s", resp.StatusCode, string(respBody))
	}

	// Parse response
	var openaiResp openAIImageResponse
	if err := json.Unmarshal(respBody, &openaiResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	if len(openaiResp.Data) == 0 {
		return nil, fmt.Errorf("no images in response")
	}

	imageData := openaiResp.Data[0]
	result := &GenerateResponse{
		Model:         openaiReq.Model,
		Backend:       b.Name(),
		RevisedPrompt: imageData.RevisedPrompt,
	}

	// OpenAI returns either URL or base64
	if imageData.URL != "" {
		result.ImageURL = imageData.URL
	} else if imageData.B64JSON != "" {
		result.ImageBase64 = imageData.B64JSON
		result.ImageURL = "data:image/png;base64," + imageData.B64JSON
	}

	return result, nil
}

// HealthCheck checks if the OpenAI API is accessible
func (b *OpenAIBackend) HealthCheck(ctx context.Context) error {
	// OpenAI doesn't have a dedicated health endpoint
	// We can try to list models as a health check
	httpReq, err := http.NewRequestWithContext(ctx, "GET", b.baseURL+"/v1/models", nil)
	if err != nil {
		return fmt.Errorf("failed to create health check request: %w", err)
	}
	httpReq.Header.Set("Authorization", "Bearer "+b.apiKey)

	resp, err := b.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("health check request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check failed with status %d", resp.StatusCode)
	}

	return nil
}

// OpenAI API types
type openAIImageRequest struct {
	Model          string `json:"model"`
	Prompt         string `json:"prompt"`
	N              int    `json:"n,omitempty"`
	Size           string `json:"size,omitempty"`
	Quality        string `json:"quality,omitempty"`
	Style          string `json:"style,omitempty"`
	ResponseFormat string `json:"response_format,omitempty"` // "url" or "b64_json"
}

type openAIImageResponse struct {
	Created int64             `json:"created"`
	Data    []openAIImageData `json:"data"`
}

type openAIImageData struct {
	URL           string `json:"url,omitempty"`
	B64JSON       string `json:"b64_json,omitempty"`
	RevisedPrompt string `json:"revised_prompt,omitempty"`
}
