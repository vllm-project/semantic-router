package classification

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// VLLMClient handles communication with vLLM REST API for classifiers
type VLLMClient struct {
	httpClient *http.Client
	endpoint   *config.ClassifierVLLMEndpoint
	baseURL    string
	accessKey  string // Optional access key for Authorization header
}

// NewVLLMClient creates a new vLLM REST API client for classifiers
func NewVLLMClient(endpoint *config.ClassifierVLLMEndpoint) *VLLMClient {
	baseURL := fmt.Sprintf("http://%s:%d", endpoint.Address, endpoint.Port)

	return &VLLMClient{
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
		endpoint: endpoint,
		baseURL:  baseURL,
	}
}

// NewVLLMClientWithAuth creates a new vLLM REST API client with access key
func NewVLLMClientWithAuth(endpoint *config.ClassifierVLLMEndpoint, accessKey string) *VLLMClient {
	client := NewVLLMClient(endpoint)
	client.accessKey = accessKey
	return client
}

// ChatCompletionRequest represents OpenAI-compatible chat completion request
type ChatCompletionRequest struct {
	Model       string                 `json:"model"`
	Messages    []ChatMessage          `json:"messages"`
	MaxTokens   int                    `json:"max_tokens,omitempty"`
	Temperature float64                `json:"temperature,omitempty"`
	Stream      bool                   `json:"stream,omitempty"`
	ExtraBody   map[string]interface{} `json:"extra_body,omitempty"`
}

// ChatMessage represents a chat message in the request
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatCompletionResponse represents OpenAI-compatible chat completion response
type ChatCompletionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   *Usage   `json:"usage,omitempty"`
}

// Choice represents a choice in the chat completion response
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

// Message represents a message in the response
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Usage represents token usage information
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// GenerationOptions contains options for vLLM generation
type GenerationOptions struct {
	MaxTokens   int
	Temperature float64
	Stream      bool
	ExtraBody   map[string]interface{}
}

// Generate sends a chat completion request to vLLM
func (c *VLLMClient) Generate(ctx context.Context, modelName string, prompt string, options *GenerationOptions) (*ChatCompletionResponse, error) {
	// Build messages - use chat template if configured
	var messages []ChatMessage
	if c.endpoint.UseChatTemplate {
		// For models like Qwen3Guard that require chat template format
		messages = []ChatMessage{
			{Role: "system", Content: "You are a safety classifier."},
			{Role: "user", Content: prompt},
		}
	} else if c.endpoint.PromptTemplate != "" {
		// Use custom prompt template if provided
		formattedPrompt := fmt.Sprintf(c.endpoint.PromptTemplate, prompt)
		messages = []ChatMessage{{Role: "user", Content: formattedPrompt}}
	} else {
		// Default: simple user message
		messages = []ChatMessage{{Role: "user", Content: prompt}}
	}

	// Build request
	req := ChatCompletionRequest{
		Model:    modelName,
		Messages: messages,
	}

	if options != nil {
		req.MaxTokens = options.MaxTokens
		req.Temperature = options.Temperature
		req.Stream = options.Stream
		req.ExtraBody = options.ExtraBody
	}

	// Default values
	if req.MaxTokens == 0 {
		req.MaxTokens = 512
	}
	if req.Temperature == 0 {
		req.Temperature = 0.0 // Deterministic for safety checks
	}

	// Marshal request
	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	url := fmt.Sprintf("%s/v1/chat/completions", c.baseURL)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json")

	// Add Authorization header if access key is provided
	if c.accessKey != "" {
		httpReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.accessKey))
	}

	// Send request
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	// Read response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	// Check status code
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("vLLM API returned status %d: %s", resp.StatusCode, string(body))
	}

	// Parse response
	var chatResp ChatCompletionResponse
	if err := json.Unmarshal(body, &chatResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	logging.Debugf("vLLM API call successful: model=%s, choices=%d", modelName, len(chatResp.Choices))

	return &chatResp, nil
}
