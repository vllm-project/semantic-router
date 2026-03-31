package nlgen

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// OpenAIClient implements LLMClient for OpenAI-compatible endpoints (vLLM, OpenAI, etc.).
type OpenAIClient struct {
	baseURL string
	model   string
	apiKey  string
	client  *http.Client
}

// NewOpenAIClient creates a new OpenAI-compatible LLM client.
// baseURL should not include "/v1/chat/completions" -- it is appended automatically.
func NewOpenAIClient(baseURL, model, apiKey string) *OpenAIClient {
	return &OpenAIClient{
		baseURL: baseURL,
		model:   model,
		apiKey:  apiKey,
		client:  http.DefaultClient,
	}
}

func (c *OpenAIClient) ChatCompletion(ctx context.Context, req ChatCompletionRequest) (string, error) {
	type message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	type chatReq struct {
		Model       string    `json:"model"`
		Messages    []message `json:"messages"`
		Temperature float64   `json:"temperature"`
		MaxTokens   int       `json:"max_tokens"`
	}

	msgs := make([]message, len(req.Messages))
	for i, m := range req.Messages {
		msgs[i] = message{Role: m.Role, Content: m.Content}
	}

	body := chatReq{
		Model:       c.model,
		Messages:    msgs,
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
	}
	bodyJSON, err := json.Marshal(body)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/v1/chat/completions", bytes.NewReader(bodyJSON))
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := c.client.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("http request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	type choice struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	}
	type chatResp struct {
		Choices []choice `json:"choices"`
	}

	var result chatResp
	if err := json.Unmarshal(respBody, &result); err != nil {
		return "", fmt.Errorf("unmarshal response: %w", err)
	}
	if len(result.Choices) == 0 {
		return "", fmt.Errorf("no choices in response")
	}

	return result.Choices[0].Message.Content, nil
}
