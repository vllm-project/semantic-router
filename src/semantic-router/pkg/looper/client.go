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

package looper

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// Client handles HTTP requests to OpenAI-compatible endpoints
type Client struct {
	httpClient   *http.Client
	endpoint     string
	headers      map[string]string
	decisionName string // Decision name to pass in looper requests
}

// NewClient creates a new looper HTTP client
func NewClient(cfg *config.LooperConfig) *Client {
	return &Client{
		httpClient: &http.Client{
			Timeout: time.Duration(cfg.GetTimeout()) * time.Second,
		},
		endpoint: cfg.Endpoint,
		headers:  cfg.Headers,
	}
}

// SetDecisionName sets the decision name for this client
func (c *Client) SetDecisionName(name string) {
	c.decisionName = name
}

// ModelResponse contains the parsed response from a model call
type ModelResponse struct {
	// Raw is the raw response body
	Raw []byte

	// Parsed is the parsed ChatCompletion (nil for streaming responses)
	Parsed *openai.ChatCompletion

	// Content is the extracted text content from the response
	Content string

	// ReasoningContent is the extracted reasoning/thinking content from vLLM models
	// This field is populated when vLLM returns reasoning in extra response fields
	// (e.g., reasoning_content, reasoning)
	ReasoningContent string

	// Model is the model name from the response
	Model string

	// Logprobs contains token logprobs if available
	Logprobs []float64

	// AverageLogprob is the average logprob across all tokens (for confidence assessment)
	// Range: negative values, closer to 0 = more confident
	AverageLogprob float64

	// TopLogprobMargins contains the margin (top1 - top2) for each token position
	// Higher margin = model is more certain about the chosen token
	TopLogprobMargins []float64

	// AverageMargin is the average margin across all tokens
	// Range: positive values, higher = more confident
	AverageMargin float64

	// IsStreaming indicates if this was a streaming response
	IsStreaming bool

	// StreamingChunks contains the raw SSE chunks for streaming responses
	StreamingChunks []string
}

// LogprobsConfig controls logprobs behavior for model calls
type LogprobsConfig struct {
	Enabled     bool // Whether to request logprobs from the model
	TopLogprobs int  // Number of top logprobs to return (0-5, default 1 for margin calculation)
}

// CallModel sends a request to the configured endpoint with a specific model
// Parameters:
//   - iteration: 1-based iteration number for tracking
//   - logprobsCfg: controls whether to enable logprobs and top_logprobs (nil = disabled)
//   - accessKey: optional API key for Authorization header (Bearer token)
func (c *Client) CallModel(ctx context.Context, req *openai.ChatCompletionNewParams, modelName string, streaming bool, iteration int, logprobsCfg *LogprobsConfig, accessKey string) (*ModelResponse, error) {
	// Clone and modify the request with the target model
	modifiedReq := cloneRequest(req)
	modifiedReq.Model = modelName

	// Configure logprobs based on config
	if logprobsCfg != nil && logprobsCfg.Enabled {
		modifiedReq.Logprobs = openai.Bool(true)
		topLogprobs := logprobsCfg.TopLogprobs
		if topLogprobs < 1 {
			topLogprobs = 1 // Need at least 1 for margin calculation
		}
		if topLogprobs > 5 {
			topLogprobs = 5 // API limit
		}
		modifiedReq.TopLogprobs = openai.Int(int64(topLogprobs))
	}

	// Marshal request to JSON first
	body, err := json.Marshal(modifiedReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Add stream parameter via JSON manipulation (SDK doesn't expose Stream field)
	body, err = setStreamParam(body, streaming)
	if err != nil {
		return nil, fmt.Errorf("failed to set stream param: %w", err)
	}

	logprobsEnabled := logprobsCfg != nil && logprobsCfg.Enabled
	logging.Infof("[Looper] Calling model %s at %s (streaming=%v, iteration=%d, logprobs=%v)",
		modelName, c.endpoint, streaming, iteration, logprobsEnabled)

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	httpReq.Header.Set("Content-Type", "application/json")
	for k, v := range c.headers {
		httpReq.Header.Set(k, v)
	}

	// Set Authorization header if access key is provided
	if accessKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+accessKey)
	}

	// Add looper identification headers
	// These allow extproc to identify looper requests and lookup decision configuration
	httpReq.Header.Set("x-vsr-looper-request", "true")
	httpReq.Header.Set("x-vsr-looper-iteration", fmt.Sprintf("%d", iteration))

	// Add decision name header for extproc to lookup decision configuration
	if c.decisionName != "" {
		httpReq.Header.Set("x-vsr-looper-decision", c.decisionName)
	}

	// Execute request
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	// Read response
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("request failed with status %d: %s", resp.StatusCode, string(respBody))
	}

	// Parse response based on streaming mode
	if streaming {
		return c.parseStreamingResponse(respBody, modelName)
	}
	return c.parseNonStreamingResponse(respBody, modelName)
}

// parseNonStreamingResponse parses a non-streaming JSON response
func (c *Client) parseNonStreamingResponse(body []byte, modelName string) (*ModelResponse, error) {
	var completion openai.ChatCompletion
	if err := json.Unmarshal(body, &completion); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	result := &ModelResponse{
		Raw:         body,
		Parsed:      &completion,
		Model:       modelName, // Use the requested model name, not the backend's response
		IsStreaming: false,
	}

	// Extract content and logprobs
	if len(completion.Choices) > 0 {
		result.Content = completion.Choices[0].Message.Content

		// Extract logprobs and margin analysis
		analysis := extractLogprobs(&completion)
		result.Logprobs = analysis.Logprobs
		result.AverageLogprob = analysis.AverageLogprob
		result.TopLogprobMargins = analysis.Margins
		result.AverageMargin = analysis.AverageMargin
	}

	// Extract reasoning content from vLLM extra fields
	result.ReasoningContent = extractReasoningFromRaw(body)

	logging.Infof("[Looper] Model %s responded: content_len=%d, reasoning_len=%d, avg_logprob=%.4f, avg_margin=%.4f",
		modelName, len(result.Content), len(result.ReasoningContent), result.AverageLogprob, result.AverageMargin)

	return result, nil
}

// parseStreamingResponse parses SSE streaming response
func (c *Client) parseStreamingResponse(body []byte, modelName string) (*ModelResponse, error) {
	result := &ModelResponse{
		Raw:         body,
		Model:       modelName,
		IsStreaming: true,
	}

	// Parse SSE chunks to extract content
	content, chunks := parseSSEContent(body)
	result.Content = content
	result.StreamingChunks = chunks

	logging.Infof("[Looper] Model %s streaming response with content length=%d", modelName, len(content))

	return result, nil
}

// parseSSEContent extracts content from SSE formatted response
func parseSSEContent(body []byte) (string, []string) {
	var content string
	var chunks []string

	lines := bytes.Split(body, []byte("\n"))
	for _, line := range lines {
		lineStr := string(line)
		if len(lineStr) > 6 && lineStr[:6] == "data: " {
			data := lineStr[6:]
			chunks = append(chunks, data)

			if data == "[DONE]" {
				continue
			}

			var chunk map[string]interface{}
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				continue
			}

			if choices, ok := chunk["choices"].([]interface{}); ok && len(choices) > 0 {
				if choice, ok := choices[0].(map[string]interface{}); ok {
					if delta, ok := choice["delta"].(map[string]interface{}); ok {
						if c, ok := delta["content"].(string); ok {
							content += c
						}
					}
				}
			}
		}
	}

	return content, chunks
}

// LogprobAnalysis contains analyzed logprob data from a response
type LogprobAnalysis struct {
	// Logprobs contains the logprob for each token (the chosen token's logprob)
	Logprobs []float64
	// AverageLogprob is the average logprob across all tokens
	// Range: negative, closer to 0 = more confident
	AverageLogprob float64

	// Margins contains the margin (top1 - top2) for each token position
	// Higher margin = model was more certain about the chosen token
	Margins []float64
	// AverageMargin is the average margin across all tokens
	// Range: positive, higher = more confident
	AverageMargin float64
}

// extractLogprobs extracts logprobs and margins from a ChatCompletion response
// Returns both the raw logprobs and the margin analysis for confidence evaluation
func extractLogprobs(completion *openai.ChatCompletion) *LogprobAnalysis {
	result := &LogprobAnalysis{}

	if len(completion.Choices) == 0 {
		return result
	}

	choice := completion.Choices[0]
	// Check if Logprobs content is empty (the struct is not a pointer in openai-go)
	if len(choice.Logprobs.Content) == 0 {
		return result
	}

	var logprobSum float64
	var marginSum float64

	for _, tokenLogprob := range choice.Logprobs.Content {
		// Extract the chosen token's logprob
		result.Logprobs = append(result.Logprobs, tokenLogprob.Logprob)
		logprobSum += tokenLogprob.Logprob

		// Calculate margin from top_logprobs (top1 - top2)
		// TopLogprobs contains alternative tokens with their logprobs
		margin := calculateMargin(tokenLogprob.Logprob, tokenLogprob.TopLogprobs)
		result.Margins = append(result.Margins, margin)
		marginSum += margin
	}

	// Calculate averages
	if len(result.Logprobs) > 0 {
		result.AverageLogprob = logprobSum / float64(len(result.Logprobs))
	}
	if len(result.Margins) > 0 {
		result.AverageMargin = marginSum / float64(len(result.Margins))
	}

	return result
}

// calculateMargin calculates the margin between the chosen token and the next best alternative
// A large margin indicates the model was very confident in its choice
// A small margin indicates the model was uncertain between multiple options
func calculateMargin(chosenLogprob float64, topLogprobs []openai.ChatCompletionTokenLogprobTopLogprob) float64 {
	if len(topLogprobs) < 2 {
		// No alternative token available, assume high confidence
		// Use a reasonable default margin
		return 2.0
	}

	// topLogprobs[0] is the chosen token (should match chosenLogprob)
	// topLogprobs[1] is the second-best alternative
	// Margin = logprob(top1) - logprob(top2)
	// Since logprobs are negative, a positive margin means top1 > top2 in probability
	top1 := topLogprobs[0].Logprob
	top2 := topLogprobs[1].Logprob

	// Margin: how much better is top1 than top2
	// Example: top1=-0.1, top2=-2.0 => margin=1.9 (high confidence)
	// Example: top1=-0.5, top2=-0.6 => margin=0.1 (low confidence, model is uncertain)
	return top1 - top2
}

// setStreamParam adds or updates the stream parameter in a JSON request body
func setStreamParam(body []byte, streaming bool) ([]byte, error) {
	var reqMap map[string]interface{}
	if err := json.Unmarshal(body, &reqMap); err != nil {
		return nil, err
	}
	reqMap["stream"] = streaming
	return json.Marshal(reqMap)
}

// extractReasoningFromRaw extracts reasoning content from vLLM response
// vLLM returns reasoning in extra response fields (not tags), which can be in multiple locations:
// - choices[0].reasoning
// - choices[0].reasoning_content
// - choices[0].message.reasoning
// - choices[0].message.reasoning_content
func extractReasoningFromRaw(rawBody []byte) string {
	var raw map[string]interface{}
	if err := json.Unmarshal(rawBody, &raw); err != nil {
		return ""
	}

	choices, ok := raw["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return ""
	}

	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		return ""
	}

	// Try choice-level fields first
	if reasoning, reasoningOk := choice["reasoning"].(string); reasoningOk {
		return reasoning
	}
	if reasoning, reasoningOk := choice["reasoning_content"].(string); reasoningOk {
		return reasoning
	}

	// Try message-level fields
	message, ok := choice["message"].(map[string]interface{})
	if !ok {
		return ""
	}

	if reasoning, ok := message["reasoning"].(string); ok {
		return reasoning
	}
	if reasoning, ok := message["reasoning_content"].(string); ok {
		return reasoning
	}

	return ""
}

// cloneRequest creates a shallow copy of the request
func cloneRequest(req *openai.ChatCompletionNewParams) *openai.ChatCompletionNewParams {
	// Create a new params with the same values
	clone := *req
	return &clone
}
