package ensemble

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

// Factory orchestrates ensemble requests across multiple model endpoints
type Factory struct {
	config     *Config
	httpClient *http.Client
	endpoints  map[string]string // model name -> endpoint URL mapping
	mu         sync.RWMutex
}

// NewFactory creates a new ensemble factory
func NewFactory(config *Config) *Factory {
	if config == nil {
		config = &Config{
			Enabled:               true,
			DefaultStrategy:       StrategyVoting,
			DefaultMinResponses:   2,
			TimeoutSeconds:        30,
			MaxConcurrentRequests: 10,
		}
	}

	timeout := time.Duration(config.TimeoutSeconds) * time.Second
	if timeout == 0 {
		timeout = 30 * time.Second
	}

	return &Factory{
		config: config,
		httpClient: &http.Client{
			Timeout: timeout,
		},
		endpoints: make(map[string]string),
	}
}

// RegisterEndpoint registers a model endpoint for ensemble queries
func (f *Factory) RegisterEndpoint(modelName, endpointURL string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.endpoints[modelName] = endpointURL
	logging.Infof("Registered ensemble endpoint: %s -> %s", modelName, endpointURL)
}

// GetDefaultStrategy returns the configured default strategy
func (f *Factory) GetDefaultStrategy() Strategy {
	return f.config.DefaultStrategy
}

// GetDefaultMinResponses returns the configured default minimum responses
func (f *Factory) GetDefaultMinResponses() int {
	return f.config.DefaultMinResponses
}

// Execute performs ensemble orchestration for the given request
func (f *Factory) Execute(req *Request) *Response {
	if !f.config.Enabled {
		return &Response{
			Error: fmt.Errorf("ensemble mode is not enabled"),
		}
	}

	if len(req.Models) == 0 {
		return &Response{
			Error: fmt.Errorf("no models specified for ensemble"),
		}
	}

	// Validate strategy
	if req.Strategy == "" {
		req.Strategy = f.config.DefaultStrategy
	}

	// Validate min responses
	if req.MinResponses == 0 {
		req.MinResponses = f.config.DefaultMinResponses
	}
	if req.MinResponses > len(req.Models) {
		req.MinResponses = len(req.Models)
	}

	// Perform parallel model queries
	startTime := time.Now()
	responses := f.queryModels(req)
	totalLatency := time.Since(startTime).Milliseconds()

	// Filter successful responses
	successfulResponses := make([]ModelResponse, 0, len(responses))
	for _, resp := range responses {
		if resp.Error == nil {
			successfulResponses = append(successfulResponses, resp)
		}
	}

	// Check if we have enough responses
	if len(successfulResponses) < req.MinResponses {
		return &Response{
			ModelsQueried:     len(req.Models),
			ResponsesReceived: len(successfulResponses),
			Strategy:          req.Strategy,
			Error: fmt.Errorf("insufficient responses: got %d, required %d",
				len(successfulResponses), req.MinResponses),
		}
	}

	// Aggregate responses based on strategy
	finalResponse, metadata, err := f.aggregateResponses(successfulResponses, req.Strategy)
	if err != nil {
		return &Response{
			ModelsQueried:     len(req.Models),
			ResponsesReceived: len(successfulResponses),
			Strategy:          req.Strategy,
			Error:             fmt.Errorf("aggregation failed: %w", err),
		}
	}

	// Build metadata (only include successful responses)
	metadata.TotalLatencyMs = totalLatency
	metadata.ModelLatenciesMs = make(map[string]int64)
	metadata.ConfidenceScores = make(map[string]float64)
	for _, resp := range successfulResponses {
		metadata.ModelLatenciesMs[resp.ModelName] = resp.Latency.Milliseconds()
		if resp.Confidence > 0 {
			metadata.ConfidenceScores[resp.ModelName] = resp.Confidence
		}
	}

	return &Response{
		FinalResponse:     finalResponse,
		ModelsQueried:     len(req.Models),
		ResponsesReceived: len(successfulResponses),
		Strategy:          req.Strategy,
		Metadata:          metadata,
	}
}

// queryModels queries all models in parallel
func (f *Factory) queryModels(req *Request) []ModelResponse {
	f.mu.RLock()
	defer f.mu.RUnlock()

	responses := make([]ModelResponse, len(req.Models))
	var wg sync.WaitGroup

	// Limit concurrent requests (ensure at least 1)
	maxConcurrent := f.config.MaxConcurrentRequests
	if maxConcurrent <= 0 {
		maxConcurrent = 10 // Default to 10 if not set or invalid
	}
	semaphore := make(chan struct{}, maxConcurrent)

	for i, modelName := range req.Models {
		wg.Add(1)
		go func(idx int, model string) {
			defer wg.Done()

			// Acquire semaphore
			semaphore <- struct{}{}
			defer func() { <-semaphore }()

			responses[idx] = f.queryModel(req.Context, model, req.OriginalRequest, req.Headers)
		}(i, modelName)
	}

	wg.Wait()
	return responses
}

// queryModel queries a single model endpoint
func (f *Factory) queryModel(ctx context.Context, modelName string, requestBody []byte, headers map[string]string) ModelResponse {
	startTime := time.Now()
	
	endpoint, ok := f.endpoints[modelName]
	if !ok {
		return ModelResponse{
			ModelName: modelName,
			Error:     fmt.Errorf("endpoint not found for model: %s", modelName),
			Latency:   time.Since(startTime),
		}
	}

	// Update the model field in the request body
	modifiedRequest, err := f.updateModelInRequest(requestBody, modelName)
	if err != nil {
		return ModelResponse{
			ModelName: modelName,
			Error:     fmt.Errorf("failed to update model in request: %w", err),
			Latency:   time.Since(startTime),
		}
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", endpoint, bytes.NewReader(modifiedRequest))
	if err != nil {
		return ModelResponse{
			ModelName: modelName,
			Error:     fmt.Errorf("failed to create HTTP request: %w", err),
			Latency:   time.Since(startTime),
		}
	}

	httpReq.Header.Set("Content-Type", "application/json")

	// Forward authentication and other headers from original request
	for key, value := range headers {
		// Forward authorization and other important headers
		lowerKey := strings.ToLower(key)
		if lowerKey == "authorization" || lowerKey == "x-api-key" || strings.HasPrefix(lowerKey, "x-") {
			httpReq.Header.Set(key, value)
		}
	}

	// Execute request
	resp, err := f.httpClient.Do(httpReq)
	if err != nil {
		return ModelResponse{
			ModelName: modelName,
			Error:     fmt.Errorf("HTTP request failed: %w", err),
			Latency:   time.Since(startTime),
		}
	}
	defer resp.Body.Close()

	// Read response body
	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return ModelResponse{
			ModelName: modelName,
			Error:     fmt.Errorf("failed to read response body: %w", err),
			Latency:   time.Since(startTime),
		}
	}

	// Check status code
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return ModelResponse{
			ModelName: modelName,
			Error:     fmt.Errorf("HTTP error %d: %s", resp.StatusCode, string(responseBody)),
			Latency:   time.Since(startTime),
		}
	}

	return ModelResponse{
		ModelName: modelName,
		Response:  responseBody,
		Latency:   time.Since(startTime),
	}
}

// updateModelInRequest updates the model field in the OpenAI request
func (f *Factory) updateModelInRequest(requestBody []byte, modelName string) ([]byte, error) {
	var request map[string]interface{}
	if err := json.Unmarshal(requestBody, &request); err != nil {
		return nil, fmt.Errorf("failed to parse request JSON: %w", err)
	}

	request["model"] = modelName

	modifiedRequest, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal modified request: %w", err)
	}

	return modifiedRequest, nil
}

// aggregateResponses aggregates model responses based on the strategy
func (f *Factory) aggregateResponses(responses []ModelResponse, strategy Strategy) ([]byte, Metadata, error) {
	metadata := Metadata{
		AggregationDetails: make(map[string]interface{}),
	}

	switch strategy {
	case StrategyFirstSuccess:
		// Return the first successful response (fastest)
		if len(responses) > 0 {
			metadata.SelectedModel = responses[0].ModelName
			return responses[0].Response, metadata, nil
		}
		return nil, metadata, fmt.Errorf("no successful responses")

	case StrategyVoting:
		// For voting, we need to extract and compare responses
		// This is simplified - in production, you'd parse the actual choices
		return f.aggregateByVoting(responses, &metadata)

	case StrategyWeighted:
		// Use confidence-weighted selection
		return f.aggregateByWeighted(responses, &metadata)

	case StrategyScoreAveraging:
		// Average numerical scores (simplified)
		return f.aggregateByScoreAveraging(responses, &metadata)

	default:
		// Default to first success
		if len(responses) > 0 {
			metadata.SelectedModel = responses[0].ModelName
			return responses[0].Response, metadata, nil
		}
		return nil, metadata, fmt.Errorf("no successful responses")
	}
}

// aggregateByVoting implements majority voting by comparing message content
func (f *Factory) aggregateByVoting(responses []ModelResponse, metadata *Metadata) ([]byte, Metadata, error) {
	// Parse responses and extract message content for voting
	type parsedResponse struct {
		content  string
		rawBytes []byte
	}
	
	contentCounts := make(map[string]int)
	contentToResponse := make(map[string]parsedResponse)

	for _, resp := range responses {
		// Try to parse OpenAI-style response
		var openAIResp map[string]interface{}
		if err := json.Unmarshal(resp.Response, &openAIResp); err != nil {
			// If parsing fails, use first response as fallback
			logging.Warnf("Failed to parse response for voting: %v", err)
			continue
		}

		// Extract content from choices array
		content := extractContentFromResponse(openAIResp)
		if content != "" {
			contentCounts[content]++
			contentToResponse[content] = parsedResponse{
				content:  content,
				rawBytes: resp.Response,
			}
		}
	}

	// Find the most common content
	var maxCount int
	var selectedContent string
	for content, count := range contentCounts {
		if count > maxCount {
			maxCount = count
			selectedContent = content
		}
	}

	metadata.AggregationDetails["vote_counts"] = contentCounts
	metadata.AggregationDetails["max_votes"] = maxCount

	// Return the response with the most votes, or first response if no clear winner
	if selectedContent != "" {
		if selected, ok := contentToResponse[selectedContent]; ok {
			return selected.rawBytes, *metadata, nil
		}
	}

	return responses[0].Response, *metadata, nil
}

// extractContentFromResponse extracts the message content from an OpenAI-style response
func extractContentFromResponse(resp map[string]interface{}) string {
	// Navigate: response["choices"][0]["message"]["content"]
	if choices, ok := resp["choices"].([]interface{}); ok && len(choices) > 0 {
		if choice, ok := choices[0].(map[string]interface{}); ok {
			if message, ok := choice["message"].(map[string]interface{}); ok {
				if content, ok := message["content"].(string); ok {
					return content
				}
			}
		}
	}
	return ""
}

// aggregateByWeighted implements confidence-weighted selection
func (f *Factory) aggregateByWeighted(responses []ModelResponse, metadata *Metadata) ([]byte, Metadata, error) {
	// Select response with highest confidence
	var maxConfidence float64
	var selectedResponse []byte
	var selectedModel string

	for _, resp := range responses {
		if resp.Confidence > maxConfidence {
			maxConfidence = resp.Confidence
			selectedResponse = resp.Response
			selectedModel = resp.ModelName
		}
	}

	// If no confidence scores, fall back to first response
	if selectedResponse == nil {
		selectedResponse = responses[0].Response
		selectedModel = responses[0].ModelName
	}

	metadata.SelectedModel = selectedModel
	metadata.AggregationDetails["max_confidence"] = maxConfidence

	return selectedResponse, *metadata, nil
}

// aggregateByScoreAveraging averages logprobs or confidence scores from multiple models
func (f *Factory) aggregateByScoreAveraging(responses []ModelResponse, metadata *Metadata) ([]byte, Metadata, error) {
	// For score averaging, we select the response with the median confidence/latency balance
	// This is more practical than trying to merge responses
	
	type scoredResponse struct {
		response ModelResponse
		score    float64
	}
	
	scored := make([]scoredResponse, 0, len(responses))
	
	for _, resp := range responses {
		// Compute a composite score based on confidence and latency
		// Higher confidence is better, lower latency is better
		score := resp.Confidence
		if resp.Latency.Seconds() > 0 {
			// Normalize latency (penalize slow responses)
			latencyPenalty := 1.0 / (1.0 + resp.Latency.Seconds())
			score = score * latencyPenalty
		}
		
		scored = append(scored, scoredResponse{
			response: resp,
			score:    score,
		})
	}
	
	// If no confidence scores available, fall back to selecting by fastest response
	allZeroConfidence := true
	for _, s := range scored {
		if s.score > 0 {
			allZeroConfidence = false
			break
		}
	}
	
	if allZeroConfidence {
		// Select fastest response
		fastest := scored[0]
		for _, s := range scored[1:] {
			if s.response.Latency < fastest.response.Latency {
				fastest = s
			}
		}
		metadata.SelectedModel = fastest.response.ModelName
		metadata.AggregationDetails["selection_method"] = "fastest_response"
		return fastest.response.Response, *metadata, nil
	}
	
	// Find highest scoring response
	best := scored[0]
	for _, s := range scored[1:] {
		if s.score > best.score {
			best = s
		}
	}
	
	metadata.SelectedModel = best.response.ModelName
	metadata.AggregationDetails["best_score"] = best.score
	metadata.AggregationDetails["selection_method"] = "score_based"
	
	return best.response.Response, *metadata, nil
}
