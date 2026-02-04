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
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// BaseLooper is a basic implementation that calls models sequentially
// and aggregates their responses. This is the POC implementation.
type BaseLooper struct {
	client *Client
	cfg    *config.LooperConfig
}

// NewBaseLooper creates a new BaseLooper instance
func NewBaseLooper(cfg *config.LooperConfig) *BaseLooper {
	return &BaseLooper{
		client: NewClient(cfg),
		cfg:    cfg,
	}
}

// Execute calls all models sequentially and aggregates the responses
func (l *BaseLooper) Execute(ctx context.Context, req *Request) (*Response, error) {
	if len(req.ModelRefs) == 0 {
		return nil, fmt.Errorf("no models configured")
	}

	// Set decision name in client for header transmission
	l.client.SetDecisionName(req.DecisionName)

	logging.Infof("[BaseLooper] Starting execution with %d models, streaming=%v",
		len(req.ModelRefs), req.IsStreaming)

	var responses []*ModelResponse
	var modelsUsed []string
	iteration := 0

	// Call each model sequentially
	for _, modelRef := range req.ModelRefs {
		iteration++
		modelName := modelRef.Model
		if modelRef.LoRAName != "" {
			modelName = modelRef.LoRAName
		}

		// Get access key from model params
		accessKey := ""
		if req.ModelParams != nil {
			if params, ok := req.ModelParams[modelRef.Model]; ok {
				accessKey = params.AccessKey
			}
		}

		logging.Infof("[BaseLooper] Calling model: %s (iteration=%d)", modelName, iteration)

		// BaseLooper doesn't need logprobs (no confidence-based routing)
		resp, err := l.client.CallModel(ctx, req.OriginalRequest, modelName, false, iteration, nil, accessKey)
		if err != nil {
			logging.Errorf("[BaseLooper] Model %s failed: %v", modelName, err)
			continue
		}

		responses = append(responses, resp)
		modelsUsed = append(modelsUsed, modelName)
	}

	if len(responses) == 0 {
		return nil, fmt.Errorf("all models failed")
	}

	// Aggregate responses
	aggregated := l.aggregateResponses(responses, modelsUsed)

	// Format output based on streaming preference
	if req.IsStreaming {
		return l.formatStreamingResponse(aggregated, modelsUsed, iteration)
	}
	return l.formatJSONResponse(aggregated, modelsUsed, iteration)
}

// aggregateResponses combines multiple model responses into one
// POC: Simply concatenates responses with model labels
func (l *BaseLooper) aggregateResponses(responses []*ModelResponse, models []string) *AggregatedResponse {
	result := &AggregatedResponse{
		Models:     models,
		Responses:  responses,
		FinalModel: models[len(models)-1],
	}

	// Simple aggregation: concatenate all responses
	var combinedContent string
	for i, resp := range responses {
		if i > 0 {
			combinedContent += "\n\n---\n\n"
		}
		combinedContent += fmt.Sprintf("**[%s]:**\n%s", models[i], resp.Content)
	}
	result.CombinedContent = combinedContent

	// Use the last response's logprobs for confidence
	if len(responses) > 0 {
		lastResp := responses[len(responses)-1]
		result.AverageLogprob = lastResp.AverageLogprob
	}

	logging.Infof("[BaseLooper] Aggregated %d responses, total content length=%d",
		len(responses), len(combinedContent))

	return result
}

// AggregatedResponse holds the combined result from multiple models
type AggregatedResponse struct {
	Models          []string
	Responses       []*ModelResponse
	CombinedContent string
	FinalModel      string
	AverageLogprob  float64
}

// formatJSONResponse creates a JSON ChatCompletion response
func (l *BaseLooper) formatJSONResponse(agg *AggregatedResponse, modelsUsed []string, iterations int) (*Response, error) {
	completion := map[string]interface{}{
		"id":      fmt.Sprintf("chatcmpl-looper-%d", time.Now().UnixNano()),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   agg.FinalModel,
		"choices": []map[string]interface{}{
			{
				"index": 0,
				"message": map[string]interface{}{
					"role":    "assistant",
					"content": agg.CombinedContent,
				},
				"finish_reason": "stop",
			},
		},
		"usage": map[string]interface{}{
			"prompt_tokens":     0,
			"completion_tokens": 0,
			"total_tokens":      0,
		},
	}

	body, err := json.Marshal(completion)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal response: %w", err)
	}

	return &Response{
		Body:          body,
		ContentType:   "application/json",
		Model:         agg.FinalModel,
		ModelsUsed:    modelsUsed,
		Iterations:    iterations,
		AlgorithmType: "simple",
	}, nil
}

// formatStreamingResponse creates an SSE streaming response
func (l *BaseLooper) formatStreamingResponse(agg *AggregatedResponse, modelsUsed []string, iterations int) (*Response, error) {
	timestamp := time.Now().Unix()
	id := fmt.Sprintf("chatcmpl-looper-%d", timestamp)

	// Split content into chunks for streaming effect
	chunks := splitIntoChunks(agg.CombinedContent, 50) // ~50 chars per chunk

	var sseBody []byte

	// First chunk with role
	firstChunk := map[string]interface{}{
		"id":      id,
		"object":  "chat.completion.chunk",
		"created": timestamp,
		"model":   agg.FinalModel,
		"choices": []map[string]interface{}{
			{
				"index": 0,
				"delta": map[string]interface{}{
					"role": "assistant",
				},
				"finish_reason": nil,
			},
		},
	}
	firstChunkJSON, _ := json.Marshal(firstChunk)
	sseBody = append(sseBody, []byte(fmt.Sprintf("data: %s\n\n", firstChunkJSON))...)

	// Content chunks
	for _, chunk := range chunks {
		contentChunk := map[string]interface{}{
			"id":      id,
			"object":  "chat.completion.chunk",
			"created": timestamp,
			"model":   agg.FinalModel,
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"delta": map[string]interface{}{
						"content": chunk,
					},
					"finish_reason": nil,
				},
			},
		}
		chunkJSON, _ := json.Marshal(contentChunk)
		sseBody = append(sseBody, []byte(fmt.Sprintf("data: %s\n\n", chunkJSON))...)
	}

	// Final chunk with finish_reason
	finalChunk := map[string]interface{}{
		"id":      id,
		"object":  "chat.completion.chunk",
		"created": timestamp,
		"model":   agg.FinalModel,
		"choices": []map[string]interface{}{
			{
				"index":         0,
				"delta":         map[string]interface{}{},
				"finish_reason": "stop",
			},
		},
	}
	finalChunkJSON, _ := json.Marshal(finalChunk)
	sseBody = append(sseBody, []byte(fmt.Sprintf("data: %s\n\n", finalChunkJSON))...)

	// [DONE] marker
	sseBody = append(sseBody, []byte("data: [DONE]\n\n")...)

	return &Response{
		Body:          sseBody,
		ContentType:   "text/event-stream",
		Model:         agg.FinalModel,
		ModelsUsed:    modelsUsed,
		Iterations:    iterations,
		AlgorithmType: "simple",
	}, nil
}

// splitIntoChunks splits a string into chunks of approximately the given size
func splitIntoChunks(s string, chunkSize int) []string {
	if len(s) == 0 {
		return nil
	}

	var chunks []string
	runes := []rune(s)

	for i := 0; i < len(runes); i += chunkSize {
		end := i + chunkSize
		if end > len(runes) {
			end = len(runes)
		}
		chunks = append(chunks, string(runes[i:end]))
	}

	return chunks
}
