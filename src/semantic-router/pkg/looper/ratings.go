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
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// RatingsLooper executes all models concurrently and returns multiple choices for comparison.
// Useful for arena-style ratings where you want responses from multiple models side by side.
type RatingsLooper struct {
	*BaseLooper
}

// NewRatingsLooper creates a new RatingsLooper instance
func NewRatingsLooper(cfg *config.LooperConfig) *RatingsLooper {
	return &RatingsLooper{
		BaseLooper: NewBaseLooper(cfg),
	}
}

// Execute calls all models concurrently and returns multiple choices
func (l *RatingsLooper) Execute(ctx context.Context, req *Request) (*Response, error) {
	if len(req.ModelRefs) == 0 {
		return nil, fmt.Errorf("no models configured")
	}

	// Set decision name in client for header transmission
	l.client.SetDecisionName(req.DecisionName)

	// Get config from algorithm
	maxConcurrent := len(req.ModelRefs)
	onError := "skip"
	if req.Algorithm != nil && req.Algorithm.Ratings != nil {
		if req.Algorithm.Ratings.MaxConcurrent > 0 {
			maxConcurrent = req.Algorithm.Ratings.MaxConcurrent
		}
		if req.Algorithm.Ratings.OnError != "" {
			onError = req.Algorithm.Ratings.OnError
		}
	}

	logging.Infof("[RatingsLooper] Starting with %d models, max_concurrent=%d, on_error=%s, streaming=%v",
		len(req.ModelRefs), maxConcurrent, onError, req.IsStreaming)

	// Use semaphore to limit concurrency
	sem := make(chan struct{}, maxConcurrent)
	var wg sync.WaitGroup
	var mu sync.Mutex

	responses := make([]*ModelResponse, len(req.ModelRefs))
	modelsUsed := make([]string, len(req.ModelRefs))
	errors := make([]error, len(req.ModelRefs))

	for i, modelRef := range req.ModelRefs {
		wg.Add(1)
		go func(idx int, ref config.ModelRef) {
			defer wg.Done()

			sem <- struct{}{}        // Acquire semaphore
			defer func() { <-sem }() // Release semaphore

			modelName := ref.Model
			if ref.LoRAName != "" {
				modelName = ref.LoRAName
			}

			// Get access key from model params
			accessKey := ""
			if req.ModelParams != nil {
				if params, ok := req.ModelParams[ref.Model]; ok {
					accessKey = params.AccessKey
				}
			}

			logging.Infof("[RatingsLooper] Calling model: %s (slot=%d)", modelName, idx+1)

			// Use idx+1 as iteration number for concurrent requests
			// RatingsLooper doesn't need logprobs (no confidence-based routing)
			resp, err := l.client.CallModel(ctx, req.OriginalRequest, modelName, false, idx+1, nil, accessKey)

			mu.Lock()
			defer mu.Unlock()

			if err != nil {
				logging.Errorf("[RatingsLooper] Model %s failed: %v", modelName, err)
				errors[idx] = err
			} else {
				responses[idx] = resp
				modelsUsed[idx] = modelName
			}
		}(i, modelRef)
	}

	wg.Wait()

	// Check for fail-fast mode
	if onError == "fail" {
		for i, err := range errors {
			if err != nil {
				return nil, fmt.Errorf("model %d failed: %w", i, err)
			}
		}
	}

	// Collect successful responses
	var successResponses []*ModelResponse
	var successModels []string

	for i := range responses {
		if responses[i] != nil {
			successResponses = append(successResponses, responses[i])
			successModels = append(successModels, modelsUsed[i])
		}
	}

	if len(successResponses) == 0 {
		return nil, fmt.Errorf("all models failed")
	}

	logging.Infof("[RatingsLooper] %d/%d models succeeded", len(successResponses), len(req.ModelRefs))

	// Iterations = total calls made (including failures)
	iterations := len(req.ModelRefs)

	if req.IsStreaming {
		return l.formatRatingsStreamingResponse(successResponses, successModels, iterations)
	}
	return l.formatRatingsJSONResponse(successResponses, successModels, iterations)
}

// formatRatingsJSONResponse creates a response with multiple choices (one per model)
func (l *RatingsLooper) formatRatingsJSONResponse(responses []*ModelResponse, modelsUsed []string, iterations int) (*Response, error) {
	// Build choices array - one choice per model response
	choices := make([]map[string]interface{}, len(responses))
	for i, resp := range responses {
		choices[i] = map[string]interface{}{
			"index": i,
			"message": map[string]interface{}{
				"role":    "assistant",
				"content": resp.Content,
			},
			"finish_reason": "stop",
			"model":         modelsUsed[i], // Include model name in each choice
		}
	}

	completion := map[string]interface{}{
		"id":      fmt.Sprintf("chatcmpl-looper-%d", time.Now().UnixNano()),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   strings.Join(modelsUsed, ","), // Combined model names
		"choices": choices,
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
		Model:         modelsUsed[len(modelsUsed)-1],
		ModelsUsed:    modelsUsed,
		Iterations:    iterations,
		AlgorithmType: "ratings",
	}, nil
}

// formatRatingsStreamingResponse creates an SSE streaming response with multiple choices
func (l *RatingsLooper) formatRatingsStreamingResponse(responses []*ModelResponse, modelsUsed []string, iterations int) (*Response, error) {
	timestamp := time.Now().Unix()
	id := fmt.Sprintf("chatcmpl-looper-%d", timestamp)

	var sseChunks []string

	// First chunk: role delta for each choice
	firstChunk := map[string]interface{}{
		"id":      id,
		"object":  "chat.completion.chunk",
		"created": timestamp,
		"model":   strings.Join(modelsUsed, ","),
		"choices": func() []map[string]interface{} {
			choices := make([]map[string]interface{}, len(responses))
			for i := range responses {
				choices[i] = map[string]interface{}{
					"index": i,
					"delta": map[string]interface{}{
						"role": "assistant",
					},
					"model": modelsUsed[i],
				}
			}
			return choices
		}(),
	}
	firstChunkBytes, _ := json.Marshal(firstChunk)
	sseChunks = append(sseChunks, fmt.Sprintf("data: %s\n\n", firstChunkBytes))

	// Content chunks: stream content for each choice
	// Find the max content length to determine chunk iterations
	maxLen := 0
	for _, resp := range responses {
		if len(resp.Content) > maxLen {
			maxLen = len(resp.Content)
		}
	}

	chunkSize := 50
	for offset := 0; offset < maxLen; offset += chunkSize {
		choices := make([]map[string]interface{}, len(responses))
		for i, resp := range responses {
			content := ""
			if offset < len(resp.Content) {
				end := offset + chunkSize
				if end > len(resp.Content) {
					end = len(resp.Content)
				}
				content = resp.Content[offset:end]
			}
			choices[i] = map[string]interface{}{
				"index": i,
				"delta": map[string]interface{}{
					"content": content,
				},
				"model": modelsUsed[i],
			}
		}

		contentChunk := map[string]interface{}{
			"id":      id,
			"object":  "chat.completion.chunk",
			"created": timestamp,
			"model":   strings.Join(modelsUsed, ","),
			"choices": choices,
		}
		contentChunkBytes, _ := json.Marshal(contentChunk)
		sseChunks = append(sseChunks, fmt.Sprintf("data: %s\n\n", contentChunkBytes))
	}

	// Final chunk: finish_reason for each choice
	finalChunk := map[string]interface{}{
		"id":      id,
		"object":  "chat.completion.chunk",
		"created": timestamp,
		"model":   strings.Join(modelsUsed, ","),
		"choices": func() []map[string]interface{} {
			choices := make([]map[string]interface{}, len(responses))
			for i := range responses {
				choices[i] = map[string]interface{}{
					"index":         i,
					"delta":         map[string]interface{}{},
					"finish_reason": "stop",
					"model":         modelsUsed[i],
				}
			}
			return choices
		}(),
	}
	finalChunkBytes, _ := json.Marshal(finalChunk)
	sseChunks = append(sseChunks, fmt.Sprintf("data: %s\n\n", finalChunkBytes))
	sseChunks = append(sseChunks, "data: [DONE]\n\n")

	return &Response{
		Body:          []byte(strings.Join(sseChunks, "")),
		ContentType:   "text/event-stream",
		Model:         modelsUsed[len(modelsUsed)-1],
		ModelsUsed:    modelsUsed,
		Iterations:    iterations,
		AlgorithmType: "ratings",
	}, nil
}
