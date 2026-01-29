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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// RLDrivenLooper implements multi-round routing using RL-driven model selection.
// This integrates Router-R1's multi-round aggregation (arXiv:2506.09033) with
// VSR's looper execution framework.
//
// The algorithm:
//  1. Uses RLDrivenSelector.SelectMultiRound() to determine which models to call
//  2. Calls each selected model and collects responses
//  3. Uses AggregateResponses() to combine results using learned quality scores
//  4. Returns the aggregated response with quality-weighted combination
type RLDrivenLooper struct {
	client   *Client
	cfg      *config.LooperConfig
	selector *selection.RLDrivenSelector
}

// NewRLDrivenLooper creates a new RLDrivenLooper instance
func NewRLDrivenLooper(cfg *config.LooperConfig) *RLDrivenLooper {
	// Get the RL-driven selector from the global registry
	var rlSelector *selection.RLDrivenSelector
	if s, ok := selection.GlobalRegistry.Get(selection.MethodRLDriven); ok {
		if typed, ok := s.(*selection.RLDrivenSelector); ok {
			rlSelector = typed
		}
	}

	// If not registered, create a new one with default config
	if rlSelector == nil {
		rlCfg := selection.DefaultRLDrivenConfig()
		rlCfg.EnableMultiRoundAggregation = true
		rlSelector = selection.NewRLDrivenSelector(rlCfg)
		logging.Infof("[RLDrivenLooper] Created new RLDrivenSelector with multi-round enabled")
	}

	return &RLDrivenLooper{
		client:   NewClient(cfg),
		cfg:      cfg,
		selector: rlSelector,
	}
}

// Execute implements the RL-driven multi-round routing algorithm.
// It uses Thompson Sampling to select models and aggregates their responses.
func (l *RLDrivenLooper) Execute(ctx context.Context, req *Request) (*Response, error) {
	if len(req.ModelRefs) == 0 {
		return nil, fmt.Errorf("no models configured")
	}

	logging.Infof("[RLDrivenLooper] Starting multi-round execution with %d candidate models, streaming=%v",
		len(req.ModelRefs), req.IsStreaming)

	// Build selection context from request
	selCtx := l.buildSelectionContext(req)

	// Use SelectMultiRound to determine which models to call
	multiRoundResult, err := l.selector.SelectMultiRound(ctx, selCtx)
	if err != nil {
		logging.Warnf("[RLDrivenLooper] SelectMultiRound failed, falling back to all models: %v", err)
		// Fallback: use all models if multi-round selection fails
		return l.executeAllModels(ctx, req)
	}

	logging.Infof("[RLDrivenLooper] Multi-round selected %d models: %v",
		len(multiRoundResult.SelectedModels), multiRoundResult.SelectedModels)

	// Call each selected model
	responses := make(map[string]string)
	scores := make(map[string]float64)
	var modelsUsed []string
	iteration := 0

	for _, modelName := range multiRoundResult.SelectedModels {
		iteration++

		// Find the model ref for this model
		var modelRef *config.ModelRef
		for i := range req.ModelRefs {
			if req.ModelRefs[i].Model == modelName || req.ModelRefs[i].LoRAName == modelName {
				modelRef = &req.ModelRefs[i]
				break
			}
		}

		if modelRef == nil {
			logging.Warnf("[RLDrivenLooper] Model %s not found in ModelRefs, skipping", modelName)
			continue
		}

		// Get access key from model params
		accessKey := ""
		if req.ModelParams != nil {
			if params, ok := req.ModelParams[modelRef.Model]; ok {
				accessKey = params.AccessKey
			}
		}

		displayName := modelRef.Model
		if modelRef.LoRAName != "" {
			displayName = modelRef.LoRAName
		}

		// Get score from multi-round result
		score := 0.0
		if multiRoundResult.Scores != nil {
			score = multiRoundResult.Scores[modelName]
		}

		logging.Infof("[RLDrivenLooper] Calling model: %s (iteration=%d, score=%.3f)",
			displayName, iteration, score)

		// Call the model
		resp, callErr := l.client.CallModel(ctx, req.OriginalRequest, displayName, false, iteration, nil, accessKey)
		if callErr != nil {
			logging.Errorf("[RLDrivenLooper] Model %s failed: %v", displayName, callErr)
			continue
		}

		responses[modelName] = resp.Content
		scores[modelName] = score
		modelsUsed = append(modelsUsed, displayName)
	}

	if len(responses) == 0 {
		return nil, fmt.Errorf("all selected models failed")
	}

	// Use the selector's AggregateResponses to combine results
	aggregatedContent, bestModel, err := l.selector.AggregateResponses(responses, scores)
	if err != nil {
		logging.Warnf("[RLDrivenLooper] AggregateResponses failed, using first response: %v", err)
		// Fallback: use first response
		for model, content := range responses {
			aggregatedContent = content
			bestModel = model
			break
		}
	}

	logging.Infof("[RLDrivenLooper] Aggregated %d responses, best model: %s, content length: %d",
		len(responses), bestModel, len(aggregatedContent))

	// Format output
	if req.IsStreaming {
		return l.formatStreamingResponse(aggregatedContent, bestModel, modelsUsed, iteration)
	}
	return l.formatJSONResponse(aggregatedContent, bestModel, modelsUsed, iteration)
}

// buildSelectionContext creates a SelectionContext from the looper Request
func (l *RLDrivenLooper) buildSelectionContext(req *Request) *selection.SelectionContext {
	// Extract query from the last user message
	// Note: In the looper context, the messages are already in the request
	// For simplicity, we'll use a generic query indicator
	query := "looper_multi_round_request"

	return &selection.SelectionContext{
		Query:           query,
		CandidateModels: req.ModelRefs,
	}
}

// executeAllModels is the fallback when SelectMultiRound fails
func (l *RLDrivenLooper) executeAllModels(ctx context.Context, req *Request) (*Response, error) {
	var responses []*ModelResponse
	var modelsUsed []string
	iteration := 0

	for _, modelRef := range req.ModelRefs {
		iteration++
		modelName := modelRef.Model
		if modelRef.LoRAName != "" {
			modelName = modelRef.LoRAName
		}

		accessKey := ""
		if req.ModelParams != nil {
			if params, ok := req.ModelParams[modelRef.Model]; ok {
				accessKey = params.AccessKey
			}
		}

		logging.Infof("[RLDrivenLooper] Fallback: Calling model: %s (iteration=%d)", modelName, iteration)

		resp, err := l.client.CallModel(ctx, req.OriginalRequest, modelName, false, iteration, nil, accessKey)
		if err != nil {
			logging.Errorf("[RLDrivenLooper] Model %s failed: %v", modelName, err)
			continue
		}

		responses = append(responses, resp)
		modelsUsed = append(modelsUsed, modelName)
	}

	if len(responses) == 0 {
		return nil, fmt.Errorf("all models failed")
	}

	// Simple aggregation for fallback
	var content string
	for i, resp := range responses {
		if i > 0 {
			content += "\n\n---\n\n"
		}
		content += fmt.Sprintf("**[%s]:**\n%s", modelsUsed[i], resp.Content)
	}

	if req.IsStreaming {
		return l.formatStreamingResponse(content, modelsUsed[len(modelsUsed)-1], modelsUsed, iteration)
	}
	return l.formatJSONResponse(content, modelsUsed[len(modelsUsed)-1], modelsUsed, iteration)
}

// formatJSONResponse creates a JSON ChatCompletion response
func (l *RLDrivenLooper) formatJSONResponse(content, model string, modelsUsed []string, iterations int) (*Response, error) {
	completion := map[string]interface{}{
		"id":      fmt.Sprintf("chatcmpl-rl-%d", time.Now().UnixNano()),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": []map[string]interface{}{
			{
				"index": 0,
				"message": map[string]interface{}{
					"role":    "assistant",
					"content": content,
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
		Model:         model,
		ModelsUsed:    modelsUsed,
		Iterations:    iterations,
		AlgorithmType: "rl_driven",
	}, nil
}

// formatStreamingResponse creates an SSE streaming response
func (l *RLDrivenLooper) formatStreamingResponse(content, model string, modelsUsed []string, iterations int) (*Response, error) {
	timestamp := time.Now().Unix()
	id := fmt.Sprintf("chatcmpl-rl-%d", timestamp)

	chunks := splitIntoChunks(content, 50)
	var sseBody []byte

	// First chunk with role
	firstChunk := map[string]interface{}{
		"id":      id,
		"object":  "chat.completion.chunk",
		"created": timestamp,
		"model":   model,
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
			"model":   model,
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

	// Final chunk
	finalChunk := map[string]interface{}{
		"id":      id,
		"object":  "chat.completion.chunk",
		"created": timestamp,
		"model":   model,
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
	sseBody = append(sseBody, []byte("data: [DONE]\n\n")...)

	return &Response{
		Body:          sseBody,
		ContentType:   "text/event-stream",
		Model:         model,
		ModelsUsed:    modelsUsed,
		Iterations:    iterations,
		AlgorithmType: "rl_driven",
	}, nil
}
