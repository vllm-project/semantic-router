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
	"fmt"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
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

	logging.ComponentEvent("looper", "execution_started", map[string]interface{}{
		"looper":           "ratings",
		"decision":         req.DecisionName,
		"candidate_models": len(req.ModelRefs),
		"max_concurrent":   maxConcurrent,
		"on_error":         onError,
		"streaming":        req.IsStreaming,
	})

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

			logging.ComponentDebugEvent("looper", "model_dispatch_started", map[string]interface{}{
				"looper":    "ratings",
				"decision":  req.DecisionName,
				"model_ref": modelName,
				"slot":      idx + 1,
			})

			// Use idx+1 as iteration number for concurrent requests.
			// RatingsLooper doesn't need logprobs (no confidence-based routing).
			resp, err := l.client.CallModel(
				ctx,
				toolFreeLooperRequest(req.executionRequest),
				modelName,
				req.IsStreaming,
				idx+1,
				nil,
			)

			mu.Lock()
			defer mu.Unlock()

			if err != nil {
				logging.ComponentWarnEvent("looper", "model_dispatch_failed", map[string]interface{}{
					"looper":    "ratings",
					"decision":  req.DecisionName,
					"model_ref": modelName,
					"slot":      idx + 1,
					"error":     err.Error(),
				})
				errors[idx] = err
			} else {
				responses[idx] = resp
				modelsUsed[idx] = modelName
			}
		}(i, modelRef)
	}

	wg.Wait()

	// Collect successful responses before evaluating fail-fast so paid work is
	// retained even when a sibling model failed.
	var successResponses []*ModelResponse
	var successModels []string
	for i := range responses {
		if responses[i] != nil {
			successResponses = append(successResponses, responses[i])
			successModels = append(successModels, modelsUsed[i])
		}
	}
	iterations := len(req.ModelRefs)
	evidence := executionEvidenceFromResponses(successResponses, successModels, iterations)

	// Check for fail-fast mode
	if onError == "fail" {
		for i, err := range errors {
			if err != nil {
				return nil, newPartialExecutionError(fmt.Errorf("model %d failed: %w", i, err), evidence)
			}
		}
	}

	if len(successResponses) == 0 {
		return nil, fmt.Errorf("all models failed")
	}

	logging.ComponentEvent("looper", "execution_completed", map[string]interface{}{
		"looper":            "ratings",
		"decision":          req.DecisionName,
		"successful_models": len(successResponses),
		"total_models":      len(req.ModelRefs),
		"models_used":       successModels,
	})

	var response *Response
	var err error
	if req.IsStreaming {
		response, err = l.formatRatingsStreamingResponse(successResponses, successModels, iterations, streamUsageRequested(req))
	} else {
		response, err = l.formatRatingsJSONResponse(successResponses, successModels, iterations)
	}
	if err != nil {
		return nil, newPartialExecutionError(err, evidence)
	}
	return response, nil
}

// formatRatingsJSONResponse creates a response with multiple choices (one per model)
func (l *RatingsLooper) formatRatingsJSONResponse(responses []*ModelResponse, modelsUsed []string, iterations int) (*Response, error) {
	if len(responses) == 0 || len(modelsUsed) == 0 {
		return nil, fmt.Errorf("ratings produced no responses")
	}
	usage := SumUsage(responses...)
	semantic := newTextSemanticResponse("response-ratings", modelsUsed[len(modelsUsed)-1], responses[0].Content, usage)
	for index := 1; index < len(responses); index++ {
		semantic.Alternatives = append(semantic.Alternatives, []llmprotocol.OutputItem{{
			ID:      llmprotocol.StableID(semantic.ID, fmt.Sprint(index)),
			Role:    llmprotocol.RoleAssistant,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: responses[index].Content}},
		}})
	}
	return newLooperResponse(semantic, false, true, modelsUsed[len(modelsUsed)-1], modelsUsed, iterations, "ratings", usage, nil), nil
}

// formatRatingsStreamingResponse creates an SSE streaming response with multiple choices
func (l *RatingsLooper) formatRatingsStreamingResponse(
	responses []*ModelResponse,
	modelsUsed []string,
	iterations int,
	includeUsage bool,
) (*Response, error) {
	_ = responses
	_ = modelsUsed
	_ = iterations
	_ = includeUsage
	return nil, llmprotocol.NewError(
		llmprotocol.ErrorUnsupportedFeature,
		"stream_alternatives_unsupported",
		"streaming multiple rating alternatives is unsupported",
		nil,
	)
}
