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
	"regexp"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

var taggedToolCallPattern = regexp.MustCompile(`(?s)<tool_call>\s*(\{.*?\})\s*</tool_call>`)

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

	logging.ComponentEvent("looper", "execution_started", map[string]interface{}{
		"looper":           "base",
		"decision":         req.DecisionName,
		"candidate_models": len(req.ModelRefs),
		"streaming":        req.IsStreaming,
	})

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

		logging.ComponentDebugEvent("looper", "model_dispatch_started", map[string]interface{}{
			"looper":    "base",
			"decision":  req.DecisionName,
			"model_ref": modelName,
			"iteration": iteration,
		})

		// BaseLooper doesn't need logprobs (no confidence-based routing).
		resp, err := l.client.CallModel(
			ctx,
			toolFreeLooperRequest(req.executionRequest),
			modelName,
			req.IsStreaming,
			iteration,
			nil,
		)
		if err != nil {
			logging.ComponentWarnEvent("looper", "model_dispatch_failed", map[string]interface{}{
				"looper":    "base",
				"decision":  req.DecisionName,
				"model_ref": modelName,
				"iteration": iteration,
				"error":     err.Error(),
			})
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

	// Format output based on streaming preference.
	if req.IsStreaming {
		return l.formatStreamingResponse(aggregated, modelsUsed, iteration, streamUsageRequested(req))
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

	// Use the last response's logprobs and tool_calls flag for confidence
	if len(responses) > 0 {
		lastResp := responses[len(responses)-1]
		result.AverageLogprob = lastResp.AverageLogprob
		result.HasToolCalls = lastResp.HasToolCalls
	}

	logging.ComponentEvent("looper", "execution_completed", map[string]interface{}{
		"looper":               "base",
		"responses":            len(responses),
		"models_used":          models,
		"selected_model":       result.FinalModel,
		"combined_content_len": len(combinedContent),
	})

	return result
}

// AggregatedResponse holds the combined result from multiple models
type AggregatedResponse struct {
	Models    []string
	Responses []*ModelResponse
	// UsageResponses contains every paid model call when Responses is narrowed
	// to the candidate whose content is safe to publish. Nil means Responses.
	UsageResponses  []*ModelResponse
	CombinedContent string
	FinalModel      string
	AverageLogprob  float64
	HasToolCalls    bool
}

func aggregatedUsageResponses(agg *AggregatedResponse) []*ModelResponse {
	if agg != nil && agg.UsageResponses != nil {
		return agg.UsageResponses
	}
	if agg == nil {
		return nil
	}
	return agg.Responses
}

// formatJSONResponse creates a JSON ChatCompletion response.
// When the final response contains tool_calls, the original raw response
// is preserved (with metadata patched) to avoid dropping tool_calls.
func (l *BaseLooper) formatJSONResponse(agg *AggregatedResponse, modelsUsed []string, iterations int) (*Response, error) {
	usage := SumUsage(aggregatedUsageResponses(agg)...)
	if agg.HasToolCalls && len(agg.Responses) > 0 {
		semantic, err := newModelSemanticResponse(
			"response-looper", agg.Responses[len(agg.Responses)-1], agg.FinalModel, usage,
		)
		if err != nil {
			return nil, err
		}
		return newLooperResponse(semantic, false, true, agg.FinalModel, modelsUsed, iterations, "simple", usage, nil), nil
	}
	if len(agg.Responses) > 0 {
		last := agg.Responses[len(agg.Responses)-1]
		if semantic, ok := newTaggedToolSemanticResponse("response-looper", agg.FinalModel, last.Content, usage); ok {
			return newLooperResponse(semantic, false, true, agg.FinalModel, modelsUsed, iterations, "simple", usage, nil), nil
		}
	}
	semantic := newTextSemanticResponse("response-looper", agg.FinalModel, agg.CombinedContent, usage)
	return newLooperResponse(semantic, false, true, agg.FinalModel, modelsUsed, iterations, "simple", usage, nil), nil
}

func parseTaggedToolCall(content string) (string, string, bool) {
	matches := taggedToolCallPattern.FindStringSubmatch(content)
	if len(matches) < 2 {
		return "", "", false
	}

	var parsed struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	if err := json.Unmarshal([]byte(matches[1]), &parsed); err != nil {
		return "", "", false
	}
	if strings.TrimSpace(parsed.Name) == "" {
		return "", "", false
	}

	argsJSON := strings.TrimSpace(string(parsed.Arguments))
	if argsJSON == "" || argsJSON == "null" {
		argsJSON = "{}"
	} else if strings.HasPrefix(argsJSON, "\"") {
		var decoded string
		if err := json.Unmarshal(parsed.Arguments, &decoded); err == nil {
			argsJSON = decoded
		}
	}

	if !json.Valid([]byte(argsJSON)) {
		fallback, _ := json.Marshal(map[string]string{"input": argsJSON})
		argsJSON = string(fallback)
	}

	return parsed.Name, argsJSON, true
}

// formatStreamingResponse creates an SSE streaming response
func (l *BaseLooper) formatStreamingResponse(
	agg *AggregatedResponse,
	modelsUsed []string,
	iterations int,
	includeUsage bool,
) (*Response, error) {
	usage := SumUsage(aggregatedUsageResponses(agg)...)
	if agg.HasToolCalls && len(agg.Responses) > 0 {
		semantic, err := newModelSemanticResponse(
			"response-looper", agg.Responses[len(agg.Responses)-1], agg.FinalModel, usage,
		)
		if err != nil {
			return nil, err
		}
		return newLooperResponse(semantic, true, includeUsage, agg.FinalModel, modelsUsed, iterations, "simple", usage, nil), nil
	}
	if semantic, ok := newTaggedToolSemanticResponse("response-looper", agg.FinalModel, agg.CombinedContent, usage); ok {
		return newLooperResponse(semantic, true, includeUsage, agg.FinalModel, modelsUsed, iterations, "simple", usage, nil), nil
	}
	semantic := newTextSemanticResponse("response-looper", agg.FinalModel, agg.CombinedContent, usage)
	return newLooperResponse(semantic, true, includeUsage, agg.FinalModel, modelsUsed, iterations, "simple", usage, nil), nil
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
