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
	"time"

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

// SetEndpointOverrides sets per-model endpoint URL overrides on the underlying client.
func (l *BaseLooper) SetEndpointOverrides(overrides map[string]string) {
	l.client.SetEndpointOverrides(overrides)
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

	// Use the last response's logprobs and tool_calls flag for confidence
	if len(responses) > 0 {
		lastResp := responses[len(responses)-1]
		result.AverageLogprob = lastResp.AverageLogprob
		result.HasToolCalls = lastResp.HasToolCalls
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
	HasToolCalls    bool
}

// formatJSONResponse creates a JSON ChatCompletion response.
// When the final response contains tool_calls, the original raw response
// is preserved (with metadata patched) to avoid dropping tool_calls.
func (l *BaseLooper) formatJSONResponse(agg *AggregatedResponse, modelsUsed []string, iterations int) (*Response, error) {
	// If the final response has tool_calls, use the original raw response
	// but patch the model name and id to reflect the looper wrapper.
	if agg.HasToolCalls && len(agg.Responses) > 0 {
		last := agg.Responses[len(agg.Responses)-1]
		if last.Raw != nil {
			var raw map[string]interface{}
			if err := json.Unmarshal(last.Raw, &raw); err == nil {
				raw["id"] = fmt.Sprintf("chatcmpl-looper-%d", time.Now().UnixNano())
				raw["model"] = agg.FinalModel
				body, err := json.Marshal(raw)
				if err == nil {
					return &Response{
						Body:          body,
						ContentType:   "application/json",
						Model:         agg.FinalModel,
						ModelsUsed:    modelsUsed,
						Iterations:    iterations,
						AlgorithmType: "simple",
					}, nil
				}
			}
		}
	}

	// Fallback compatibility path:
	// Some OpenAI-compatible backends emit "<tool_call>{...}</tool_call>" in content
	// under tool_choice=auto instead of structured tool_calls. Convert that payload
	// to tool_calls so downstream agents can execute tools.
	if len(agg.Responses) > 0 {
		last := agg.Responses[len(agg.Responses)-1]
		if body, ok := rewriteTaggedToolCallResponse(last.Raw, agg.FinalModel); ok {
			return &Response{
				Body:          body,
				ContentType:   "application/json",
				Model:         agg.FinalModel,
				ModelsUsed:    modelsUsed,
				Iterations:    iterations,
				AlgorithmType: "simple",
			}, nil
		}
	}

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

func rewriteTaggedToolCallResponse(raw []byte, finalModel string) ([]byte, bool) {
	if len(raw) == 0 {
		return nil, false
	}

	var completion map[string]interface{}
	if err := json.Unmarshal(raw, &completion); err != nil {
		return nil, false
	}

	choices, ok := completion["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return nil, false
	}

	firstChoice, ok := choices[0].(map[string]interface{})
	if !ok {
		return nil, false
	}

	message, ok := firstChoice["message"].(map[string]interface{})
	if !ok {
		return nil, false
	}

	content, _ := message["content"].(string)
	toolName, argsJSON, ok := parseTaggedToolCall(content)
	if !ok {
		return nil, false
	}

	message["content"] = ""
	message["tool_calls"] = []map[string]interface{}{
		{
			"id":   fmt.Sprintf("chatcmpl-tool-%d", time.Now().UnixNano()),
			"type": "function",
			"function": map[string]interface{}{
				"name":      toolName,
				"arguments": argsJSON,
			},
		},
	}
	firstChoice["finish_reason"] = "tool_calls"
	completion["id"] = fmt.Sprintf("chatcmpl-looper-%d", time.Now().UnixNano())
	completion["model"] = finalModel

	body, err := json.Marshal(completion)
	if err != nil {
		return nil, false
	}
	return body, true
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
func (l *BaseLooper) formatStreamingResponse(agg *AggregatedResponse, modelsUsed []string, iterations int) (*Response, error) {
	timestamp := time.Now().Unix()
	id := fmt.Sprintf("chatcmpl-looper-%d", timestamp)

	toolName, toolArgs, toolCallID, hasToolCall := resolveToolCallForStreaming(agg)

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

	if hasToolCall {
		toolChunk := map[string]interface{}{
			"id":      id,
			"object":  "chat.completion.chunk",
			"created": timestamp,
			"model":   agg.FinalModel,
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"delta": map[string]interface{}{
						"tool_calls": []map[string]interface{}{
							{
								"index": 0,
								"id":    toolCallID,
								"type":  "function",
								"function": map[string]interface{}{
									"name":      toolName,
									"arguments": toolArgs,
								},
							},
						},
					},
					"finish_reason": nil,
				},
			},
		}
		toolChunkJSON, _ := json.Marshal(toolChunk)
		sseBody = append(sseBody, []byte(fmt.Sprintf("data: %s\n\n", toolChunkJSON))...)
	} else {
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
	}

	// Final chunk with finish_reason
	finalReason := "stop"
	if hasToolCall {
		finalReason = "tool_calls"
	}
	finalChunk := map[string]interface{}{
		"id":      id,
		"object":  "chat.completion.chunk",
		"created": timestamp,
		"model":   agg.FinalModel,
		"choices": []map[string]interface{}{
			{
				"index":         0,
				"delta":         map[string]interface{}{},
				"finish_reason": finalReason,
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

func resolveToolCallForStreaming(agg *AggregatedResponse) (string, string, string, bool) {
	if len(agg.Responses) == 0 {
		return "", "", "", false
	}

	last := agg.Responses[len(agg.Responses)-1]
	if name, args, callID, ok := parseFirstToolCallFromRaw(last.Raw); ok {
		return name, args, callID, true
	}

	if name, args, ok := parseTaggedToolCall(agg.CombinedContent); ok {
		return name, args, fmt.Sprintf("chatcmpl-tool-%d", time.Now().UnixNano()), true
	}

	return "", "", "", false
}

func parseFirstToolCallFromRaw(raw []byte) (string, string, string, bool) {
	if len(raw) == 0 {
		return "", "", "", false
	}

	var completion map[string]interface{}
	if err := json.Unmarshal(raw, &completion); err != nil {
		return "", "", "", false
	}

	choices, ok := completion["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return "", "", "", false
	}

	firstChoice, ok := choices[0].(map[string]interface{})
	if !ok {
		return "", "", "", false
	}

	message, ok := firstChoice["message"].(map[string]interface{})
	if !ok {
		return "", "", "", false
	}

	toolCalls, ok := message["tool_calls"].([]interface{})
	if !ok || len(toolCalls) == 0 {
		return "", "", "", false
	}

	firstTool, ok := toolCalls[0].(map[string]interface{})
	if !ok {
		return "", "", "", false
	}

	function, ok := firstTool["function"].(map[string]interface{})
	if !ok {
		return "", "", "", false
	}

	name, _ := function["name"].(string)
	args, _ := function["arguments"].(string)
	callID, _ := firstTool["id"].(string)
	if strings.TrimSpace(name) == "" {
		return "", "", "", false
	}
	if strings.TrimSpace(args) == "" {
		args = "{}"
	}
	if strings.TrimSpace(callID) == "" {
		callID = fmt.Sprintf("chatcmpl-tool-%d", time.Now().UnixNano())
	}
	return name, args, callID, true
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
