package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

type contextRecoveryCall struct {
	ID  string
	Key string
	Raw map[string]interface{}
}

func (r *OpenAIRouter) handleContextRecoveryFollowup(
	ctx context.Context,
	responseBody []byte,
	requestCtx *RequestContext,
) ([]byte, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	plugin := activeContextRecoveryPlugin(requestCtx)
	if plugin == nil {
		return responseBody, nil
	}
	calls, assistant, err := parseContextRecoveryCalls(responseBody)
	if err != nil || len(calls) == 0 {
		return responseBody, err
	}
	if len(calls) > contextRecoveryLimit(plugin) {
		return responseBody, fmt.Errorf("context recovery retrieval limit exceeded")
	}
	toolMessages, err := r.loadContextRecoveryToolMessages(
		ctx,
		requestCtx,
		plugin,
		calls,
	)
	if err != nil {
		return responseBody, err
	}
	followup, err := r.executeContextRecoveryFollowup(
		ctx,
		requestCtx,
		assistant,
		toolMessages,
	)
	if err != nil {
		return responseBody, err
	}
	return mergeContextRecoveryUsage(responseBody, followup), nil
}

func activeContextRecoveryPlugin(
	requestCtx *RequestContext,
) *config.ContextCompressionPluginConfig {
	if requestCtx == nil ||
		requestCtx.VSRSelectedDecision == nil ||
		requestCtx.ExpectStreamingResponse ||
		len(requestCtx.ContextCompressionRecoveryKeys) == 0 {
		return nil
	}
	plugin := requestCtx.VSRSelectedDecision.GetContextCompressionConfig()
	if plugin == nil || plugin.Recovery == nil || !plugin.Recovery.Enabled {
		return nil
	}
	return plugin
}

func contextRecoveryLimit(
	plugin *config.ContextCompressionPluginConfig,
) int {
	if plugin.Recovery.MaxRetrievals > 0 {
		return plugin.Recovery.MaxRetrievals
	}
	return 8
}

func (r *OpenAIRouter) loadContextRecoveryToolMessages(
	ctx context.Context,
	requestCtx *RequestContext,
	plugin *config.ContextCompressionPluginConfig,
	calls []contextRecoveryCall,
) ([]interface{}, error) {
	store := r.contextCompressionRecoveryStore(plugin)
	scope := r.contextCompressionScope(requestCtx)
	if store == nil || scope == "" {
		if service := r.contextCompressionService(); service != nil {
			service.RecordRecoveryFailure()
		}
		metrics.RecordContextCompressionRecovery("scope_unavailable", 0, 0)
		return nil, fmt.Errorf("context recovery store or trusted scope is unavailable")
	}
	toolMessages := make([]interface{}, 0, len(calls))
	allowed := stringSet(requestCtx.ContextCompressionRecoveryKeys)
	for _, call := range calls {
		if _, ok := allowed[call.Key]; !ok {
			return nil, fmt.Errorf("context recovery key was not issued for this request")
		}
		entry, getErr := store.Get(ctx, scope, call.Key)
		if getErr != nil {
			if service := r.contextCompressionService(); service != nil {
				service.RecordRecoveryFailure()
			}
			metrics.RecordContextCompressionRecovery("retrieve_failed", 0, 0)
			return nil, fmt.Errorf("retrieve compressed context: %w", getErr)
		}
		r.recordContextRecoveryRead(entry)
		toolMessages = append(toolMessages, map[string]interface{}{
			"role":         "tool",
			"tool_call_id": call.ID,
			"content":      entry.Content,
		})
	}
	return toolMessages, nil
}

func (r *OpenAIRouter) recordContextRecoveryRead(
	entry contextcompression.RecoveryEntry,
) {
	if service := r.contextCompressionService(); service != nil {
		service.RecordRecoveryRead()
	}
	now := time.Now()
	metrics.RecordContextCompressionRecovery(
		"retrieved",
		max(0, now.Sub(entry.CreatedAt).Seconds()),
		max(0, entry.ExpiresAt.Sub(now).Seconds()),
	)
}

func (r *OpenAIRouter) executeContextRecoveryFollowup(
	ctx context.Context,
	requestCtx *RequestContext,
	assistant map[string]interface{},
	toolMessages []interface{},
) ([]byte, error) {
	followupBody, err := buildContextRecoveryFollowupBody(
		requestCtx.workingRequestBody(),
		assistant,
		toolMessages,
	)
	if err != nil {
		return nil, err
	}
	followupRequest, err := parseOpenAIRequest(followupBody)
	if err != nil {
		return nil, fmt.Errorf("parse context recovery followup: %w", err)
	}
	model := strings.TrimSpace(requestCtx.VSRSelectedModel)
	if model == "" {
		model = strings.TrimSpace(requestCtx.RequestModel)
	}
	client := looper.NewClient(&r.Config.Looper)
	client.SetDecisionName(requestCtx.VSRSelectedDecisionName)
	followup, err := client.CallModel(
		ctx,
		followupRequest,
		model,
		false,
		1,
		nil,
		r.getModelAccessKey(model),
	)
	if err != nil {
		return nil, fmt.Errorf("context recovery followup failed: %w", err)
	}
	return followup.Raw, nil
}

func stringSet(values []string) map[string]struct{} {
	result := make(map[string]struct{}, len(values))
	for _, value := range values {
		result[value] = struct{}{}
	}
	return result
}

func parseContextRecoveryCalls(
	body []byte,
) ([]contextRecoveryCall, map[string]interface{}, error) {
	var response map[string]interface{}
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, nil, nil
	}
	choices, _ := response["choices"].([]interface{})
	if len(choices) == 0 {
		return nil, nil, nil
	}
	choice, _ := choices[0].(map[string]interface{})
	message, _ := choice["message"].(map[string]interface{})
	rawCalls, _ := message["tool_calls"].([]interface{})
	calls := make([]contextRecoveryCall, 0)
	assistantCalls := make([]interface{}, 0)
	for _, rawCall := range rawCalls {
		call, ok := rawCall.(map[string]interface{})
		if !ok {
			continue
		}
		function, _ := call["function"].(map[string]interface{})
		if function["name"] != contextcompression.RetrieveToolName {
			continue
		}
		id, _ := call["id"].(string)
		key, err := contextRecoveryKey(function["arguments"])
		if err != nil {
			return nil, nil, err
		}
		if id == "" || key == "" {
			return nil, nil, fmt.Errorf("context recovery tool call is incomplete")
		}
		calls = append(calls, contextRecoveryCall{ID: id, Key: key, Raw: call})
		assistantCalls = append(assistantCalls, call)
	}
	if len(calls) == 0 {
		return nil, nil, nil
	}
	assistant := map[string]interface{}{
		"role":       "assistant",
		"content":    message["content"],
		"tool_calls": assistantCalls,
	}
	return calls, assistant, nil
}

func contextRecoveryKey(arguments interface{}) (string, error) {
	switch typed := arguments.(type) {
	case string:
		var decoded map[string]interface{}
		if err := json.Unmarshal([]byte(typed), &decoded); err != nil {
			return "", fmt.Errorf("invalid context recovery arguments")
		}
		key, _ := decoded["key"].(string)
		return key, nil
	case map[string]interface{}:
		key, _ := typed["key"].(string)
		return key, nil
	default:
		return "", fmt.Errorf("invalid context recovery arguments")
	}
}

func buildContextRecoveryFollowupBody(
	workingBody []byte,
	assistant map[string]interface{},
	toolMessages []interface{},
) ([]byte, error) {
	var request map[string]interface{}
	if err := json.Unmarshal(workingBody, &request); err != nil {
		return nil, fmt.Errorf("parse working body for context recovery: %w", err)
	}
	messages, _ := request["messages"].([]interface{})
	messages = append(messages, assistant)
	messages = append(messages, toolMessages...)
	request["messages"] = messages
	request["stream"] = false
	return json.Marshal(request)
}

func mergeContextRecoveryUsage(
	initialBody []byte,
	followupBody []byte,
) []byte {
	var initial map[string]interface{}
	var followup map[string]interface{}
	if json.Unmarshal(initialBody, &initial) != nil ||
		json.Unmarshal(followupBody, &followup) != nil {
		return followupBody
	}
	initialUsage, _ := initial["usage"].(map[string]interface{})
	followupUsage, _ := followup["usage"].(map[string]interface{})
	if len(initialUsage) == 0 {
		return followupBody
	}
	if followupUsage == nil {
		followupUsage = make(map[string]interface{})
		followup["usage"] = followupUsage
	}
	for _, field := range []string{
		"prompt_tokens",
		"completion_tokens",
		"total_tokens",
		"input_tokens",
		"output_tokens",
	} {
		initialValue, initialOK := numericUsageValue(initialUsage[field])
		if !initialOK {
			continue
		}
		followupValue, _ := numericUsageValue(followupUsage[field])
		followupUsage[field] = initialValue + followupValue
	}
	merged, err := json.Marshal(followup)
	if err != nil {
		return followupBody
	}
	return merged
}

func numericUsageValue(value interface{}) (float64, bool) {
	switch typed := value.(type) {
	case float64:
		return typed, true
	case json.Number:
		parsed, err := typed.Float64()
		return parsed, err == nil
	default:
		return 0, false
	}
}

func redactContextRecoveryToolCalls(responseBody []byte) []byte {
	var response map[string]interface{}
	if json.Unmarshal(responseBody, &response) != nil {
		return responseBody
	}
	choices, _ := response["choices"].([]interface{})
	changed := false
	for _, rawChoice := range choices {
		choice, _ := rawChoice.(map[string]interface{})
		message, _ := choice["message"].(map[string]interface{})
		rawCalls, _ := message["tool_calls"].([]interface{})
		if len(rawCalls) == 0 {
			continue
		}
		kept := make([]interface{}, 0, len(rawCalls))
		for _, rawCall := range rawCalls {
			call, _ := rawCall.(map[string]interface{})
			function, _ := call["function"].(map[string]interface{})
			if function["name"] == contextcompression.RetrieveToolName {
				changed = true
				continue
			}
			kept = append(kept, rawCall)
		}
		if len(kept) > 0 {
			message["tool_calls"] = kept
			continue
		}
		delete(message, "tool_calls")
		if content, _ := message["content"].(string); strings.TrimSpace(content) == "" {
			message["content"] = "Additional compressed context could not be retrieved."
		}
		choice["finish_reason"] = "stop"
	}
	if !changed {
		return responseBody
	}
	redacted, err := json.Marshal(response)
	if err != nil {
		return responseBody
	}
	return redacted
}
