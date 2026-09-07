package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

type contextRecoveryCall struct {
	ID  string
	Key string
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
	engine, err := r.protocolEngine()
	if err != nil {
		return responseBody, err
	}
	format := requestCtx.SourceFormat
	if format == "" {
		format = llmprotocol.OpenAIChatV1
	}
	decoded, err := engine.TranslateResponse(format, format, responseBody, nil)
	if err != nil {
		return responseBody, fmt.Errorf("decode context recovery response: %w", err)
	}
	calls, assistant, err := parseContextRecoveryCalls(decoded.Response)
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
		decoded.Response,
	)
	if err != nil {
		return responseBody, err
	}
	return followup, nil
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
) ([]llmprotocol.Message, error) {
	store := r.contextCompressionRecoveryStore(plugin)
	scope := r.contextCompressionScope(requestCtx)
	if store == nil || scope == "" {
		if service := r.contextCompressionService(); service != nil {
			service.RecordRecoveryFailure()
		}
		metrics.RecordContextCompressionRecovery("scope_unavailable", 0, 0)
		return nil, fmt.Errorf("context recovery store or trusted scope is unavailable")
	}
	toolMessages := make([]llmprotocol.Message, 0, len(calls))
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
		toolMessages = append(toolMessages, llmprotocol.Message{
			Role: llmprotocol.RoleTool,
			Content: []llmprotocol.Content{{
				Kind: llmprotocol.ContentToolResult,
				ToolResult: &llmprotocol.ToolResult{
					CallID:  call.ID,
					Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: entry.Content}},
				},
			}},
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
	assistant []llmprotocol.Message,
	toolMessages []llmprotocol.Message,
	initial llmprotocol.Response,
) ([]byte, error) {
	if requestCtx == nil || requestCtx.SemanticRequest == nil {
		return nil, fmt.Errorf("neutral context recovery request is unavailable")
	}
	followupRequest := *requestCtx.SemanticRequest
	followupRequest.Messages = append([]llmprotocol.Message(nil), requestCtx.SemanticRequest.Messages...)
	followupRequest.Messages = append(followupRequest.Messages, assistant...)
	followupRequest.Messages = append(followupRequest.Messages, toolMessages...)
	followupRequest.Stream = false
	followupRequest.Generation++
	model := strings.TrimSpace(requestCtx.VSRSelectedModel)
	if model == "" {
		model = strings.TrimSpace(requestCtx.RequestModel)
	}
	engine, err := r.protocolEngine()
	if err != nil {
		return nil, err
	}
	encodedRequest, err := engine.EncodeRequest(
		llmprotocol.OpenAIChatV1,
		followupRequest,
		llmprotocol.Envelope{},
	)
	if err != nil {
		return nil, fmt.Errorf("encode context recovery followup: %w", err)
	}
	openAIRequest, err := parseOpenAIRequest(encodedRequest.Body)
	if err != nil {
		return nil, fmt.Errorf("prepare context recovery followup: %w", err)
	}
	client := looper.NewClient(&r.Config.Looper)
	client.SetDecisionName(requestCtx.VSRSelectedDecisionName)
	followup, err := client.CallModel(
		ctx,
		openAIRequest,
		model,
		false,
		1,
		nil,
		r.Config.GetModelAccessKey(model),
	)
	if err != nil {
		return nil, fmt.Errorf("context recovery followup failed: %w", err)
	}
	decoded, err := engine.TranslateResponse(
		llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, followup.Raw, nil,
	)
	if err != nil {
		return nil, fmt.Errorf("decode context recovery followup: %w", err)
	}
	decoded.Response.Usage, err = mergeContextRecoveryUsage(initial.Usage, decoded.Response.Usage)
	if err != nil {
		return nil, err
	}
	decoded.Response.Model = requestCtx.RequestModel
	decoded.Response.Generation++
	format := requestCtx.SourceFormat
	if format == "" {
		format = llmprotocol.OpenAIChatV1
	}
	encoded, err := engine.EncodeResponse(format, decoded.Response, llmprotocol.Envelope{})
	if err != nil {
		return nil, fmt.Errorf("encode context recovery followup: %w", err)
	}
	return encoded.Body, nil
}

func stringSet(values []string) map[string]struct{} {
	result := make(map[string]struct{}, len(values))
	for _, value := range values {
		result[value] = struct{}{}
	}
	return result
}

func parseContextRecoveryCalls(
	response llmprotocol.Response,
) ([]contextRecoveryCall, []llmprotocol.Message, error) {
	calls := make([]contextRecoveryCall, 0)
	assistant := make([]llmprotocol.Message, 0, len(response.Output))
	for _, item := range response.Output {
		message := llmprotocol.Message{ID: item.ID, Role: item.Role}
		for _, content := range item.Content {
			if content.Kind != llmprotocol.ContentToolCall || content.ToolCall == nil ||
				content.ToolCall.Name != contextcompression.RetrieveToolName {
				continue
			}
			key, err := contextRecoveryKey(content.ToolCall.Arguments)
			if err != nil {
				return nil, nil, err
			}
			if content.ToolCall.ID == "" || key == "" {
				return nil, nil, fmt.Errorf("context recovery tool call is incomplete")
			}
			calls = append(calls, contextRecoveryCall{ID: content.ToolCall.ID, Key: key})
			message.Content = append(message.Content, content)
		}
		if len(message.Content) > 0 {
			if message.Role == "" {
				message.Role = llmprotocol.RoleAssistant
			}
			assistant = append(assistant, message)
		}
	}
	if len(calls) == 0 {
		return nil, nil, nil
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
	default:
		return "", fmt.Errorf("invalid context recovery arguments")
	}
}

func mergeContextRecoveryUsage(
	initial llmprotocol.Usage,
	followup llmprotocol.Usage,
) (llmprotocol.Usage, error) {
	if initial.State != llmprotocol.UsageAvailable || followup.State != llmprotocol.UsageAvailable {
		return llmprotocol.Usage{State: llmprotocol.UsageUnavailable}, nil
	}
	result := llmprotocol.Usage{State: llmprotocol.UsageAvailable}
	pairs := []struct {
		left, right llmprotocol.TokenCount
		target      *llmprotocol.TokenCount
	}{
		{initial.InputUncached, followup.InputUncached, &result.InputUncached},
		{initial.InputCacheRead, followup.InputCacheRead, &result.InputCacheRead},
		{initial.InputCacheWrite, followup.InputCacheWrite, &result.InputCacheWrite},
		{initial.OutputReasoning, followup.OutputReasoning, &result.OutputReasoning},
		{initial.OutputOther, followup.OutputOther, &result.OutputOther},
		{initial.InputTotal, followup.InputTotal, &result.InputTotal},
		{initial.OutputTotal, followup.OutputTotal, &result.OutputTotal},
		{initial.Total, followup.Total, &result.Total},
	}
	for _, pair := range pairs {
		merged, err := mergeContextRecoveryTokenCount(pair.left, pair.right)
		if err != nil {
			return llmprotocol.Usage{}, err
		}
		*pair.target = merged
	}
	return result, nil
}

func mergeContextRecoveryTokenCount(left, right llmprotocol.TokenCount) (llmprotocol.TokenCount, error) {
	if left.Value == nil || right.Value == nil {
		return llmprotocol.TokenCount{Provenance: llmprotocol.UsageUnknown}, nil
	}
	if *left.Value > math.MaxInt64-*right.Value {
		return llmprotocol.TokenCount{}, fmt.Errorf("context recovery usage overflow")
	}
	value := *left.Value + *right.Value
	provenance := llmprotocol.UsageDerived
	if left.Provenance == llmprotocol.UsageAuthoritative && right.Provenance == llmprotocol.UsageAuthoritative {
		provenance = llmprotocol.UsageAuthoritative
	}
	return llmprotocol.TokenCount{Value: &value, Provenance: provenance}, nil
}

//nolint:cyclop // Redaction walks every neutral output and nested tool result fail-closed.
func (r *OpenAIRouter) redactContextRecoveryToolCalls(responseBody []byte, ctx *RequestContext) []byte {
	if ctx == nil {
		return responseBody
	}
	engine, err := r.protocolEngine()
	if err != nil {
		return responseBody
	}
	format := ctx.SourceFormat
	if format == "" {
		format = llmprotocol.OpenAIChatV1
	}
	decoded, err := engine.TranslateResponse(format, format, responseBody, nil)
	if err != nil {
		return responseBody
	}
	changed := false
	for itemIndex := range decoded.Response.Output {
		item := &decoded.Response.Output[itemIndex]
		kept := item.Content[:0]
		for _, content := range item.Content {
			if content.Kind == llmprotocol.ContentToolCall && content.ToolCall != nil &&
				content.ToolCall.Name == contextcompression.RetrieveToolName {
				changed = true
				continue
			}
			kept = append(kept, content)
		}
		item.Content = kept
		if len(item.Content) == 0 {
			item.Content = []llmprotocol.Content{{
				Kind: llmprotocol.ContentText,
				Text: "Additional compressed context could not be retrieved.",
			}}
		}
	}
	if !changed {
		return responseBody
	}
	decoded.Response.StopReason = llmprotocol.StopEndTurn
	decoded.Response.Generation++
	encoded, err := engine.EncodeResponse(format, decoded.Response, llmprotocol.Envelope{})
	if err != nil {
		return responseBody
	}
	return encoded.Body
}
