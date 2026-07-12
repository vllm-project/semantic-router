package extproc

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

func (r *OpenAIRouter) buildResponseAPIStreamingBodyMutation(
	responseBody []byte,
	ctx *RequestContext,
) *ext_proc.BodyMutation {
	translatedBody := r.translateChatCompletionSSEToResponsesSSE(responseBody, ctx)
	return &ext_proc.BodyMutation{
		Mutation: &ext_proc.BodyMutation_Body{
			Body: translatedBody,
		},
	}
}

func responseAPIStreamingHeaderMutation() *ext_proc.HeaderMutation {
	return &ext_proc.HeaderMutation{
		SetHeaders: []*core.HeaderValueOption{
			{
				Header: &core.HeaderValue{
					Key:   "content-type",
					Value: "text/event-stream; charset=utf-8",
				},
			},
		},
		RemoveHeaders: []string{"content-length"},
	}
}

func (r *OpenAIRouter) translateChatCompletionSSEToResponsesSSE(
	responseBody []byte,
	ctx *RequestContext,
) []byte {
	if ctx == nil || ctx.ResponseAPICtx == nil || ctx.ResponseAPICtx.OriginalRequest == nil {
		return nil
	}

	var out bytes.Buffer
	lines := strings.Split(string(responseBody), "\n")
	for _, line := range lines {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimSpace(strings.TrimPrefix(line, "data: "))
		if data == "" {
			continue
		}

		if data == "[DONE]" {
			r.writeResponseAPIStreamDoneEvents(&out, ctx)
			continue
		}

		var chunkData map[string]interface{}
		if err := json.Unmarshal([]byte(data), &chunkData); err != nil {
			continue
		}
		r.writeResponseAPIStreamDeltaEvents(&out, ctx, chunkData)
	}

	return out.Bytes()
}

func (r *OpenAIRouter) writeResponseAPIStreamDeltaEvents(
	out *bytes.Buffer,
	ctx *RequestContext,
	chunkData map[string]interface{},
) {
	deltas := responseAPITextDeltas(chunkData)
	if len(deltas) == 0 {
		return
	}

	r.ensureResponseAPIStreamStarted(out, ctx)
	for _, delta := range deltas {
		writeResponseAPIStreamEvent(out, "response.output_text.delta", map[string]interface{}{
			"type":          "response.output_text.delta",
			"item_id":       ctx.ResponseAPIStreamItemID,
			"output_index":  0,
			"content_index": 0,
			"delta":         delta,
		})
	}
}

func (r *OpenAIRouter) ensureResponseAPIStreamStarted(out *bytes.Buffer, ctx *RequestContext) {
	if ctx.ResponseAPIStreamStarted {
		return
	}

	ctx.ResponseAPIStreamStarted = true
	if ctx.ResponseAPIStreamItemID == "" {
		ctx.ResponseAPIStreamItemID = responseapi.GenerateItemID()
	}
	if ctx.ResponseAPICtx.GeneratedResponseID == "" {
		ctx.ResponseAPICtx.GeneratedResponseID = responseapi.GenerateResponseID()
	}
	if ctx.ResponseAPIStreamCreatedAt == 0 {
		ctx.ResponseAPIStreamCreatedAt = time.Now().Unix()
	}

	startedResponse := responseAPIStreamResponse(ctx, responseapi.StatusInProgress, []interface{}{}, "", nil)

	writeResponseAPIStreamEvent(out, "response.created", map[string]interface{}{
		"type":     "response.created",
		"response": startedResponse,
	})
	writeResponseAPIStreamEvent(out, "response.in_progress", map[string]interface{}{
		"type":     "response.in_progress",
		"response": startedResponse,
	})
	writeResponseAPIStreamEvent(out, "response.output_item.added", map[string]interface{}{
		"type":         "response.output_item.added",
		"output_index": 0,
		"item":         responseAPIStreamMessageItem(ctx, responseapi.StatusInProgress, ""),
	})
	writeResponseAPIStreamEvent(out, "response.content_part.added", map[string]interface{}{
		"type":          "response.content_part.added",
		"item_id":       ctx.ResponseAPIStreamItemID,
		"output_index":  0,
		"content_index": 0,
		"part": map[string]interface{}{
			"type":        responseapi.ContentTypeOutputText,
			"text":        "",
			"annotations": []interface{}{},
		},
	})
}

func (r *OpenAIRouter) writeResponseAPIStreamDoneEvents(out *bytes.Buffer, ctx *RequestContext) {
	r.ensureResponseAPIStreamStarted(out, ctx)
	text := ctx.StreamingContent
	usage := responseAPIStreamUsage(ctx)

	writeResponseAPIStreamEvent(out, "response.output_text.done", map[string]interface{}{
		"type":          "response.output_text.done",
		"item_id":       ctx.ResponseAPIStreamItemID,
		"output_index":  0,
		"content_index": 0,
		"text":          text,
	})
	writeResponseAPIStreamEvent(out, "response.content_part.done", map[string]interface{}{
		"type":          "response.content_part.done",
		"item_id":       ctx.ResponseAPIStreamItemID,
		"output_index":  0,
		"content_index": 0,
		"part": map[string]interface{}{
			"type":        responseapi.ContentTypeOutputText,
			"text":        text,
			"annotations": []interface{}{},
		},
	})
	writeResponseAPIStreamEvent(out, "response.output_item.done", map[string]interface{}{
		"type":         "response.output_item.done",
		"output_index": 0,
		"item":         responseAPIStreamMessageItem(ctx, responseapi.StatusCompleted, text),
	})
	writeResponseAPIStreamEvent(out, "response.completed", map[string]interface{}{
		"type":     "response.completed",
		"response": responseAPIStreamCompletedResponse(ctx, text, usage),
	})
}

func responseAPITextDeltas(chunkData map[string]interface{}) []string {
	choices, ok := chunkData["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return nil
	}

	deltas := make([]string, 0, len(choices))
	for _, rawChoice := range choices {
		choice, ok := rawChoice.(map[string]interface{})
		if !ok {
			continue
		}
		delta, ok := choice["delta"].(map[string]interface{})
		if !ok {
			continue
		}
		content, ok := delta["content"].(string)
		if ok && content != "" {
			deltas = append(deltas, content)
		}
	}
	return deltas
}

func responseAPIStreamModel(ctx *RequestContext) string {
	if model, ok := ctx.StreamingMetadata["model"].(string); ok && model != "" {
		return model
	}
	if ctx.RequestModel != "" {
		return ctx.RequestModel
	}
	if ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.OriginalRequest != nil {
		return ctx.ResponseAPICtx.OriginalRequest.Model
	}
	return ""
}

func responseAPIStreamMessageItem(ctx *RequestContext, status string, text string) map[string]interface{} {
	content := []interface{}{}
	if status == responseapi.StatusCompleted || text != "" {
		content = append(content, map[string]interface{}{
			"type":        responseapi.ContentTypeOutputText,
			"text":        text,
			"annotations": []interface{}{},
		})
	}

	return map[string]interface{}{
		"id":      ctx.ResponseAPIStreamItemID,
		"type":    responseapi.ItemTypeMessage,
		"role":    responseapi.RoleAssistant,
		"status":  status,
		"content": content,
	}
}

func responseAPIStreamCompletedResponse(
	ctx *RequestContext,
	text string,
	usage map[string]interface{},
) map[string]interface{} {
	output := []interface{}{responseAPIStreamMessageItem(ctx, responseapi.StatusCompleted, text)}
	return responseAPIStreamResponse(ctx, responseapi.StatusCompleted, output, text, usage)
}

func responseAPIStreamResponse(
	ctx *RequestContext,
	status string,
	output []interface{},
	text string,
	usage map[string]interface{},
) map[string]interface{} {
	req := ctx.ResponseAPICtx.OriginalRequest
	store := true
	if req.Store != nil {
		store = *req.Store
	}
	metadata := req.Metadata
	if metadata == nil {
		metadata = map[string]string{}
	}

	response := map[string]interface{}{
		"id":                   ctx.ResponseAPICtx.GeneratedResponseID,
		"object":               "response",
		"created_at":           ctx.ResponseAPIStreamCreatedAt,
		"model":                responseAPIStreamModel(ctx),
		"status":               status,
		"output":               output,
		"previous_response_id": nil,
		"error":                nil,
		"incomplete_details":   nil,
		"instructions":         nil,
		"max_output_tokens":    req.MaxOutputTokens,
		"parallel_tool_calls":  true,
		"reasoning": map[string]interface{}{
			"effort":  nil,
			"summary": nil,
		},
		"store":       store,
		"temperature": responseAPIStreamTemperature(req),
		"text": map[string]interface{}{
			"format": map[string]interface{}{
				"type": "text",
			},
		},
		"tool_choice": req.ToolChoice,
		"tools":       req.Tools,
		"top_p":       responseAPIStreamTopP(req),
		"truncation":  "disabled",
		"usage":       usage,
		"user":        nil,
		"metadata":    metadata,
	}
	if req.PreviousResponseID != "" {
		response["previous_response_id"] = req.PreviousResponseID
	}
	if req.Instructions != "" {
		response["instructions"] = req.Instructions
	}
	if req.ToolChoice == nil {
		response["tool_choice"] = "auto"
	}
	if req.Tools == nil {
		response["tools"] = []interface{}{}
	}
	if status == responseapi.StatusCompleted {
		response["output_text"] = text
	}
	if req.ConversationID != "" {
		response["conversation_id"] = req.ConversationID
	}
	return response
}

func responseAPIStreamUsage(ctx *RequestContext) map[string]interface{} {
	usage := extractStreamingUsage(ctx)
	if usage.PromptTokens == 0 && usage.CompletionTokens == 0 && usage.TotalTokens == 0 {
		return nil
	}

	inputTokens := int(usage.PromptTokens)
	cachedTokens := 0
	if cached, ok := streamingCachedPromptTokens(ctx, inputTokens); ok {
		cachedTokens = cached
	}

	reasoningTokens := 0
	if usageMap, ok := ctx.StreamingMetadata["usage"].(map[string]interface{}); ok {
		if details, ok := usageMap["completion_tokens_details"].(map[string]interface{}); ok {
			if rawReasoningTokens, ok := details["reasoning_tokens"].(float64); ok {
				reasoningTokens = int(rawReasoningTokens)
			}
		}
	}

	return map[string]interface{}{
		"input_tokens": inputTokens,
		"input_tokens_details": map[string]interface{}{
			"cached_tokens": cachedTokens,
		},
		"output_tokens": int(usage.CompletionTokens),
		"output_tokens_details": map[string]interface{}{
			"reasoning_tokens": reasoningTokens,
		},
		"total_tokens": int(usage.TotalTokens),
	}
}

func responseAPIStreamTemperature(req *responseapi.ResponseAPIRequest) float64 {
	if req.Temperature != nil {
		return *req.Temperature
	}
	return 1.0
}

func responseAPIStreamTopP(req *responseapi.ResponseAPIRequest) float64 {
	if req.TopP != nil {
		return *req.TopP
	}
	return 1.0
}

func writeResponseAPIStreamEvent(out *bytes.Buffer, event string, payload map[string]interface{}) {
	body, err := json.Marshal(payload)
	if err != nil {
		return
	}
	fmt.Fprintf(out, "event: %s\ndata: %s\n\n", event, body)
}
