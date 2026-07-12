package extproc

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
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
	r.writeResponseAPIReasoningDeltaEvents(out, ctx, chunkData)

	deltas := responseAPITextDeltas(chunkData)
	if len(deltas) > 0 {
		r.ensureResponseAPIStreamStarted(out, ctx)
		for _, delta := range deltas {
			writeResponseAPIStreamEvent(out, "response.output_text.delta", map[string]interface{}{
				"type":          "response.output_text.delta",
				"item_id":       ctx.ResponseAPIStreamItemID,
				"output_index":  responseAPIMessageOutputIndex(ctx),
				"content_index": 0,
				"delta":         delta,
			})
		}
	}

	r.writeResponseAPIRefusalDeltaEvents(out, ctx, chunkData)
	r.writeResponseAPIToolCallDeltaEvents(out, ctx, chunkData)
}

func (r *OpenAIRouter) ensureResponseAPIResponseStarted(out *bytes.Buffer, ctx *RequestContext) {
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
}

func (r *OpenAIRouter) ensureResponseAPIStreamStarted(out *bytes.Buffer, ctx *RequestContext) {
	r.ensureResponseAPIResponseStarted(out, ctx)
	if ctx.ResponseAPIStreamMessageStarted {
		return
	}

	ctx.ResponseAPIStreamMessageStarted = true
	ctx.ResponseAPIStreamMessageOutputIndex = responseAPIAssignOutputIndex(ctx)
	writeResponseAPIStreamEvent(out, "response.output_item.added", map[string]interface{}{
		"type":         "response.output_item.added",
		"output_index": responseAPIMessageOutputIndex(ctx),
		"item":         responseAPIStreamMessageItem(ctx, responseapi.StatusInProgress, ""),
	})
	writeResponseAPIStreamEvent(out, "response.content_part.added", map[string]interface{}{
		"type":          "response.content_part.added",
		"item_id":       ctx.ResponseAPIStreamItemID,
		"output_index":  responseAPIMessageOutputIndex(ctx),
		"content_index": 0,
		"part": map[string]interface{}{
			"type":        responseapi.ContentTypeOutputText,
			"text":        "",
			"annotations": []interface{}{},
		},
	})
}

func (r *OpenAIRouter) writeResponseAPIStreamDoneEvents(out *bytes.Buffer, ctx *RequestContext) {
	if responseAPIStreamNeedsMessageItem(ctx) {
		r.ensureResponseAPIStreamStarted(out, ctx)
	} else {
		r.ensureResponseAPIResponseStarted(out, ctx)
	}
	text := ctx.StreamingContent
	usage := responseAPIStreamUsage(ctx)

	r.writeResponseAPIReasoningDoneEvents(out, ctx)
	r.writeResponseAPIMessageDoneEvents(out, ctx, text)
	r.writeResponseAPIToolCallDoneEvents(out, ctx)
	writeResponseAPIStreamEvent(out, "response.completed", map[string]interface{}{
		"type":     "response.completed",
		"response": responseAPIStreamCompletedResponse(ctx, text, usage),
	})
}

func responseAPIStreamNeedsMessageItem(ctx *RequestContext) bool {
	if ctx == nil {
		return true
	}
	return ctx.ResponseAPIStreamMessageStarted || ctx.StreamingContent != "" || ctx.StreamingRefusal != "" || len(ctx.StreamingToolCalls) == 0
}

func (r *OpenAIRouter) writeResponseAPIMessageDoneEvents(out *bytes.Buffer, ctx *RequestContext, text string) {
	if !ctx.ResponseAPIStreamMessageStarted {
		return
	}
	if text != "" {
		writeResponseAPIStreamEvent(out, "response.output_text.done", map[string]interface{}{
			"type":          "response.output_text.done",
			"item_id":       ctx.ResponseAPIStreamItemID,
			"output_index":  responseAPIMessageOutputIndex(ctx),
			"content_index": 0,
			"text":          text,
		})
	}
	if ctx.StreamingRefusal != "" {
		writeResponseAPIStreamEvent(out, "response.refusal.done", map[string]interface{}{
			"type":          "response.refusal.done",
			"item_id":       ctx.ResponseAPIStreamItemID,
			"output_index":  responseAPIMessageOutputIndex(ctx),
			"content_index": 0,
			"refusal":       ctx.StreamingRefusal,
		})
	}
	writeResponseAPIStreamEvent(out, "response.content_part.done", map[string]interface{}{
		"type":          "response.content_part.done",
		"item_id":       ctx.ResponseAPIStreamItemID,
		"output_index":  responseAPIMessageOutputIndex(ctx),
		"content_index": 0,
		"part":          responseAPIStreamMessageContentPart(ctx, text),
	})
	writeResponseAPIStreamEvent(out, "response.output_item.done", map[string]interface{}{
		"type":         "response.output_item.done",
		"output_index": responseAPIMessageOutputIndex(ctx),
		"item":         responseAPIStreamMessageItem(ctx, responseapi.StatusCompleted, text),
	})
}

func responseAPIAssignOutputIndex(ctx *RequestContext) int {
	index := ctx.ResponseAPIStreamNextOutputIndex
	ctx.ResponseAPIStreamNextOutputIndex++
	return index
}

func responseAPIMessageOutputIndex(ctx *RequestContext) int {
	return ctx.ResponseAPIStreamMessageOutputIndex
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

func (r *OpenAIRouter) writeResponseAPIRefusalDeltaEvents(
	out *bytes.Buffer,
	ctx *RequestContext,
	chunkData map[string]interface{},
) {
	for _, delta := range responseAPIStringDeltas(chunkData, "refusal") {
		r.ensureResponseAPIStreamStarted(out, ctx)
		if !ctx.ResponseAPIStreamRefusalStarted {
			ctx.ResponseAPIStreamRefusalStarted = true
		}
		writeResponseAPIStreamEvent(out, "response.refusal.delta", map[string]interface{}{
			"type":          "response.refusal.delta",
			"item_id":       ctx.ResponseAPIStreamItemID,
			"output_index":  responseAPIMessageOutputIndex(ctx),
			"content_index": 0,
			"delta":         delta,
		})
	}
}

func (r *OpenAIRouter) writeResponseAPIReasoningDeltaEvents(
	out *bytes.Buffer,
	ctx *RequestContext,
	chunkData map[string]interface{},
) {
	for _, delta := range responseAPIStringDeltas(chunkData, "reasoning_content") {
		r.ensureResponseAPIReasoningStarted(out, ctx)
		writeResponseAPIStreamEvent(out, "response.reasoning_text.delta", map[string]interface{}{
			"type":          "response.reasoning_text.delta",
			"item_id":       ctx.ResponseAPIStreamReasoningItemID,
			"output_index":  responseAPIReasoningOutputIndex(ctx),
			"content_index": 0,
			"delta":         delta,
		})
	}
}

func responseAPIStringDeltas(chunkData map[string]interface{}, field string) []string {
	choices := replayStreamingChoices(chunkData)
	if len(choices) == 0 {
		return nil
	}

	deltas := make([]string, 0, len(choices))
	for _, choice := range choices {
		delta, ok := choice["delta"].(map[string]interface{})
		if !ok {
			continue
		}
		value, ok := delta[field].(string)
		if ok && value != "" {
			deltas = append(deltas, value)
		}
	}
	return deltas
}

func (r *OpenAIRouter) ensureResponseAPIReasoningStarted(out *bytes.Buffer, ctx *RequestContext) {
	r.ensureResponseAPIResponseStarted(out, ctx)
	if ctx.ResponseAPIStreamReasoningStarted {
		return
	}

	ctx.ResponseAPIStreamReasoningStarted = true
	if ctx.ResponseAPIStreamReasoningItemID == "" {
		ctx.ResponseAPIStreamReasoningItemID = responseapi.GenerateItemID()
	}
	ctx.ResponseAPIStreamReasoningOutputIndex = responseAPIAssignOutputIndex(ctx)
	writeResponseAPIStreamEvent(out, "response.output_item.added", map[string]interface{}{
		"type":         "response.output_item.added",
		"output_index": responseAPIReasoningOutputIndex(ctx),
		"item": map[string]interface{}{
			"id":      ctx.ResponseAPIStreamReasoningItemID,
			"type":    "reasoning",
			"status":  responseapi.StatusInProgress,
			"summary": []interface{}{},
		},
	})
}

func (r *OpenAIRouter) writeResponseAPIReasoningDoneEvents(out *bytes.Buffer, ctx *RequestContext) {
	if !ctx.ResponseAPIStreamReasoningStarted {
		return
	}
	writeResponseAPIStreamEvent(out, "response.reasoning_text.done", map[string]interface{}{
		"type":          "response.reasoning_text.done",
		"item_id":       ctx.ResponseAPIStreamReasoningItemID,
		"output_index":  responseAPIReasoningOutputIndex(ctx),
		"content_index": 0,
		"text":          ctx.StreamingReasoning,
	})
	writeResponseAPIStreamEvent(out, "response.output_item.done", map[string]interface{}{
		"type":         "response.output_item.done",
		"output_index": responseAPIReasoningOutputIndex(ctx),
		"item": map[string]interface{}{
			"id":      ctx.ResponseAPIStreamReasoningItemID,
			"type":    "reasoning",
			"status":  responseapi.StatusCompleted,
			"summary": []interface{}{},
		},
	})
}

func responseAPIReasoningOutputIndex(ctx *RequestContext) int {
	return ctx.ResponseAPIStreamReasoningOutputIndex
}

func (r *OpenAIRouter) writeResponseAPIToolCallDeltaEvents(
	out *bytes.Buffer,
	ctx *RequestContext,
	chunkData map[string]interface{},
) {
	for _, choice := range replayStreamingChoices(chunkData) {
		for _, indexedToolCall := range replayStreamingToolCalls(choice) {
			index := replayStreamingToolCallIndex(indexedToolCall.rawIndex, indexedToolCall.toolCall)
			state := ctx.StreamingToolCalls[index]
			r.ensureResponseAPIToolCallStarted(out, ctx, index, state)
			if fn, ok := indexedToolCall.toolCall["function"].(map[string]interface{}); ok {
				if arguments, ok := fn["arguments"].(string); ok && arguments != "" {
					writeResponseAPIStreamEvent(out, "response.function_call_arguments.delta", map[string]interface{}{
						"type":         "response.function_call_arguments.delta",
						"item_id":      ctx.ResponseAPIStreamToolCallItemIDs[index],
						"output_index": responseAPIToolCallOutputIndex(ctx, index),
						"delta":        arguments,
					})
				}
			}
		}
	}
}

func (r *OpenAIRouter) ensureResponseAPIToolCallStarted(
	out *bytes.Buffer,
	ctx *RequestContext,
	index int,
	state *StreamingToolCallState,
) {
	r.ensureResponseAPIResponseStarted(out, ctx)
	if ctx.ResponseAPIStreamToolCallItemIDs == nil {
		ctx.ResponseAPIStreamToolCallItemIDs = make(map[int]string)
	}
	if ctx.ResponseAPIStreamToolCallOutputIndex == nil {
		ctx.ResponseAPIStreamToolCallOutputIndex = make(map[int]int)
	}
	if ctx.ResponseAPIStreamToolCallItemIDs[index] == "" {
		ctx.ResponseAPIStreamToolCallItemIDs[index] = responseapi.GenerateItemID()
		ctx.ResponseAPIStreamToolCallOutputIndex[index] = responseAPIAssignOutputIndex(ctx)
		writeResponseAPIStreamEvent(out, "response.output_item.added", map[string]interface{}{
			"type":         "response.output_item.added",
			"output_index": responseAPIToolCallOutputIndex(ctx, index),
			"item":         responseAPIStreamToolCallItem(ctx, index, state, responseapi.StatusInProgress),
		})
	}
}

func (r *OpenAIRouter) writeResponseAPIToolCallDoneEvents(out *bytes.Buffer, ctx *RequestContext) {
	for _, index := range responseAPISortedToolCallIndexes(ctx) {
		itemID := ctx.ResponseAPIStreamToolCallItemIDs[index]
		if itemID == "" {
			continue
		}
		state := ctx.StreamingToolCalls[index]
		writeResponseAPIStreamEvent(out, "response.function_call_arguments.done", map[string]interface{}{
			"type":         "response.function_call_arguments.done",
			"item_id":      itemID,
			"output_index": responseAPIToolCallOutputIndex(ctx, index),
			"arguments":    responseAPIToolCallArguments(state),
		})
		writeResponseAPIStreamEvent(out, "response.output_item.done", map[string]interface{}{
			"type":         "response.output_item.done",
			"output_index": responseAPIToolCallOutputIndex(ctx, index),
			"item":         responseAPIStreamToolCallItem(ctx, index, state, responseapi.StatusCompleted),
		})
	}
}

func responseAPIToolCallOutputIndex(ctx *RequestContext, index int) int {
	if ctx != nil && ctx.ResponseAPIStreamToolCallOutputIndex != nil {
		return ctx.ResponseAPIStreamToolCallOutputIndex[index]
	}
	return index
}

func responseAPISortedToolCallIndexes(ctx *RequestContext) []int {
	if ctx == nil || len(ctx.StreamingToolCalls) == 0 {
		return nil
	}
	indexes := make([]int, 0, len(ctx.StreamingToolCalls))
	for index := range ctx.StreamingToolCalls {
		indexes = append(indexes, index)
	}
	sort.Ints(indexes)
	return indexes
}

func responseAPIStreamToolCallItem(
	ctx *RequestContext,
	index int,
	state *StreamingToolCallState,
	status string,
) map[string]interface{} {
	return map[string]interface{}{
		"id":        ctx.ResponseAPIStreamToolCallItemIDs[index],
		"type":      responseapi.ItemTypeFunctionCall,
		"call_id":   responseAPIToolCallID(state),
		"name":      responseAPIToolCallName(state),
		"arguments": responseAPIToolCallArguments(state),
		"status":    status,
	}
}

func responseAPIToolCallID(state *StreamingToolCallState) string {
	if state == nil {
		return ""
	}
	return state.ID
}

func responseAPIToolCallName(state *StreamingToolCallState) string {
	if state == nil {
		return ""
	}
	return state.Name
}

func responseAPIToolCallArguments(state *StreamingToolCallState) string {
	if state == nil {
		return ""
	}
	return state.Arguments
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
	if status == responseapi.StatusCompleted || text != "" || ctx.StreamingRefusal != "" {
		content = append(content, responseAPIStreamMessageContentPart(ctx, text))
	}

	return map[string]interface{}{
		"id":      ctx.ResponseAPIStreamItemID,
		"type":    responseapi.ItemTypeMessage,
		"role":    responseapi.RoleAssistant,
		"status":  status,
		"content": content,
	}
}

func responseAPIStreamMessageContentPart(ctx *RequestContext, text string) map[string]interface{} {
	if ctx != nil && ctx.StreamingRefusal != "" && text == "" {
		return map[string]interface{}{
			"type":    "refusal",
			"refusal": ctx.StreamingRefusal,
		}
	}
	return map[string]interface{}{
		"type":        responseapi.ContentTypeOutputText,
		"text":        text,
		"annotations": []interface{}{},
	}
}

func responseAPIStreamCompletedResponse(
	ctx *RequestContext,
	text string,
	usage map[string]interface{},
) map[string]interface{} {
	output := responseAPIStreamOutput(ctx, text)
	return responseAPIStreamResponse(ctx, responseapi.StatusCompleted, output, text, usage)
}

func responseAPIStreamOutput(ctx *RequestContext, text string) []interface{} {
	indexed := []responseAPIIndexedOutput{}
	if responseAPIStreamNeedsMessageItem(ctx) {
		indexed = append(indexed, responseAPIIndexedOutput{
			index: responseAPIMessageOutputIndex(ctx),
			item:  responseAPIStreamMessageItem(ctx, responseapi.StatusCompleted, text),
		})
	}
	if ctx != nil && ctx.ResponseAPIStreamReasoningStarted {
		indexed = append(indexed, responseAPIIndexedOutput{
			index: responseAPIReasoningOutputIndex(ctx),
			item: map[string]interface{}{
				"id":      ctx.ResponseAPIStreamReasoningItemID,
				"type":    "reasoning",
				"status":  responseapi.StatusCompleted,
				"summary": []interface{}{},
			},
		})
	}
	for _, index := range responseAPISortedToolCallIndexes(ctx) {
		indexed = append(indexed, responseAPIIndexedOutput{
			index: responseAPIToolCallOutputIndex(ctx, index),
			item: responseAPIStreamToolCallItem(
				ctx,
				index,
				ctx.StreamingToolCalls[index],
				responseapi.StatusCompleted,
			),
		})
	}
	sort.Slice(indexed, func(i, j int) bool { return indexed[i].index < indexed[j].index })
	output := make([]interface{}, 0, len(indexed))
	for _, entry := range indexed {
		output = append(output, entry.item)
	}
	return output
}

type responseAPIIndexedOutput struct {
	index int
	item  interface{}
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
	if ctx.ResponseAPICtx.ConversationID != "" {
		response["conversation_id"] = ctx.ResponseAPICtx.ConversationID
	} else if req.ConversationID != "" {
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
	if cached, cachedReported, _, _ := streamingPromptTokenDetails(ctx, inputTokens); cachedReported {
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
