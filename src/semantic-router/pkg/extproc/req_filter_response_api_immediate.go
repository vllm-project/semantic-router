package extproc

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

func (r *OpenAIRouter) createResponseAPICacheHitResponse(
	ctx *RequestContext,
	cachedResponse []byte,
	category string,
	decisionName string,
	matchedKeywords []string,
	similarity float32,
) (*ext_proc.ProcessingResponse, bool) {
	resp, ok := r.responseAPIResponseFromChatCompletion(ctx, cachedResponse)
	if !ok {
		return nil, false
	}

	var procResp *ext_proc.ProcessingResponse
	if ctx.ExpectStreamingResponse {
		procResp = r.createSSEResponseWithBody(200, responseAPIStreamingSSEFromResponse(resp), headers.ResponsePathCache)
	} else {
		body, err := json.Marshal(resp)
		if err != nil {
			return nil, false
		}
		procResp = r.createJSONResponseWithBody(200, body, headers.ResponsePathCache)
	}
	appendResponseAPIImmediateHeaders(procResp, responseAPICacheHitHeaders(category, decisionName, matchedKeywords, similarity)...)
	return procResp, true
}

func (r *OpenAIRouter) createResponseAPIFastResponse(
	ctx *RequestContext,
	message string,
	decisionName string,
) (*ext_proc.ProcessingResponse, bool) {
	resp, ok := r.responseAPIResponseFromChatCompletion(ctx, responseAPIChatCompletionFastBody(message))
	if !ok {
		return nil, false
	}

	var procResp *ext_proc.ProcessingResponse
	if ctx.ExpectStreamingResponse {
		procResp = r.createSSEResponseWithBody(200, responseAPIStreamingSSEFromResponse(resp), headers.ResponsePathFastResponse)
	} else {
		body, err := json.Marshal(resp)
		if err != nil {
			return nil, false
		}
		procResp = r.createJSONResponseWithBody(200, body, headers.ResponsePathFastResponse)
	}
	appendResponseAPIImmediateHeaders(procResp, responseAPIFastResponseHeaders(decisionName)...)
	return procResp, true
}

func (r *OpenAIRouter) responseAPIResponseFromChatCompletion(
	ctx *RequestContext,
	body []byte,
) (*responseapi.ResponseAPIResponse, bool) {
	if r == nil || r.ResponseAPIFilter == nil || ctx == nil || ctx.ResponseAPICtx == nil {
		return nil, false
	}
	translatedBody, err := r.ResponseAPIFilter.TranslateResponse(ctx.TraceContext, ctx.ResponseAPICtx, body)
	if err != nil {
		return nil, false
	}
	var resp responseapi.ResponseAPIResponse
	if err := json.Unmarshal(translatedBody, &resp); err != nil || resp.ID == "" {
		return nil, false
	}
	return &resp, true
}

func responseAPIChatCompletionFastBody(message string) []byte {
	now := time.Now().Unix()
	body := map[string]interface{}{
		"id":      fmt.Sprintf("chatcmpl-fast-%d", now),
		"object":  "chat.completion",
		"created": now,
		"model":   "router",
		"choices": []map[string]interface{}{{
			"index": 0,
			"message": map[string]interface{}{
				"role":    "assistant",
				"content": message,
			},
			"finish_reason": "stop",
		}},
		"usage": map[string]interface{}{
			"prompt_tokens":     0,
			"completion_tokens": 0,
			"total_tokens":      0,
		},
	}
	encoded, _ := json.Marshal(body)
	return encoded
}

func responseAPIStreamingSSEFromResponse(resp *responseapi.ResponseAPIResponse) []byte {
	if resp == nil {
		return nil
	}
	var out bytes.Buffer
	startedResponse := responseAPIResponseEventMap(resp, responseapi.StatusInProgress, nil, "")
	writeResponseAPIStreamEvent(&out, "response.created", map[string]interface{}{
		"type":     "response.created",
		"response": startedResponse,
	})
	writeResponseAPIStreamEvent(&out, "response.in_progress", map[string]interface{}{
		"type":     "response.in_progress",
		"response": startedResponse,
	})

	for outputIndex, item := range resp.Output {
		responseAPIWriteImmediateOutputItemEvents(&out, outputIndex, item)
	}

	writeResponseAPIStreamEvent(&out, "response.completed", map[string]interface{}{
		"type":     "response.completed",
		"response": responseAPIResponseEventMap(resp, responseapi.StatusCompleted, resp.Output, resp.OutputText),
	})
	return out.Bytes()
}

func responseAPIWriteImmediateOutputItemEvents(out *bytes.Buffer, outputIndex int, item responseapi.OutputItem) {
	inProgressItem := responseAPIOutputItemEventMap(item, responseapi.StatusInProgress)
	writeResponseAPIStreamEvent(out, "response.output_item.added", map[string]interface{}{
		"type":         "response.output_item.added",
		"output_index": outputIndex,
		"item":         inProgressItem,
	})

	switch item.Type {
	case responseapi.ItemTypeFunctionCall:
		if item.Arguments != "" {
			writeResponseAPIStreamEvent(out, "response.function_call_arguments.delta", map[string]interface{}{
				"type":         "response.function_call_arguments.delta",
				"item_id":      item.ID,
				"output_index": outputIndex,
				"delta":        item.Arguments,
			})
		}
		writeResponseAPIStreamEvent(out, "response.function_call_arguments.done", map[string]interface{}{
			"type":         "response.function_call_arguments.done",
			"item_id":      item.ID,
			"output_index": outputIndex,
			"arguments":    item.Arguments,
		})
	default:
		responseAPIWriteImmediateMessageEvents(out, outputIndex, item)
	}

	writeResponseAPIStreamEvent(out, "response.output_item.done", map[string]interface{}{
		"type":         "response.output_item.done",
		"output_index": outputIndex,
		"item":         responseAPIOutputItemEventMap(item, responseapi.StatusCompleted),
	})
}

func responseAPIWriteImmediateMessageEvents(out *bytes.Buffer, outputIndex int, item responseapi.OutputItem) {
	for contentIndex, part := range item.Content {
		partMap := responseAPIContentPartEventMap(part)
		writeResponseAPIStreamEvent(out, "response.content_part.added", map[string]interface{}{
			"type":          "response.content_part.added",
			"item_id":       item.ID,
			"output_index":  outputIndex,
			"content_index": contentIndex,
			"part":          responseAPIEmptyContentPartEventMap(part),
		})
		switch part.Type {
		case "refusal":
			if part.Text != "" {
				writeResponseAPIStreamEvent(out, "response.refusal.delta", map[string]interface{}{
					"type":          "response.refusal.delta",
					"item_id":       item.ID,
					"output_index":  outputIndex,
					"content_index": contentIndex,
					"delta":         part.Text,
				})
			}
			writeResponseAPIStreamEvent(out, "response.refusal.done", map[string]interface{}{
				"type":          "response.refusal.done",
				"item_id":       item.ID,
				"output_index":  outputIndex,
				"content_index": contentIndex,
				"refusal":       part.Text,
			})
		default:
			if part.Text != "" {
				writeResponseAPIStreamEvent(out, "response.output_text.delta", map[string]interface{}{
					"type":          "response.output_text.delta",
					"item_id":       item.ID,
					"output_index":  outputIndex,
					"content_index": contentIndex,
					"delta":         part.Text,
				})
			}
			writeResponseAPIStreamEvent(out, "response.output_text.done", map[string]interface{}{
				"type":          "response.output_text.done",
				"item_id":       item.ID,
				"output_index":  outputIndex,
				"content_index": contentIndex,
				"text":          part.Text,
			})
		}
		writeResponseAPIStreamEvent(out, "response.content_part.done", map[string]interface{}{
			"type":          "response.content_part.done",
			"item_id":       item.ID,
			"output_index":  outputIndex,
			"content_index": contentIndex,
			"part":          partMap,
		})
	}
}

func responseAPIResponseEventMap(
	resp *responseapi.ResponseAPIResponse,
	status string,
	output []responseapi.OutputItem,
	outputText string,
) map[string]interface{} {
	body := responseAPIMarshalMap(resp)
	body["status"] = status
	if output == nil {
		body["output"] = []interface{}{}
	} else {
		body["output"] = output
	}
	if status == responseapi.StatusCompleted {
		body["output_text"] = outputText
	} else {
		delete(body, "output_text")
	}
	return body
}

func responseAPIOutputItemEventMap(item responseapi.OutputItem, status string) map[string]interface{} {
	body := responseAPIMarshalMap(item)
	body["status"] = status
	if item.Type == responseapi.ItemTypeMessage && status == responseapi.StatusInProgress {
		body["content"] = []interface{}{}
	}
	return body
}

func responseAPIContentPartEventMap(part responseapi.ContentPart) map[string]interface{} {
	body := responseAPIMarshalMap(part)
	if part.Type == responseapi.ContentTypeOutputText {
		body["annotations"] = []interface{}{}
	}
	if part.Type == "refusal" && part.Text != "" {
		body["refusal"] = part.Text
		delete(body, "text")
	}
	return body
}

func responseAPIEmptyContentPartEventMap(part responseapi.ContentPart) map[string]interface{} {
	if part.Type == "refusal" {
		return map[string]interface{}{"type": "refusal", "refusal": ""}
	}
	return map[string]interface{}{
		"type":        responseapi.ContentTypeOutputText,
		"text":        "",
		"annotations": []interface{}{},
	}
}

func responseAPIMarshalMap(v interface{}) map[string]interface{} {
	body, err := json.Marshal(v)
	if err != nil {
		return map[string]interface{}{}
	}
	var out map[string]interface{}
	if err := json.Unmarshal(body, &out); err != nil {
		return map[string]interface{}{}
	}
	return out
}

func appendResponseAPIImmediateHeaders(procResp *ext_proc.ProcessingResponse, extra ...*core.HeaderValueOption) {
	if procResp == nil || procResp.GetImmediateResponse() == nil {
		return
	}
	headers := procResp.GetImmediateResponse().Headers
	if headers == nil {
		headers = &ext_proc.HeaderMutation{}
		procResp.GetImmediateResponse().Headers = headers
	}
	headers.SetHeaders = append(headers.SetHeaders, extra...)
}

func responseAPICacheHitHeaders(
	category string,
	decisionName string,
	matchedKeywords []string,
	similarity float32,
) []*core.HeaderValueOption {
	setHeaders := []*core.HeaderValueOption{
		rawHeaderOption(headers.VSRCacheHit, "true"),
		rawHeaderOption(headers.VSRSelectedDecision, decisionName),
	}
	if category != "" {
		setHeaders = append(setHeaders, rawHeaderOption(headers.VSRSelectedCategory, category))
	}
	if similarity > 0 {
		setHeaders = append(setHeaders, rawHeaderOption("x-vsr-cache-similarity", fmt.Sprintf("%.4f", similarity)))
	}
	if len(matchedKeywords) > 0 {
		setHeaders = append(setHeaders, rawHeaderOption(headers.VSRMatchedKeywords, strings.Join(matchedKeywords, ",")))
	}
	return setHeaders
}

func responseAPIFastResponseHeaders(decisionName string) []*core.HeaderValueOption {
	return []*core.HeaderValueOption{
		rawHeaderOption(headers.VSRSelectedDecision, decisionName),
		rawHeaderOption(headers.VSRFastResponse, "true"),
	}
}

func rawHeaderOption(key string, value string) *core.HeaderValueOption {
	return &core.HeaderValueOption{
		Header: &core.HeaderValue{
			Key:      key,
			RawValue: []byte(value),
		},
	}
}
