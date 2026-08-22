package extproc

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	anthropicapi "github.com/vllm-project/semantic-router/src/semantic-router/pkg/anthropic"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ir"
	httputil "github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/http"
)

func createAnthropicFastResponse(
	ctx *RequestContext,
	message string,
	decisionName string,
) *ext_proc.ProcessingResponse {
	streaming := ctx != nil && ctx.ExpectStreamingResponse
	response := httputil.CreateFastResponse(message, streaming, decisionName)
	immediate := response.GetImmediateResponse()
	if immediate == nil {
		return response
	}
	model := anthropicFastResponseModel(ctx)
	if streaming {
		immediate.Body = buildAnthropicFastResponseStream(message, model)
		return response
	}
	var extensions *ir.IRExtensions
	if ctx != nil {
		extensions = ctx.IRExtensions
	}
	body, err := anthropicapi.EmitAnthropicResponse(
		immediate.Body,
		extensions,
		model,
	)
	if err == nil {
		immediate.Body = body
	}
	return response
}

func anthropicFastResponseModel(ctx *RequestContext) string {
	if ctx == nil {
		return "vllm-semantic-router"
	}
	if model := strings.TrimSpace(ctx.VSRSelectedModel); model != "" {
		return model
	}
	if model := strings.TrimSpace(ctx.RequestModel); model != "" {
		return model
	}
	return "vllm-semantic-router"
}

func buildAnthropicFastResponseStream(message string, model string) []byte {
	id := fmt.Sprintf("msg_fast_%d", time.Now().UnixNano())
	events := []struct {
		event string
		data  any
	}{
		{
			event: "message_start",
			data: map[string]any{
				"type": "message_start",
				"message": map[string]any{
					"id":            id,
					"type":          "message",
					"role":          "assistant",
					"content":       []any{},
					"model":         model,
					"stop_reason":   nil,
					"stop_sequence": nil,
					"usage": map[string]int{
						"input_tokens":  0,
						"output_tokens": 0,
					},
				},
			},
		},
		{
			event: "content_block_start",
			data: map[string]any{
				"type":  "content_block_start",
				"index": 0,
				"content_block": map[string]string{
					"type": "text",
					"text": "",
				},
			},
		},
		{
			event: "content_block_delta",
			data: map[string]any{
				"type":  "content_block_delta",
				"index": 0,
				"delta": map[string]string{
					"type": "text_delta",
					"text": message,
				},
			},
		},
		{
			event: "content_block_stop",
			data:  map[string]any{"type": "content_block_stop", "index": 0},
		},
		{
			event: "message_delta",
			data: map[string]any{
				"type": "message_delta",
				"delta": map[string]any{
					"stop_reason":   "end_turn",
					"stop_sequence": nil,
				},
				"usage": map[string]int{"output_tokens": 0},
			},
		},
		{
			event: "message_stop",
			data:  map[string]string{"type": "message_stop"},
		},
	}
	var builder strings.Builder
	for _, item := range events {
		data, err := json.Marshal(item.data)
		if err != nil {
			continue
		}
		fmt.Fprintf(&builder, "event: %s\ndata: %s\n\n", item.event, data)
	}
	return []byte(builder.String())
}
