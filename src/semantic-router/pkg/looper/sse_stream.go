package looper

import (
	"encoding/json"
	"fmt"
)

func appendSSEDataLine(body []byte, payload map[string]interface{}) []byte {
	chunkJSON, _ := json.Marshal(payload)
	return append(body, []byte(fmt.Sprintf("data: %s\n\n", chunkJSON))...)
}

func appendSSEDone(body []byte) []byte {
	return append(body, []byte("data: [DONE]\n\n")...)
}

func chatCompletionChunkPayload(
	id string,
	created int64,
	model string,
	choice map[string]interface{},
	topLevel map[string]interface{},
) map[string]interface{} {
	payload := map[string]interface{}{
		"id":      id,
		"object":  "chat.completion.chunk",
		"created": created,
		"model":   model,
		"choices": []map[string]interface{}{choice},
	}
	for k, v := range topLevel {
		payload[k] = v
	}
	return payload
}

func buildSimulatedChatCompletionSSE(
	id string,
	created int64,
	model string,
	contentChunks []string,
	toolName, toolArgs, toolCallID string,
	hasToolCall bool,
) []byte {
	var body []byte
	roleChoice := map[string]interface{}{
		"index":         0,
		"delta":         map[string]interface{}{"role": "assistant"},
		"finish_reason": nil,
	}
	body = appendSSEDataLine(body, chatCompletionChunkPayload(id, created, model, roleChoice, nil))

	if hasToolCall {
		toolChoice := map[string]interface{}{
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
		}
		body = appendSSEDataLine(body, chatCompletionChunkPayload(id, created, model, toolChoice, nil))
	} else {
		for _, chunk := range contentChunks {
			contentChoice := map[string]interface{}{
				"index":         0,
				"delta":         map[string]interface{}{"content": chunk},
				"finish_reason": nil,
			}
			body = appendSSEDataLine(body, chatCompletionChunkPayload(id, created, model, contentChoice, nil))
		}
	}

	finalReason := "stop"
	if hasToolCall {
		finalReason = "tool_calls"
	}
	finalChoice := map[string]interface{}{
		"index":         0,
		"delta":         map[string]interface{}{},
		"finish_reason": finalReason,
	}
	body = appendSSEDataLine(body, chatCompletionChunkPayload(id, created, model, finalChoice, nil))
	return appendSSEDone(body)
}

func buildReMoMStreamingSSE(
	id string,
	created int64,
	final IntermediateResp,
	allRoundResponses []RoundResponse,
	includeIntermediate bool,
) []byte {
	var body []byte
	roleChoice := map[string]interface{}{
		"index":         0,
		"delta":         map[string]interface{}{"role": "assistant"},
		"finish_reason": nil,
	}
	var extra map[string]interface{}
	if includeIntermediate {
		extra = map[string]interface{}{"reasoning_mom_responses": allRoundResponses}
	}
	body = appendSSEDataLine(body, chatCompletionChunkPayload(id, created, final.Model, roleChoice, extra))

	contentChoice := map[string]interface{}{
		"index":         0,
		"delta":         map[string]interface{}{"content": final.Content},
		"finish_reason": nil,
	}
	body = appendSSEDataLine(body, chatCompletionChunkPayload(id, created, final.Model, contentChoice, nil))

	finalChoice := map[string]interface{}{
		"index":         0,
		"delta":         map[string]interface{}{},
		"finish_reason": "stop",
	}
	body = appendSSEDataLine(body, chatCompletionChunkPayload(id, created, final.Model, finalChoice, nil))
	return appendSSEDone(body)
}

func streamingLooperResponse(body []byte, model string, modelsUsed []string, iterations int, algorithmType string) *Response {
	return &Response{
		Body:          body,
		ContentType:   "text/event-stream",
		Model:         model,
		ModelsUsed:    modelsUsed,
		Iterations:    iterations,
		AlgorithmType: algorithmType,
	}
}
