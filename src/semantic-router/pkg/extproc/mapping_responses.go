package extproc

import (
	"encoding/json"
	"fmt"
	"strings"
)

// mapResponsesRequestToChatCompletions converts a minimal OpenAI Responses API request
// into a legacy Chat Completions request JSON. Supports only text input for PR1.
func mapResponsesRequestToChatCompletions(original []byte) ([]byte, error) {
	var req map[string]interface{}
	if err := json.Unmarshal(original, &req); err != nil {
		return nil, err
	}

	// Extract model
	model, _ := req["model"].(string)
	if model == "" {
		return nil, fmt.Errorf("missing model")
	}

	// Derive user content
	var userContent string
	if input, ok := req["input"]; ok {
		switch v := input.(type) {
		case string:
			userContent = v
		case []interface{}:
			// Join any string elements; ignore non-string for now
			var parts []string
			for _, it := range v {
				if s, ok := it.(string); ok {
					parts = append(parts, s)
				} else if m, ok := it.(map[string]interface{}); ok {
					// Try common shapes: {type:"input_text"|"text", text:"..."}
					if t, _ := m["type"].(string); t == "input_text" || t == "text" {
						if txt, _ := m["text"].(string); txt != "" {
							parts = append(parts, txt)
						}
					}
				}
			}
			userContent = strings.TrimSpace(strings.Join(parts, " "))
		default:
			// unsupported multimodal
			return nil, fmt.Errorf("unsupported input type")
		}
	} else if msgs, ok := req["messages"].([]interface{}); ok {
		// Fallback: if caller already provided messages, pass them through
		// This enables easy migration from chat/completions
		mapped := map[string]interface{}{
			"model":    model,
			"messages": msgs,
		}
		// Map basic params
		if v, ok := req["temperature"]; ok {
			mapped["temperature"] = v
		}
		if v, ok := req["top_p"]; ok {
			mapped["top_p"] = v
		}
		if v, ok := req["max_output_tokens"]; ok {
			mapped["max_tokens"] = v
		}
		return json.Marshal(mapped)
	}

	if userContent == "" {
		return nil, fmt.Errorf("empty input")
	}

	// Build minimal Chat Completions request
	mapped := map[string]interface{}{
		"model": model,
		"messages": []map[string]interface{}{
			{"role": "user", "content": userContent},
		},
	}
	// Map basic params
	if v, ok := req["temperature"]; ok {
		mapped["temperature"] = v
	}
	if v, ok := req["top_p"]; ok {
		mapped["top_p"] = v
	}
	if v, ok := req["max_output_tokens"]; ok {
		mapped["max_tokens"] = v
	}

	return json.Marshal(mapped)
}

// mapChatCompletionToResponses converts an OpenAI ChatCompletion JSON
// into a minimal Responses API JSON (non-streaming only) for PR1.
func mapChatCompletionToResponses(chatCompletionJSON []byte) ([]byte, error) {
	var parsed struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		Model   string `json:"model"`
		Choices []struct {
			Index        int    `json:"index"`
			FinishReason string `json:"finish_reason"`
			Message      struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(chatCompletionJSON, &parsed); err != nil {
		return nil, err
	}

	content := ""
	stopReason := "stop"
	if len(parsed.Choices) > 0 {
		content = parsed.Choices[0].Message.Content
		if parsed.Choices[0].FinishReason != "" {
			stopReason = parsed.Choices[0].FinishReason
		}
	}

	out := map[string]interface{}{
		"id":      parsed.ID,
		"object":  "response",
		"created": parsed.Created,
		"model":   parsed.Model,
		"output": []map[string]interface{}{
			{"type": "message", "role": "assistant", "content": content},
		},
		"stop_reason": stopReason,
		"usage": map[string]int{
			"input_tokens":  parsed.Usage.PromptTokens,
			"output_tokens": parsed.Usage.CompletionTokens,
			"total_tokens":  parsed.Usage.TotalTokens,
		},
	}

	return json.Marshal(out)
}

// translateSSEChunkToResponses converts a single OpenAI chat.completion.chunk SSE payload
// (the JSON after "data: ") into Responses SSE events (delta/stop). Returns empty when not applicable.
func translateSSEChunkToResponses(chunk []byte) ([][]byte, bool) {
	// Expect chunk JSON like {"id":"...","object":"chat.completion.chunk","created":...,"model":"...","choices":[{"index":0,"delta":{"role":"assistant","content":"..."},"finish_reason":null}]}
	var parsed map[string]interface{}
	if err := json.Unmarshal(chunk, &parsed); err != nil {
		return nil, false
	}
	if parsed["object"] != "chat.completion.chunk" {
		return nil, false
	}

	created, _ := parsed["created"].(float64)
	// Emit a created event only once per stream (handled by caller)

	// Extract content delta and finish_reason
	var deltaText string
	var finish string
	if arr, ok := parsed["choices"].([]interface{}); ok && len(arr) > 0 {
		if ch, ok := arr[0].(map[string]interface{}); ok {
			if fr, ok := ch["finish_reason"].(string); ok && fr != "" {
				finish = fr
			}
			if d, ok := ch["delta"].(map[string]interface{}); ok {
				if c, ok := d["content"].(string); ok {
					deltaText = c
				}
			}
		}
	}

	var events [][]byte
	if deltaText != "" {
		ev := map[string]interface{}{
			"type":  "response.output_text.delta",
			"delta": deltaText,
		}
		if created > 0 {
			ev["created"] = int64(created)
		}
		b, _ := json.Marshal(ev)
		events = append(events, b)
	}

	if finish != "" {
		ev := map[string]interface{}{
			"type":        "response.completed",
			"stop_reason": finish,
		}
		b, _ := json.Marshal(ev)
		events = append(events, b)
	}

	if len(events) == 0 {
		return nil, false
	}
	return events, true
}
