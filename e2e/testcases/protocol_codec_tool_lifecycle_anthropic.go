package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
)

func runAnthropicClientToolLifecycle(
	ctx context.Context,
	session *fixtures.ServiceSession,
	model string,
) error {
	tool := map[string]any{
		"name": "lookup", "description": "Look up a value",
		"input_schema": map[string]any{
			"type":       "object",
			"properties": map[string]any{"query": map[string]any{"type": "string"}},
			"required":   []string{"query"},
		},
	}
	firstBody := map[string]any{
		"model": model, "max_tokens": 64,
		"messages":    []any{map[string]any{"role": "user", "content": "__mock_tool_call__"}},
		"tools":       []any{tool},
		"tool_choice": map[string]any{"type": "tool", "name": "lookup"},
	}
	firstBody["stream"] = true
	streamedFirst, err := sendProtocolMatrixRequest(ctx, session, "/v1/messages", firstBody, true)
	if err != nil {
		return fmt.Errorf("streaming tool-call turn: %w", err)
	}
	streamCall, err := decodeAnthropicToolUseStream(streamedFirst)
	if err != nil {
		return fmt.Errorf("streaming tool-call turn: %w", err)
	}
	delete(firstBody, "stream")
	first, err := sendProtocolMatrixRequest(ctx, session, "/v1/messages", firstBody, false)
	if err != nil {
		return fmt.Errorf("tool-call turn: %w", err)
	}
	call, err := decodeAnthropicToolUse(first)
	if err != nil {
		return fmt.Errorf("tool-call turn: %w", err)
	}
	if call != streamCall {
		return fmt.Errorf("buffered and streamed calls differ: buffered=%+v streamed=%+v", call, streamCall)
	}
	var input map[string]any
	if err := json.Unmarshal([]byte(call.Arguments), &input); err != nil {
		return fmt.Errorf("decode tool input: %w", err)
	}

	secondBody := map[string]any{
		"model": model, "max_tokens": 64,
		"messages": []any{
			map[string]any{"role": "user", "content": "__mock_tool_call__"},
			map[string]any{
				"role": "assistant",
				"content": []any{map[string]any{
					"type": "tool_use", "id": call.CallID, "name": call.Name, "input": input,
				}},
			},
			map[string]any{
				"role": "user",
				"content": []any{map[string]any{
					"type": "tool_result", "tool_use_id": call.CallID, "content": "sunny",
				}},
			},
		},
		"tools": []any{tool},
	}
	secondBody["stream"] = true
	streamedSecond, err := sendProtocolMatrixRequest(ctx, session, "/v1/messages", secondBody, true)
	if err != nil {
		return fmt.Errorf("streaming tool-result turn: %w", err)
	}
	if err := assertAnthropicTextStream(streamedSecond, "tool result accepted"); err != nil {
		return fmt.Errorf("streaming tool-result turn: %w", err)
	}
	delete(secondBody, "stream")
	second, err := sendProtocolMatrixRequest(ctx, session, "/v1/messages", secondBody, false)
	if err != nil {
		return fmt.Errorf("tool-result turn: %w", err)
	}
	if err := assertAnthropicText(second, "tool result accepted"); err != nil {
		return fmt.Errorf("tool-result turn: %w", err)
	}
	return nil
}

func decodeAnthropicToolUse(body []byte) (responsesFunctionCall, error) {
	var response struct {
		StopReason string `json:"stop_reason"`
		Content    []struct {
			Type  string          `json:"type"`
			ID    string          `json:"id"`
			Name  string          `json:"name"`
			Input json.RawMessage `json:"input"`
		} `json:"content"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		return responsesFunctionCall{}, err
	}
	if response.StopReason != "tool_use" || len(response.Content) != 1 || response.Content[0].Type != "tool_use" {
		return responsesFunctionCall{}, fmt.Errorf("Anthropic tool use is missing or malformed: %s", truncateString(string(body), 1000))
	}
	content := response.Content[0]
	call := responsesFunctionCall{CallID: content.ID, Name: content.Name, Arguments: string(content.Input)}
	if call.CallID != "call_mock_lookup" || call.Name != "lookup" || call.Arguments != `{"query":"weather"}` || !json.Valid(content.Input) {
		return responsesFunctionCall{}, fmt.Errorf("Anthropic tool use changed identity or input: %s", truncateString(string(body), 1000))
	}
	return call, nil
}

func decodeAnthropicToolUseStream(body []byte) (responsesFunctionCall, error) {
	stream := string(body)
	for _, leaked := range []string{"chat.completion.chunk", "data: [DONE]", "event: response."} {
		if strings.Contains(stream, leaked) {
			return responsesFunctionCall{}, fmt.Errorf("backend stream format leaked %q: %s", leaked, truncateString(stream, 1200))
		}
	}
	var call responsesFunctionCall
	stopReason := ""
	blockStopped := false
	messageStopped := false
	for _, frame := range strings.Split(stream, "\n\n") {
		var data string
		for _, line := range strings.Split(frame, "\n") {
			if strings.HasPrefix(line, "data: ") {
				data = strings.TrimPrefix(line, "data: ")
				break
			}
		}
		if data == "" {
			continue
		}
		var event struct {
			Type         string `json:"type"`
			Index        int    `json:"index"`
			ContentBlock struct {
				Type string `json:"type"`
				ID   string `json:"id"`
				Name string `json:"name"`
			} `json:"content_block"`
			Delta struct {
				Type        string  `json:"type"`
				PartialJSON string  `json:"partial_json"`
				StopReason  *string `json:"stop_reason"`
			} `json:"delta"`
		}
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			return responsesFunctionCall{}, err
		}
		switch event.Type {
		case "content_block_start":
			if event.Index != 0 || event.ContentBlock.Type != "tool_use" {
				return responsesFunctionCall{}, fmt.Errorf("invalid Anthropic tool block start: %s", data)
			}
			call.CallID, call.Name = event.ContentBlock.ID, event.ContentBlock.Name
		case "content_block_delta":
			if event.Index != 0 || event.Delta.Type != "input_json_delta" {
				return responsesFunctionCall{}, fmt.Errorf("invalid Anthropic tool block delta: %s", data)
			}
			call.Arguments += event.Delta.PartialJSON
		case "content_block_stop":
			blockStopped = event.Index == 0
		case "message_delta":
			if event.Delta.StopReason != nil {
				stopReason = *event.Delta.StopReason
			}
		case "message_stop":
			messageStopped = true
		}
	}
	if !blockStopped || !messageStopped || stopReason != "tool_use" || call.CallID != "call_mock_lookup" ||
		call.Name != "lookup" || call.Arguments != `{"query":"weather"}` || !json.Valid([]byte(call.Arguments)) {
		return responsesFunctionCall{}, fmt.Errorf("Anthropic tool stream lifecycle is incomplete: %s", truncateString(stream, 1200))
	}
	return call, nil
}

func assertAnthropicText(body []byte, contains string) error {
	var response struct {
		StopReason string `json:"stop_reason"`
		Content    []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		return err
	}
	if response.StopReason != "end_turn" || len(response.Content) != 1 ||
		response.Content[0].Type != "text" || !strings.Contains(response.Content[0].Text, contains) {
		return fmt.Errorf("Anthropic output text does not contain %q: %s", contains, truncateString(string(body), 800))
	}
	return nil
}

func assertAnthropicTextStream(body []byte, contains string) error {
	stream := string(body)
	var text strings.Builder
	stopReason := ""
	messageStopped := false
	for _, frame := range strings.Split(stream, "\n\n") {
		var data string
		for _, line := range strings.Split(frame, "\n") {
			if strings.HasPrefix(line, "data: ") {
				data = strings.TrimPrefix(line, "data: ")
				break
			}
		}
		if data == "" {
			continue
		}
		var event struct {
			Type  string `json:"type"`
			Delta struct {
				Type       string  `json:"type"`
				Text       string  `json:"text"`
				StopReason *string `json:"stop_reason"`
			} `json:"delta"`
		}
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			return err
		}
		if event.Type == "content_block_delta" && event.Delta.Type == "text_delta" {
			text.WriteString(event.Delta.Text)
		}
		if event.Type == "message_delta" && event.Delta.StopReason != nil {
			stopReason = *event.Delta.StopReason
		}
		if event.Type == "message_stop" {
			messageStopped = true
		}
	}
	if !messageStopped || stopReason != "end_turn" || !strings.Contains(text.String(), contains) {
		return fmt.Errorf("Anthropic text stream is incomplete: %s", truncateString(stream, 1200))
	}
	return nil
}
