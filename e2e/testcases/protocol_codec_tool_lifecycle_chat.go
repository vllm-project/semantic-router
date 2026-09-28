package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
)

func runChatClientToolLifecycle(
	ctx context.Context,
	session *fixtures.ServiceSession,
	model string,
) error {
	tool := map[string]any{
		"type": "function",
		"function": map[string]any{
			"name": "lookup", "description": "Look up a value",
			"parameters": map[string]any{
				"type":       "object",
				"properties": map[string]any{"query": map[string]any{"type": "string"}},
				"required":   []string{"query"},
			},
		},
	}
	firstBody := map[string]any{
		"model":    model,
		"messages": []any{map[string]any{"role": "user", "content": "__mock_tool_call__"}},
		"tools":    []any{tool},
		"tool_choice": map[string]any{
			"type": "function", "function": map[string]any{"name": "lookup"},
		},
	}
	firstBody["stream"] = true
	streamedFirst, err := sendProtocolMatrixRequest(ctx, session, "/v1/chat/completions", firstBody, true)
	if err != nil {
		return fmt.Errorf("streaming tool-call turn: %w", err)
	}
	streamCall, err := decodeChatFunctionCallStream(streamedFirst)
	if err != nil {
		return fmt.Errorf("streaming tool-call turn: %w", err)
	}
	delete(firstBody, "stream")
	first, err := sendProtocolMatrixRequest(ctx, session, "/v1/chat/completions", firstBody, false)
	if err != nil {
		return fmt.Errorf("tool-call turn: %w", err)
	}
	call, err := decodeChatFunctionCall(first)
	if err != nil {
		return fmt.Errorf("tool-call turn: %w", err)
	}
	if call != streamCall {
		return fmt.Errorf("buffered and streamed calls differ: buffered=%+v streamed=%+v", call, streamCall)
	}

	secondBody := map[string]any{
		"model": model,
		"messages": []any{
			map[string]any{"role": "user", "content": "__mock_tool_call__"},
			map[string]any{
				"role": "assistant", "content": nil,
				"tool_calls": []any{map[string]any{
					"id": call.CallID, "type": "function",
					"function": map[string]any{"name": call.Name, "arguments": call.Arguments},
				}},
			},
			map[string]any{"role": "tool", "tool_call_id": call.CallID, "content": "sunny"},
		},
		"tools": []any{tool},
	}
	secondBody["stream"] = true
	streamedSecond, err := sendProtocolMatrixRequest(ctx, session, "/v1/chat/completions", secondBody, true)
	if err != nil {
		return fmt.Errorf("streaming tool-result turn: %w", err)
	}
	if err := assertChatTextStream(streamedSecond, "tool result accepted"); err != nil {
		return fmt.Errorf("streaming tool-result turn: %w", err)
	}
	delete(secondBody, "stream")
	second, err := sendProtocolMatrixRequest(ctx, session, "/v1/chat/completions", secondBody, false)
	if err != nil {
		return fmt.Errorf("tool-result turn: %w", err)
	}
	if err := assertChatText(second, "tool result accepted"); err != nil {
		return fmt.Errorf("tool-result turn: %w", err)
	}
	return nil
}

func decodeChatFunctionCall(body []byte) (responsesFunctionCall, error) {
	var response struct {
		Choices []struct {
			Message struct {
				ToolCalls []struct {
					ID       string `json:"id"`
					Type     string `json:"type"`
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		return responsesFunctionCall{}, err
	}
	if len(response.Choices) != 1 || response.Choices[0].FinishReason != "tool_calls" || len(response.Choices[0].Message.ToolCalls) != 1 {
		return responsesFunctionCall{}, fmt.Errorf("Chat tool call is missing or malformed: %s", truncateString(string(body), 1000))
	}
	tool := response.Choices[0].Message.ToolCalls[0]
	call := responsesFunctionCall{CallID: tool.ID, Name: tool.Function.Name, Arguments: tool.Function.Arguments}
	if tool.Type != "function" || call.CallID != "call_mock_lookup" || call.Name != "lookup" ||
		call.Arguments != `{"query":"weather"}` || !json.Valid([]byte(call.Arguments)) {
		return responsesFunctionCall{}, fmt.Errorf("Chat tool call changed identity or arguments: %s", truncateString(string(body), 1000))
	}
	return call, nil
}

type chatToolStreamChunk struct {
	Object  string `json:"object"`
	Choices []struct {
		FinishReason *string `json:"finish_reason"`
		Delta        struct {
			ToolCalls []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
	} `json:"choices"`
}

type chatToolStreamState struct {
	call         responsesFunctionCall
	finishReason string
	doneCount    int
}

func decodeChatFunctionCallStream(body []byte) (responsesFunctionCall, error) {
	stream := string(body)
	if err := rejectStreamFragments(stream, "Chat", []string{"event: response.", "event: message_start", `"type":"message_start"`}); err != nil {
		return responsesFunctionCall{}, err
	}
	state := chatToolStreamState{}
	for _, data := range protocolSSEDataFrames(body) {
		if data == "[DONE]" {
			state.doneCount++
			continue
		}
		var chunk chatToolStreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return responsesFunctionCall{}, fmt.Errorf("decode Chat tool stream chunk: %w", err)
		}
		if err := state.consume(chunk, data); err != nil {
			return responsesFunctionCall{}, err
		}
	}
	return state.result(stream)
}

func (state *chatToolStreamState) consume(chunk chatToolStreamChunk, data string) error {
	if chunk.Object != "chat.completion.chunk" {
		return fmt.Errorf("invalid Chat stream object %q: %s", chunk.Object, data)
	}
	for _, choice := range chunk.Choices {
		if choice.FinishReason != nil {
			state.finishReason = *choice.FinishReason
		}
		for _, tool := range choice.Delta.ToolCalls {
			if err := state.consumeTool(tool.Index, tool.ID, tool.Function.Name, tool.Function.Arguments, data); err != nil {
				return err
			}
		}
	}
	return nil
}

func (state *chatToolStreamState) consumeTool(index int, id, name, arguments, data string) error {
	if index != 0 {
		return fmt.Errorf("unexpected Chat tool index %d: %s", index, data)
	}
	if id != "" && state.call.CallID != "" && state.call.CallID != id {
		return fmt.Errorf("Chat stream changed tool ID: %s", data)
	}
	if name != "" && state.call.Name != "" && state.call.Name != name {
		return fmt.Errorf("Chat stream changed tool name: %s", data)
	}
	if id != "" {
		state.call.CallID = id
	}
	if name != "" {
		state.call.Name = name
	}
	state.call.Arguments += arguments
	return nil
}

func (state *chatToolStreamState) result(stream string) (responsesFunctionCall, error) {
	call := state.call
	if state.doneCount != 1 || state.finishReason != "tool_calls" || call.CallID != "call_mock_lookup" ||
		call.Name != "lookup" || call.Arguments != `{"query":"weather"}` || !json.Valid([]byte(call.Arguments)) {
		return responsesFunctionCall{}, fmt.Errorf("Chat tool stream lifecycle is incomplete: %s", truncateString(stream, 1200))
	}
	return call, nil
}

func assertChatText(body []byte, contains string) error {
	var response struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		return err
	}
	if len(response.Choices) != 1 || response.Choices[0].FinishReason != "stop" ||
		!strings.Contains(response.Choices[0].Message.Content, contains) {
		return fmt.Errorf("Chat output text does not contain %q: %s", contains, truncateString(string(body), 800))
	}
	return nil
}

func assertChatTextStream(body []byte, contains string) error {
	stream := string(body)
	state := chatTextStreamState{}
	for _, data := range protocolSSEDataFrames(body) {
		if data == "[DONE]" {
			state.doneCount++
			continue
		}
		var chunk chatTextStreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return err
		}
		if err := state.consume(chunk, data); err != nil {
			return err
		}
	}
	if state.doneCount != 1 || state.finishReason != "stop" || !strings.Contains(state.text.String(), contains) {
		return fmt.Errorf("Chat text stream is incomplete: %s", truncateString(stream, 1200))
	}
	return nil
}

type chatTextStreamChunk struct {
	Object  string `json:"object"`
	Choices []struct {
		FinishReason *string `json:"finish_reason"`
		Delta        struct {
			Content string `json:"content"`
		} `json:"delta"`
	} `json:"choices"`
}

type chatTextStreamState struct {
	text         strings.Builder
	finishReason string
	doneCount    int
}

func (state *chatTextStreamState) consume(chunk chatTextStreamChunk, data string) error {
	if chunk.Object != "chat.completion.chunk" {
		return fmt.Errorf("invalid Chat stream object %q: %s", chunk.Object, data)
	}
	for _, choice := range chunk.Choices {
		state.text.WriteString(choice.Delta.Content)
		if choice.FinishReason != nil {
			state.finishReason = *choice.FinishReason
		}
	}
	return nil
}
