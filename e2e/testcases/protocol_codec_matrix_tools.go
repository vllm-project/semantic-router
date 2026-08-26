package testcases

import (
	"bytes"
	"encoding/json"
	"fmt"
)

type protocolMatrixToolCall struct {
	id        string
	name      string
	arguments string
}

func decodeProtocolMatrixToolBuffered(format string, body []byte) (protocolMatrixToolCall, protocolMatrixUsage, error) {
	switch format {
	case matrixOpenAIChat:
		return decodeProtocolMatrixChatToolBuffered(body)
	case matrixOpenAIResponses:
		return decodeProtocolMatrixResponsesToolBuffered(body)
	case matrixAnthropic:
		return decodeProtocolMatrixAnthropicToolBuffered(body)
	default:
		return protocolMatrixToolCall{}, protocolMatrixUsage{}, fmt.Errorf("unsupported response format %q", format)
	}
}

func decodeProtocolMatrixChatToolBuffered(body []byte) (protocolMatrixToolCall, protocolMatrixUsage, error) {
	var response struct {
		Choices []struct {
			Message struct {
				ToolCalls []struct {
					ID       string `json:"id"`
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			Input  int64 `json:"prompt_tokens"`
			Output int64 `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &response); err != nil ||
		len(response.Choices) != 1 || len(response.Choices[0].Message.ToolCalls) != 1 {
		return protocolMatrixToolCall{}, protocolMatrixUsage{}, fmt.Errorf("invalid Chat Completions tool response")
	}
	call := response.Choices[0].Message.ToolCalls[0]
	return protocolMatrixToolCall{
		id: call.ID, name: call.Function.Name, arguments: call.Function.Arguments,
	}, protocolMatrixUsage{input: response.Usage.Input, output: response.Usage.Output}, nil
}

func decodeProtocolMatrixResponsesToolBuffered(body []byte) (protocolMatrixToolCall, protocolMatrixUsage, error) {
	var response struct {
		Output []struct {
			Type      string `json:"type"`
			CallID    string `json:"call_id"`
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"output"`
		Usage struct {
			Input  int64 `json:"input_tokens"`
			Output int64 `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		return protocolMatrixToolCall{}, protocolMatrixUsage{}, fmt.Errorf("invalid Responses tool response")
	}
	for _, item := range response.Output {
		if item.Type == "function_call" {
			return protocolMatrixToolCall{
				id: item.CallID, name: item.Name, arguments: item.Arguments,
			}, protocolMatrixUsage{input: response.Usage.Input, output: response.Usage.Output}, nil
		}
	}
	return protocolMatrixToolCall{}, protocolMatrixUsage{}, fmt.Errorf("Responses tool call is missing")
}

func decodeProtocolMatrixAnthropicToolBuffered(body []byte) (protocolMatrixToolCall, protocolMatrixUsage, error) {
	var response struct {
		Content []struct {
			Type  string          `json:"type"`
			ID    string          `json:"id"`
			Name  string          `json:"name"`
			Input json.RawMessage `json:"input"`
		} `json:"content"`
		Usage struct {
			Input  int64 `json:"input_tokens"`
			Output int64 `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		return protocolMatrixToolCall{}, protocolMatrixUsage{}, fmt.Errorf("invalid Messages tool response")
	}
	for _, block := range response.Content {
		if block.Type == "tool_use" {
			return protocolMatrixToolCall{
				id: block.ID, name: block.Name, arguments: string(block.Input),
			}, protocolMatrixUsage{input: response.Usage.Input, output: response.Usage.Output}, nil
		}
	}
	return protocolMatrixToolCall{}, protocolMatrixUsage{}, fmt.Errorf("Messages tool call is missing")
}

func decodeProtocolMatrixToolStream(format string, body []byte) (protocolMatrixToolCall, protocolMatrixUsage, error) {
	frames, err := parseProtocolMatrixSSE(body)
	if err != nil {
		return protocolMatrixToolCall{}, protocolMatrixUsage{}, err
	}
	switch format {
	case matrixOpenAIChat:
		return decodeProtocolMatrixChatToolStream(frames)
	case matrixOpenAIResponses:
		return decodeProtocolMatrixResponsesToolStream(frames)
	case matrixAnthropic:
		return decodeProtocolMatrixAnthropicToolStream(frames)
	default:
		return protocolMatrixToolCall{}, protocolMatrixUsage{}, fmt.Errorf("unsupported stream format %q", format)
	}
}

type protocolMatrixChatToolDelta struct {
	ID       string `json:"id"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type protocolMatrixChatToolChunk struct {
	Choices []struct {
		Delta struct {
			ToolCalls []protocolMatrixChatToolDelta `json:"tool_calls"`
		} `json:"delta"`
	} `json:"choices"`
	Usage *struct {
		Input  int64 `json:"prompt_tokens"`
		Output int64 `json:"completion_tokens"`
	} `json:"usage"`
}

func decodeProtocolMatrixChatToolStream(frames []protocolMatrixSSEFrame) (protocolMatrixToolCall, protocolMatrixUsage, error) {
	call := protocolMatrixToolCall{}
	usage := protocolMatrixUsage{}
	done := 0
	for _, frame := range frames {
		chunk, terminal, err := decodeProtocolMatrixChatToolFrame(frame)
		if err != nil {
			return protocolMatrixToolCall{}, protocolMatrixUsage{}, err
		}
		if terminal {
			done++
			continue
		}
		mergeProtocolMatrixChatToolChunk(&call, chunk)
		if chunk.Usage != nil {
			usage = protocolMatrixUsage{input: chunk.Usage.Input, output: chunk.Usage.Output}
		}
	}
	if done != 1 {
		return protocolMatrixToolCall{}, protocolMatrixUsage{}, fmt.Errorf("expected one Chat Completions terminal, got %d", done)
	}
	return call, usage, nil
}

func decodeProtocolMatrixChatToolFrame(
	frame protocolMatrixSSEFrame,
) (protocolMatrixChatToolChunk, bool, error) {
	if string(bytes.TrimSpace(frame.data)) == "[DONE]" {
		return protocolMatrixChatToolChunk{}, true, nil
	}
	var chunk protocolMatrixChatToolChunk
	if err := json.Unmarshal(frame.data, &chunk); err != nil {
		return protocolMatrixChatToolChunk{}, false, err
	}
	return chunk, false, nil
}

func mergeProtocolMatrixChatToolChunk(call *protocolMatrixToolCall, chunk protocolMatrixChatToolChunk) {
	for _, choice := range chunk.Choices {
		for _, delta := range choice.Delta.ToolCalls {
			mergeProtocolMatrixChatToolDelta(call, delta)
		}
	}
}

func mergeProtocolMatrixChatToolDelta(call *protocolMatrixToolCall, delta protocolMatrixChatToolDelta) {
	if delta.ID != "" {
		call.id = delta.ID
	}
	if delta.Function.Name != "" {
		call.name = delta.Function.Name
	}
	call.arguments += delta.Function.Arguments
}

type protocolMatrixResponsesUsage struct {
	Input  int64 `json:"input_tokens"`
	Output int64 `json:"output_tokens"`
}

type protocolMatrixResponsesToolItem struct {
	Type      string `json:"type"`
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type protocolMatrixResponsesToolEvent struct {
	Type     string                           `json:"type"`
	Delta    string                           `json:"delta"`
	Item     *protocolMatrixResponsesToolItem `json:"item"`
	Response *struct {
		Usage *protocolMatrixResponsesUsage `json:"usage"`
	} `json:"response"`
}

func decodeProtocolMatrixResponsesToolStream(frames []protocolMatrixSSEFrame) (protocolMatrixToolCall, protocolMatrixUsage, error) {
	call := protocolMatrixToolCall{}
	usage := protocolMatrixUsage{}
	terminal := 0
	for _, frame := range frames {
		event, err := decodeProtocolMatrixResponsesToolEvent(frame)
		if err != nil {
			return protocolMatrixToolCall{}, protocolMatrixUsage{}, err
		}
		eventTerminal, err := applyProtocolMatrixResponsesToolEvent(event, &call, &usage)
		if err != nil {
			return protocolMatrixToolCall{}, protocolMatrixUsage{}, err
		}
		if eventTerminal {
			terminal++
		}
	}
	if terminal != 1 {
		return protocolMatrixToolCall{}, protocolMatrixUsage{}, fmt.Errorf("expected one Responses terminal, got %d", terminal)
	}
	return call, usage, nil
}

func decodeProtocolMatrixResponsesToolEvent(
	frame protocolMatrixSSEFrame,
) (protocolMatrixResponsesToolEvent, error) {
	var event protocolMatrixResponsesToolEvent
	if err := json.Unmarshal(frame.data, &event); err != nil {
		return protocolMatrixResponsesToolEvent{}, err
	}
	event.Type = protocolMatrixEventType(event.Type, frame.event)
	return event, nil
}

func applyProtocolMatrixResponsesToolEvent(
	event protocolMatrixResponsesToolEvent,
	call *protocolMatrixToolCall,
	usage *protocolMatrixUsage,
) (bool, error) {
	switch event.Type {
	case "response.output_item.added", "response.output_item.done":
		applyProtocolMatrixResponsesToolItem(call, event.Item)
	case "response.function_call_arguments.delta":
		call.arguments += event.Delta
	case "response.completed":
		applyProtocolMatrixResponsesUsage(usage, event)
		return true, nil
	case "response.failed":
		return false, fmt.Errorf("Responses tool stream failed")
	}
	return false, nil
}

func applyProtocolMatrixResponsesToolItem(
	call *protocolMatrixToolCall,
	item *protocolMatrixResponsesToolItem,
) {
	if item == nil || item.Type != "function_call" {
		return
	}
	call.id, call.name, call.arguments = item.CallID, item.Name, item.Arguments
}

func applyProtocolMatrixResponsesUsage(
	usage *protocolMatrixUsage,
	event protocolMatrixResponsesToolEvent,
) {
	if event.Response == nil || event.Response.Usage == nil {
		return
	}
	usage.input = event.Response.Usage.Input
	usage.output = event.Response.Usage.Output
}

type protocolMatrixAnthropicToolEvent struct {
	Type    string `json:"type"`
	Message *struct {
		Usage *struct {
			Input int64 `json:"input_tokens"`
		} `json:"usage"`
	} `json:"message"`
	ContentBlock *struct {
		Type  string          `json:"type"`
		ID    string          `json:"id"`
		Name  string          `json:"name"`
		Input json.RawMessage `json:"input"`
	} `json:"content_block"`
	Delta *struct {
		Type        string `json:"type"`
		PartialJSON string `json:"partial_json"`
	} `json:"delta"`
	Usage *struct {
		Output int64 `json:"output_tokens"`
	} `json:"usage"`
}

func decodeProtocolMatrixAnthropicToolStream(frames []protocolMatrixSSEFrame) (protocolMatrixToolCall, protocolMatrixUsage, error) {
	call := protocolMatrixToolCall{}
	usage := protocolMatrixUsage{}
	terminal := 0
	for _, frame := range frames {
		event, err := decodeProtocolMatrixAnthropicToolEvent(frame)
		if err != nil {
			return protocolMatrixToolCall{}, protocolMatrixUsage{}, err
		}
		eventTerminal, err := applyProtocolMatrixAnthropicToolEvent(event, &call, &usage)
		if err != nil {
			return protocolMatrixToolCall{}, protocolMatrixUsage{}, err
		}
		if eventTerminal {
			terminal++
		}
	}
	if terminal != 1 {
		return protocolMatrixToolCall{}, protocolMatrixUsage{}, fmt.Errorf("expected one Messages terminal, got %d", terminal)
	}
	return call, usage, nil
}

func decodeProtocolMatrixAnthropicToolEvent(
	frame protocolMatrixSSEFrame,
) (protocolMatrixAnthropicToolEvent, error) {
	var event protocolMatrixAnthropicToolEvent
	if err := json.Unmarshal(frame.data, &event); err != nil {
		return protocolMatrixAnthropicToolEvent{}, err
	}
	event.Type = protocolMatrixEventType(event.Type, frame.event)
	return event, nil
}

func applyProtocolMatrixAnthropicToolEvent(
	event protocolMatrixAnthropicToolEvent,
	call *protocolMatrixToolCall,
	usage *protocolMatrixUsage,
) (bool, error) {
	switch event.Type {
	case "message_start":
		applyProtocolMatrixAnthropicToolInputUsage(usage, event)
	case "content_block_start":
		applyProtocolMatrixAnthropicToolStart(call, event)
	case "content_block_delta":
		applyProtocolMatrixAnthropicToolDelta(call, event)
	case "message_delta":
		applyProtocolMatrixAnthropicToolOutputUsage(usage, event)
	case "message_stop":
		return true, nil
	case "error":
		return false, fmt.Errorf("Messages tool stream failed")
	}
	return false, nil
}

func applyProtocolMatrixAnthropicToolInputUsage(
	usage *protocolMatrixUsage,
	event protocolMatrixAnthropicToolEvent,
) {
	if event.Message != nil && event.Message.Usage != nil {
		usage.input = event.Message.Usage.Input
	}
}

func applyProtocolMatrixAnthropicToolStart(
	call *protocolMatrixToolCall,
	event protocolMatrixAnthropicToolEvent,
) {
	if event.ContentBlock == nil || event.ContentBlock.Type != "tool_use" {
		return
	}
	call.id, call.name = event.ContentBlock.ID, event.ContentBlock.Name
	if value := string(event.ContentBlock.Input); value != "" && value != "{}" {
		call.arguments = value
	}
}

func applyProtocolMatrixAnthropicToolDelta(
	call *protocolMatrixToolCall,
	event protocolMatrixAnthropicToolEvent,
) {
	if event.Delta != nil && event.Delta.Type == "input_json_delta" {
		call.arguments += event.Delta.PartialJSON
	}
}

func applyProtocolMatrixAnthropicToolOutputUsage(
	usage *protocolMatrixUsage,
	event protocolMatrixAnthropicToolEvent,
) {
	if event.Usage != nil {
		usage.output = event.Usage.Output
	}
}

func protocolMatrixEventType(payloadType string, frameType string) string {
	if payloadType != "" {
		return payloadType
	}
	return frameType
}

func assertProtocolMatrixToolCall(call protocolMatrixToolCall, expectedBackendFormat string) error {
	if call.id != "call_protocol_123" || call.name != "protocol_marker" {
		return fmt.Errorf("tool identity = {id:%q name:%q}", call.id, call.name)
	}
	var arguments struct {
		Protocol string `json:"protocol"`
	}
	if err := json.Unmarshal([]byte(call.arguments), &arguments); err != nil {
		return fmt.Errorf("tool arguments are invalid JSON %q: %w", call.arguments, err)
	}
	if arguments.Protocol != expectedBackendFormat {
		return fmt.Errorf("tool backend protocol = %q, want %q", arguments.Protocol, expectedBackendFormat)
	}
	return nil
}
