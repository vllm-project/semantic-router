package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

const (
	matrixOpenAIChat      = "openai.chat.v1"
	matrixOpenAIResponses = "openai.responses.v1"
	matrixAnthropic       = "anthropic.messages.v1"
)

var protocolMatrixModels = []struct {
	model         string
	backendFormat string
}{
	{model: "protocol-matrix/chat", backendFormat: matrixOpenAIChat},
	{model: "protocol-matrix/responses", backendFormat: matrixOpenAIResponses},
	{model: "protocol-matrix/messages", backendFormat: matrixAnthropic},
}

var protocolMatrixClientFormats = []string{
	matrixOpenAIChat,
	matrixOpenAIResponses,
	matrixAnthropic,
}

func init() {
	pkgtestcases.Register("protocol-codec-buffered-matrix", pkgtestcases.TestCase{
		Description: "Every public protocol translates to every backend protocol with authoritative usage",
		Tags:        []string{"protocol", "codec", "matrix", "functional"},
		Fn:          testProtocolCodecBufferedMatrix,
	})
	pkgtestcases.Register("protocol-codec-streaming-matrix", pkgtestcases.TestCase{
		Description: "Every public streaming protocol translates to every backend streaming protocol and terminates once",
		Tags:        []string{"protocol", "codec", "matrix", "streaming", "sse"},
		Fn:          testProtocolCodecStreamingMatrix,
	})
	pkgtestcases.Register("protocol-codec-tool-buffered-matrix", pkgtestcases.TestCase{
		Description: "Every public protocol preserves tool-call identity across every backend protocol",
		Tags:        []string{"protocol", "codec", "matrix", "tools", "functional"},
		Fn: func(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
			return runProtocolCodecToolMatrix(ctx, client, opts, false)
		},
	})
	pkgtestcases.Register("protocol-codec-tool-streaming-matrix", pkgtestcases.TestCase{
		Description: "Every public streaming protocol preserves tool-call lifecycle and authoritative usage",
		Tags:        []string{"protocol", "codec", "matrix", "tools", "streaming", "sse"},
		Fn: func(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
			return runProtocolCodecToolMatrix(ctx, client, opts, true)
		},
	})
}

func testProtocolCodecBufferedMatrix(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	return runProtocolCodecMatrix(ctx, client, opts, false)
}

func testProtocolCodecStreamingMatrix(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	return runProtocolCodecMatrix(ctx, client, opts, true)
}

func runProtocolCodecMatrix(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
	stream bool,
) error {
	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	httpClient := &http.Client{Timeout: 90 * time.Second}
	baseURL := fmt.Sprintf("http://localhost:%s", localPort)
	completed := 0
	for _, sourceFormat := range protocolMatrixClientFormats {
		for _, backend := range protocolMatrixModels {
			result, requestErr := requestProtocolMatrix(
				ctx, httpClient, baseURL, sourceFormat, backend.model, stream,
			)
			if requestErr != nil {
				return fmt.Errorf(
					"%s to %s (stream=%t): %w",
					sourceFormat, backend.backendFormat, stream, requestErr,
				)
			}
			if result.statusCode != http.StatusOK {
				return fmt.Errorf(
					"%s to %s (stream=%t) returned %d: %s",
					sourceFormat, backend.backendFormat, stream, result.statusCode,
					truncateString(string(result.body), 500),
				)
			}
			if stream && !strings.Contains(result.contentType, "text/event-stream") {
				return fmt.Errorf(
					"%s to %s returned streaming content type %q",
					sourceFormat, backend.backendFormat, result.contentType,
				)
			}

			var content string
			var usage protocolMatrixUsage
			if stream {
				content, usage, err = decodeProtocolMatrixStream(sourceFormat, result.body)
			} else {
				content, usage, err = decodeProtocolMatrixBuffered(sourceFormat, result.body)
			}
			if err != nil {
				return fmt.Errorf(
					"decode %s response from %s backend (stream=%t): %w",
					sourceFormat, backend.backendFormat, stream, err,
				)
			}
			if err := assertProtocolMatrixMarker(content, backend.backendFormat); err != nil {
				return fmt.Errorf(
					"%s to %s (stream=%t): %w",
					sourceFormat, backend.backendFormat, stream, err,
				)
			}
			if usage.input <= 0 || usage.output <= 0 || usage.total() <= 0 {
				return fmt.Errorf(
					"%s to %s (stream=%t) omitted authoritative usage: %+v",
					sourceFormat, backend.backendFormat, stream, usage,
				)
			}
			completed++
		}
	}
	if completed != len(protocolMatrixClientFormats)*len(protocolMatrixModels) {
		return fmt.Errorf("protocol matrix completed %d cells", completed)
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"client_formats":  len(protocolMatrixClientFormats),
			"backend_formats": len(protocolMatrixModels),
			"matrix_cells":    completed,
			"streaming":       stream,
		})
	}
	return nil
}

func runProtocolCodecToolMatrix(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
	stream bool,
) error {
	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	httpClient := &http.Client{Timeout: 90 * time.Second}
	baseURL := fmt.Sprintf("http://localhost:%s", localPort)
	completed := 0
	for _, sourceFormat := range protocolMatrixClientFormats {
		for _, backend := range protocolMatrixModels {
			result, requestErr := requestProtocolToolMatrix(
				ctx, httpClient, baseURL, sourceFormat, backend.model, stream,
			)
			if requestErr != nil {
				return fmt.Errorf("tool %s to %s (stream=%t): %w", sourceFormat, backend.backendFormat, stream, requestErr)
			}
			if result.statusCode != http.StatusOK {
				return fmt.Errorf("tool %s to %s (stream=%t) returned %d: %s", sourceFormat, backend.backendFormat, stream, result.statusCode, truncateString(string(result.body), 500))
			}
			var call protocolMatrixToolCall
			var usage protocolMatrixUsage
			if stream {
				call, usage, err = decodeProtocolMatrixToolStream(sourceFormat, result.body)
			} else {
				call, usage, err = decodeProtocolMatrixToolBuffered(sourceFormat, result.body)
			}
			if err != nil {
				return fmt.Errorf("decode tool %s response from %s backend (stream=%t): %w", sourceFormat, backend.backendFormat, stream, err)
			}
			if err := assertProtocolMatrixToolCall(call, backend.backendFormat); err != nil {
				return fmt.Errorf("tool %s to %s (stream=%t): %w", sourceFormat, backend.backendFormat, stream, err)
			}
			if usage.input <= 0 || usage.output <= 0 || usage.total() <= 0 {
				return fmt.Errorf("tool %s to %s (stream=%t) omitted authoritative usage: %+v", sourceFormat, backend.backendFormat, stream, usage)
			}
			completed++
		}
	}
	if completed != len(protocolMatrixClientFormats)*len(protocolMatrixModels) {
		return fmt.Errorf("protocol tool matrix completed %d cells", completed)
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"client_formats": len(protocolMatrixClientFormats), "backend_formats": len(protocolMatrixModels),
			"matrix_cells": completed, "streaming": stream, "scenario": "tool_call",
		})
	}
	return nil
}

type protocolMatrixUsage struct {
	input  int64
	output int64
}

func (usage protocolMatrixUsage) total() int64 {
	return usage.input + usage.output
}

func decodeProtocolMatrixBuffered(format string, body []byte) (string, protocolMatrixUsage, error) {
	switch format {
	case matrixOpenAIChat:
		var response struct {
			Choices []struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
			} `json:"choices"`
			Usage struct {
				Input  int64 `json:"prompt_tokens"`
				Output int64 `json:"completion_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal(body, &response); err != nil || len(response.Choices) != 1 {
			return "", protocolMatrixUsage{}, fmt.Errorf("invalid Chat Completions response")
		}
		return response.Choices[0].Message.Content, protocolMatrixUsage{
			input: response.Usage.Input, output: response.Usage.Output,
		}, nil
	case matrixOpenAIResponses:
		var response struct {
			Output []struct {
				Content []struct {
					Type string `json:"type"`
					Text string `json:"text"`
				} `json:"content"`
			} `json:"output"`
			Usage struct {
				Input  int64 `json:"input_tokens"`
				Output int64 `json:"output_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal(body, &response); err != nil || len(response.Output) != 1 {
			return "", protocolMatrixUsage{}, fmt.Errorf("invalid Responses response")
		}
		content := ""
		for _, part := range response.Output[0].Content {
			if part.Type == "output_text" {
				content += part.Text
			}
		}
		return content, protocolMatrixUsage{input: response.Usage.Input, output: response.Usage.Output}, nil
	case matrixAnthropic:
		var response struct {
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
			Usage struct {
				Input  int64 `json:"input_tokens"`
				Output int64 `json:"output_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal(body, &response); err != nil {
			return "", protocolMatrixUsage{}, fmt.Errorf("invalid Messages response")
		}
		content := ""
		for _, part := range response.Content {
			if part.Type == "text" {
				content += part.Text
			}
		}
		return content, protocolMatrixUsage{input: response.Usage.Input, output: response.Usage.Output}, nil
	default:
		return "", protocolMatrixUsage{}, fmt.Errorf("unsupported response format %q", format)
	}
}

type protocolMatrixToolCall struct {
	id        string
	name      string
	arguments string
}

func decodeProtocolMatrixToolBuffered(format string, body []byte) (protocolMatrixToolCall, protocolMatrixUsage, error) {
	switch format {
	case matrixOpenAIChat:
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
		if err := json.Unmarshal(body, &response); err != nil || len(response.Choices) != 1 || len(response.Choices[0].Message.ToolCalls) != 1 {
			return protocolMatrixToolCall{}, protocolMatrixUsage{}, fmt.Errorf("invalid Chat Completions tool response")
		}
		call := response.Choices[0].Message.ToolCalls[0]
		return protocolMatrixToolCall{id: call.ID, name: call.Function.Name, arguments: call.Function.Arguments}, protocolMatrixUsage{input: response.Usage.Input, output: response.Usage.Output}, nil
	case matrixOpenAIResponses:
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
				return protocolMatrixToolCall{id: item.CallID, name: item.Name, arguments: item.Arguments}, protocolMatrixUsage{input: response.Usage.Input, output: response.Usage.Output}, nil
			}
		}
		return protocolMatrixToolCall{}, protocolMatrixUsage{}, fmt.Errorf("Responses tool call is missing")
	case matrixAnthropic:
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
				return protocolMatrixToolCall{id: block.ID, name: block.Name, arguments: string(block.Input)}, protocolMatrixUsage{input: response.Usage.Input, output: response.Usage.Output}, nil
			}
		}
		return protocolMatrixToolCall{}, protocolMatrixUsage{}, fmt.Errorf("Messages tool call is missing")
	default:
		return protocolMatrixToolCall{}, protocolMatrixUsage{}, fmt.Errorf("unsupported response format %q", format)
	}
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

func decodeProtocolMatrixChatToolStream(frames []protocolMatrixSSEFrame) (protocolMatrixToolCall, protocolMatrixUsage, error) {
	call := protocolMatrixToolCall{}
	usage := protocolMatrixUsage{}
	done := 0
	for _, frame := range frames {
		if string(bytes.TrimSpace(frame.data)) == "[DONE]" {
			done++
			continue
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					ToolCalls []struct {
						ID       string `json:"id"`
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
			} `json:"choices"`
			Usage *struct {
				Input  int64 `json:"prompt_tokens"`
				Output int64 `json:"completion_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal(frame.data, &chunk); err != nil {
			return protocolMatrixToolCall{}, protocolMatrixUsage{}, err
		}
		for _, choice := range chunk.Choices {
			for _, delta := range choice.Delta.ToolCalls {
				if delta.ID != "" {
					call.id = delta.ID
				}
				if delta.Function.Name != "" {
					call.name = delta.Function.Name
				}
				call.arguments += delta.Function.Arguments
			}
		}
		if chunk.Usage != nil {
			usage = protocolMatrixUsage{input: chunk.Usage.Input, output: chunk.Usage.Output}
		}
	}
	if done != 1 {
		return protocolMatrixToolCall{}, protocolMatrixUsage{}, fmt.Errorf("expected one Chat Completions terminal, got %d", done)
	}
	return call, usage, nil
}

func decodeProtocolMatrixResponsesToolStream(frames []protocolMatrixSSEFrame) (protocolMatrixToolCall, protocolMatrixUsage, error) {
	call := protocolMatrixToolCall{}
	usage := protocolMatrixUsage{}
	terminal := 0
	for _, frame := range frames {
		var event struct {
			Type  string `json:"type"`
			Delta string `json:"delta"`
			Item  *struct {
				Type      string `json:"type"`
				CallID    string `json:"call_id"`
				Name      string `json:"name"`
				Arguments string `json:"arguments"`
			} `json:"item"`
			Response *struct {
				Usage *struct {
					Input  int64 `json:"input_tokens"`
					Output int64 `json:"output_tokens"`
				} `json:"usage"`
			} `json:"response"`
		}
		if err := json.Unmarshal(frame.data, &event); err != nil {
			return protocolMatrixToolCall{}, protocolMatrixUsage{}, err
		}
		if event.Type == "" {
			event.Type = frame.event
		}
		switch event.Type {
		case "response.output_item.added":
			if event.Item != nil && event.Item.Type == "function_call" {
				call.id, call.name, call.arguments = event.Item.CallID, event.Item.Name, event.Item.Arguments
			}
		case "response.function_call_arguments.delta":
			call.arguments += event.Delta
		case "response.output_item.done":
			if event.Item != nil && event.Item.Type == "function_call" {
				call.id, call.name, call.arguments = event.Item.CallID, event.Item.Name, event.Item.Arguments
			}
		case "response.completed":
			terminal++
			if event.Response != nil && event.Response.Usage != nil {
				usage = protocolMatrixUsage{input: event.Response.Usage.Input, output: event.Response.Usage.Output}
			}
		case "response.failed":
			return protocolMatrixToolCall{}, protocolMatrixUsage{}, fmt.Errorf("Responses tool stream failed")
		}
	}
	if terminal != 1 {
		return protocolMatrixToolCall{}, protocolMatrixUsage{}, fmt.Errorf("expected one Responses terminal, got %d", terminal)
	}
	return call, usage, nil
}

func decodeProtocolMatrixAnthropicToolStream(frames []protocolMatrixSSEFrame) (protocolMatrixToolCall, protocolMatrixUsage, error) {
	call := protocolMatrixToolCall{}
	usage := protocolMatrixUsage{}
	terminal := 0
	for _, frame := range frames {
		var event struct {
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
		if err := json.Unmarshal(frame.data, &event); err != nil {
			return protocolMatrixToolCall{}, protocolMatrixUsage{}, err
		}
		if event.Type == "" {
			event.Type = frame.event
		}
		switch event.Type {
		case "message_start":
			if event.Message != nil && event.Message.Usage != nil {
				usage.input = event.Message.Usage.Input
			}
		case "content_block_start":
			if event.ContentBlock != nil && event.ContentBlock.Type == "tool_use" {
				call.id, call.name = event.ContentBlock.ID, event.ContentBlock.Name
				if value := string(event.ContentBlock.Input); value != "" && value != "{}" {
					call.arguments = value
				}
			}
		case "content_block_delta":
			if event.Delta != nil && event.Delta.Type == "input_json_delta" {
				call.arguments += event.Delta.PartialJSON
			}
		case "message_delta":
			if event.Usage != nil {
				usage.output = event.Usage.Output
			}
		case "message_stop":
			terminal++
		case "error":
			return protocolMatrixToolCall{}, protocolMatrixUsage{}, fmt.Errorf("Messages tool stream failed")
		}
	}
	if terminal != 1 {
		return protocolMatrixToolCall{}, protocolMatrixUsage{}, fmt.Errorf("expected one Messages terminal, got %d", terminal)
	}
	return call, usage, nil
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

type protocolMatrixSSEFrame struct {
	event string
	data  []byte
}

func parseProtocolMatrixSSE(body []byte) ([]protocolMatrixSSEFrame, error) {
	normalized := strings.ReplaceAll(string(body), "\r\n", "\n")
	normalized = strings.ReplaceAll(normalized, "\r", "\n")
	frames := make([]protocolMatrixSSEFrame, 0)
	for _, rawFrame := range strings.Split(normalized, "\n\n") {
		if strings.TrimSpace(rawFrame) == "" {
			continue
		}
		var frame protocolMatrixSSEFrame
		var data []string
		for _, line := range strings.Split(rawFrame, "\n") {
			switch {
			case strings.HasPrefix(line, "event:"):
				frame.event = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
			case strings.HasPrefix(line, "data:"):
				data = append(data, strings.TrimPrefix(strings.TrimPrefix(line, "data:"), " "))
			}
		}
		if len(data) == 0 {
			continue
		}
		frame.data = []byte(strings.Join(data, "\n"))
		frames = append(frames, frame)
	}
	if len(frames) == 0 {
		return nil, fmt.Errorf("empty SSE stream")
	}
	return frames, nil
}

func decodeProtocolMatrixStream(format string, body []byte) (string, protocolMatrixUsage, error) {
	frames, err := parseProtocolMatrixSSE(body)
	if err != nil {
		return "", protocolMatrixUsage{}, err
	}
	switch format {
	case matrixOpenAIChat:
		return decodeProtocolMatrixChatStream(frames)
	case matrixOpenAIResponses:
		return decodeProtocolMatrixResponsesStream(frames)
	case matrixAnthropic:
		return decodeProtocolMatrixAnthropicStream(frames)
	default:
		return "", protocolMatrixUsage{}, fmt.Errorf("unsupported stream format %q", format)
	}
}

func decodeProtocolMatrixChatStream(frames []protocolMatrixSSEFrame) (string, protocolMatrixUsage, error) {
	content := ""
	usage := protocolMatrixUsage{}
	done := 0
	for _, frame := range frames {
		if string(bytes.TrimSpace(frame.data)) == "[DONE]" {
			done++
			continue
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
			Usage *struct {
				Input  int64 `json:"prompt_tokens"`
				Output int64 `json:"completion_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal(frame.data, &chunk); err != nil {
			return "", protocolMatrixUsage{}, fmt.Errorf("invalid Chat Completions SSE data: %w", err)
		}
		for _, choice := range chunk.Choices {
			content += choice.Delta.Content
		}
		if chunk.Usage != nil {
			usage = protocolMatrixUsage{input: chunk.Usage.Input, output: chunk.Usage.Output}
		}
	}
	if done != 1 {
		return "", protocolMatrixUsage{}, fmt.Errorf("expected one Chat Completions terminal, got %d", done)
	}
	return content, usage, nil
}

func decodeProtocolMatrixResponsesStream(frames []protocolMatrixSSEFrame) (string, protocolMatrixUsage, error) {
	content := ""
	usage := protocolMatrixUsage{}
	terminal := 0
	for _, frame := range frames {
		var event struct {
			Type     string `json:"type"`
			Delta    string `json:"delta"`
			Response *struct {
				Usage *struct {
					Input  int64 `json:"input_tokens"`
					Output int64 `json:"output_tokens"`
				} `json:"usage"`
			} `json:"response"`
		}
		if err := json.Unmarshal(frame.data, &event); err != nil {
			return "", protocolMatrixUsage{}, fmt.Errorf("invalid Responses SSE data: %w", err)
		}
		if event.Type == "" {
			event.Type = frame.event
		}
		switch event.Type {
		case "response.output_text.delta":
			content += event.Delta
		case "response.completed":
			terminal++
			if event.Response != nil && event.Response.Usage != nil {
				usage = protocolMatrixUsage{
					input: event.Response.Usage.Input, output: event.Response.Usage.Output,
				}
			}
		case "response.failed":
			return "", protocolMatrixUsage{}, fmt.Errorf("Responses stream failed")
		}
	}
	if terminal != 1 {
		return "", protocolMatrixUsage{}, fmt.Errorf("expected one Responses terminal, got %d", terminal)
	}
	return content, usage, nil
}

func decodeProtocolMatrixAnthropicStream(frames []protocolMatrixSSEFrame) (string, protocolMatrixUsage, error) {
	content := ""
	usage := protocolMatrixUsage{}
	terminal := 0
	for _, frame := range frames {
		var event struct {
			Type    string `json:"type"`
			Message *struct {
				Usage *struct {
					Input int64 `json:"input_tokens"`
				} `json:"usage"`
			} `json:"message"`
			Delta *struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"delta"`
			Usage *struct {
				Input  int64 `json:"input_tokens"`
				Output int64 `json:"output_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal(frame.data, &event); err != nil {
			return "", protocolMatrixUsage{}, fmt.Errorf("invalid Messages SSE data: %w", err)
		}
		if event.Type == "" {
			event.Type = frame.event
		}
		switch event.Type {
		case "message_start":
			if event.Message != nil && event.Message.Usage != nil {
				usage.input = event.Message.Usage.Input
			}
		case "content_block_delta":
			if event.Delta != nil && event.Delta.Type == "text_delta" {
				content += event.Delta.Text
			}
		case "message_delta":
			if event.Usage != nil {
				if event.Usage.Input > 0 {
					usage.input = event.Usage.Input
				}
				usage.output = event.Usage.Output
			}
		case "message_stop":
			terminal++
		case "error":
			return "", protocolMatrixUsage{}, fmt.Errorf("Messages stream failed")
		}
	}
	if terminal != 1 {
		return "", protocolMatrixUsage{}, fmt.Errorf("expected one Messages terminal, got %d", terminal)
	}
	return content, usage, nil
}

func assertProtocolMatrixMarker(content string, expectedBackendFormat string) error {
	var marker struct {
		Mock     string `json:"mock"`
		Protocol string `json:"protocol"`
	}
	if err := json.Unmarshal([]byte(content), &marker); err != nil {
		return fmt.Errorf("upstream marker is not valid JSON: %w", err)
	}
	if marker.Mock != "mock-vllm" || marker.Protocol != expectedBackendFormat {
		return fmt.Errorf(
			"upstream marker = {mock:%q protocol:%q}, want protocol %q",
			marker.Mock, marker.Protocol, expectedBackendFormat,
		)
	}
	return nil
}
