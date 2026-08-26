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
			if err := runProtocolCodecMatrixCell(
				ctx, httpClient, baseURL, sourceFormat, backend.model, backend.backendFormat, stream,
			); err != nil {
				return err
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

func runProtocolCodecMatrixCell(
	ctx context.Context,
	httpClient *http.Client,
	baseURL string,
	sourceFormat string,
	backendModel string,
	backendFormat string,
	stream bool,
) error {
	result, err := requestProtocolMatrix(ctx, httpClient, baseURL, sourceFormat, backendModel, stream)
	if err != nil {
		return fmt.Errorf("%s to %s (stream=%t): %w", sourceFormat, backendFormat, stream, err)
	}
	if result.statusCode != http.StatusOK {
		return fmt.Errorf(
			"%s to %s (stream=%t) returned %d: %s",
			sourceFormat, backendFormat, stream, result.statusCode,
			truncateString(string(result.body), 500),
		)
	}
	if stream && !strings.Contains(result.contentType, "text/event-stream") {
		return fmt.Errorf(
			"%s to %s returned streaming content type %q",
			sourceFormat, backendFormat, result.contentType,
		)
	}

	content, usage, err := decodeProtocolMatrixResult(sourceFormat, result.body, stream)
	if err != nil {
		return fmt.Errorf(
			"decode %s response from %s backend (stream=%t): %w",
			sourceFormat, backendFormat, stream, err,
		)
	}
	if err := assertProtocolMatrixMarker(content, backendFormat); err != nil {
		return fmt.Errorf("%s to %s (stream=%t): %w", sourceFormat, backendFormat, stream, err)
	}
	if usage.input <= 0 || usage.output <= 0 || usage.total() <= 0 {
		return fmt.Errorf(
			"%s to %s (stream=%t) omitted authoritative usage: %+v",
			sourceFormat, backendFormat, stream, usage,
		)
	}
	return nil
}

func decodeProtocolMatrixResult(
	format string,
	body []byte,
	stream bool,
) (string, protocolMatrixUsage, error) {
	if stream {
		return decodeProtocolMatrixStream(format, body)
	}
	return decodeProtocolMatrixBuffered(format, body)
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
			if err := runProtocolCodecToolMatrixCell(
				ctx, httpClient, baseURL, sourceFormat, backend.model, backend.backendFormat, stream,
			); err != nil {
				return err
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

func runProtocolCodecToolMatrixCell(
	ctx context.Context,
	httpClient *http.Client,
	baseURL string,
	sourceFormat string,
	backendModel string,
	backendFormat string,
	stream bool,
) error {
	result, err := requestProtocolToolMatrix(ctx, httpClient, baseURL, sourceFormat, backendModel, stream)
	if err != nil {
		return fmt.Errorf("tool %s to %s (stream=%t): %w", sourceFormat, backendFormat, stream, err)
	}
	if result.statusCode != http.StatusOK {
		return fmt.Errorf(
			"tool %s to %s (stream=%t) returned %d: %s",
			sourceFormat, backendFormat, stream, result.statusCode,
			truncateString(string(result.body), 500),
		)
	}

	call, usage, err := decodeProtocolMatrixToolResult(sourceFormat, result.body, stream)
	if err != nil {
		return fmt.Errorf(
			"decode tool %s response from %s backend (stream=%t): %w",
			sourceFormat, backendFormat, stream, err,
		)
	}
	if err := assertProtocolMatrixToolCall(call, backendFormat); err != nil {
		return fmt.Errorf("tool %s to %s (stream=%t): %w", sourceFormat, backendFormat, stream, err)
	}
	if usage.input <= 0 || usage.output <= 0 || usage.total() <= 0 {
		return fmt.Errorf(
			"tool %s to %s (stream=%t) omitted authoritative usage: %+v",
			sourceFormat, backendFormat, stream, usage,
		)
	}
	return nil
}

func decodeProtocolMatrixToolResult(
	format string,
	body []byte,
	stream bool,
) (protocolMatrixToolCall, protocolMatrixUsage, error) {
	if stream {
		return decodeProtocolMatrixToolStream(format, body)
	}
	return decodeProtocolMatrixToolBuffered(format, body)
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
		return decodeProtocolMatrixChatBuffered(body)
	case matrixOpenAIResponses:
		return decodeProtocolMatrixResponsesBuffered(body)
	case matrixAnthropic:
		return decodeProtocolMatrixAnthropicBuffered(body)
	default:
		return "", protocolMatrixUsage{}, fmt.Errorf("unsupported response format %q", format)
	}
}

func decodeProtocolMatrixChatBuffered(body []byte) (string, protocolMatrixUsage, error) {
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
}

func decodeProtocolMatrixResponsesBuffered(body []byte) (string, protocolMatrixUsage, error) {
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
}

func decodeProtocolMatrixAnthropicBuffered(body []byte) (string, protocolMatrixUsage, error) {
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

type protocolMatrixAnthropicStreamEvent struct {
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

func decodeProtocolMatrixAnthropicStream(frames []protocolMatrixSSEFrame) (string, protocolMatrixUsage, error) {
	content := ""
	usage := protocolMatrixUsage{}
	terminal := 0
	for _, frame := range frames {
		event, err := decodeProtocolMatrixAnthropicEvent(frame)
		if err != nil {
			return "", protocolMatrixUsage{}, fmt.Errorf("invalid Messages SSE data: %w", err)
		}
		eventTerminal, err := applyProtocolMatrixAnthropicEvent(event, &content, &usage)
		if err != nil {
			return "", protocolMatrixUsage{}, err
		}
		if eventTerminal {
			terminal++
		}
	}
	if terminal != 1 {
		return "", protocolMatrixUsage{}, fmt.Errorf("expected one Messages terminal, got %d", terminal)
	}
	return content, usage, nil
}

func decodeProtocolMatrixAnthropicEvent(
	frame protocolMatrixSSEFrame,
) (protocolMatrixAnthropicStreamEvent, error) {
	var event protocolMatrixAnthropicStreamEvent
	if err := json.Unmarshal(frame.data, &event); err != nil {
		return protocolMatrixAnthropicStreamEvent{}, err
	}
	event.Type = protocolMatrixEventType(event.Type, frame.event)
	return event, nil
}

func applyProtocolMatrixAnthropicEvent(
	event protocolMatrixAnthropicStreamEvent,
	content *string,
	usage *protocolMatrixUsage,
) (bool, error) {
	switch event.Type {
	case "message_start":
		applyProtocolMatrixAnthropicInputUsage(usage, event)
	case "content_block_delta":
		applyProtocolMatrixAnthropicTextDelta(content, event)
	case "message_delta":
		applyProtocolMatrixAnthropicUsage(usage, event)
	case "message_stop":
		return true, nil
	case "error":
		return false, fmt.Errorf("Messages stream failed")
	}
	return false, nil
}

func applyProtocolMatrixAnthropicInputUsage(
	usage *protocolMatrixUsage,
	event protocolMatrixAnthropicStreamEvent,
) {
	if event.Message != nil && event.Message.Usage != nil {
		usage.input = event.Message.Usage.Input
	}
}

func applyProtocolMatrixAnthropicTextDelta(
	content *string,
	event protocolMatrixAnthropicStreamEvent,
) {
	if event.Delta != nil && event.Delta.Type == "text_delta" {
		*content += event.Delta.Text
	}
}

func applyProtocolMatrixAnthropicUsage(
	usage *protocolMatrixUsage,
	event protocolMatrixAnthropicStreamEvent,
) {
	if event.Usage == nil {
		return
	}
	if event.Usage.Input > 0 {
		usage.input = event.Usage.Input
	}
	usage.output = event.Usage.Output
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
