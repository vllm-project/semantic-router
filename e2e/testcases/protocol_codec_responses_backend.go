package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

const (
	chatBackendModel            = "openai/gpt-oss-20b"
	nativeResponsesBackendModel = "mock/native-responses"
)

func init() {
	pkgtestcases.Register("protocol-codec-chat-backend-buffered-matrix", pkgtestcases.TestCase{
		Description: "All client protocols round-trip through a native Chat Completions backend",
		Tags:        []string{"protocol-codec", "response-api", "matrix"},
		Fn:          testProtocolCodecChatBackendBufferedMatrix,
	})
	pkgtestcases.Register("protocol-codec-chat-backend-streaming-matrix", pkgtestcases.TestCase{
		Description: "All client streaming protocols round-trip through a native Chat Completions backend",
		Tags:        []string{"protocol-codec", "response-api", "matrix", "streaming"},
		Fn:          testProtocolCodecChatBackendStreamingMatrix,
	})
	pkgtestcases.Register("protocol-codec-responses-backend-buffered-matrix", pkgtestcases.TestCase{
		Description: "All client protocols round-trip through a native Responses backend",
		Tags:        []string{"protocol-codec", "response-api", "matrix"},
		Fn:          testProtocolCodecResponsesBackendBufferedMatrix,
	})
	pkgtestcases.Register("protocol-codec-responses-backend-streaming-matrix", pkgtestcases.TestCase{
		Description: "All client streaming protocols round-trip through a native Responses backend",
		Tags:        []string{"protocol-codec", "response-api", "matrix", "streaming"},
		Fn:          testProtocolCodecResponsesBackendStreamingMatrix,
	})
	pkgtestcases.Register("protocol-codec-anthropic-backend-buffered-matrix", pkgtestcases.TestCase{
		Description: "All client protocols round-trip through a native Anthropic Messages backend",
		Tags:        []string{"protocol-codec", "response-api", "anthropic", "matrix"},
		Fn:          testProtocolCodecAnthropicBackendBufferedMatrix,
	})
	pkgtestcases.Register("protocol-codec-anthropic-backend-streaming-matrix", pkgtestcases.TestCase{
		Description: "All client streaming protocols round-trip through a native Anthropic Messages backend",
		Tags:        []string{"protocol-codec", "response-api", "anthropic", "matrix", "streaming"},
		Fn:          testProtocolCodecAnthropicBackendStreamingMatrix,
	})
	pkgtestcases.Register("protocol-codec-incomplete-stream-terminal", pkgtestcases.TestCase{
		Description: "A Responses backend stream that closes early fails every client protocol",
		Tags:        []string{"protocol-codec", "response-api", "streaming", "failure"},
		Fn:          testProtocolCodecIncompleteStreamTerminal,
	})
	pkgtestcases.Register("protocol-codec-chat-backend-incomplete-stream-matrix", pkgtestcases.TestCase{
		Description: "A Chat Completions backend stream that closes early fails every client protocol",
		Tags:        []string{"protocol-codec", "response-api", "streaming", "failure", "matrix"},
		Fn:          testProtocolCodecChatBackendIncompleteStreamMatrix,
	})
	pkgtestcases.Register("protocol-codec-anthropic-backend-incomplete-stream-matrix", pkgtestcases.TestCase{
		Description: "An Anthropic Messages backend stream that closes early fails every client protocol",
		Tags:        []string{"protocol-codec", "response-api", "anthropic", "streaming", "failure", "matrix"},
		Fn:          testProtocolCodecAnthropicBackendIncompleteStreamMatrix,
	})
	pkgtestcases.Register("protocol-codec-chat-backend-midstream-error-matrix", pkgtestcases.TestCase{
		Description: "A Chat Completions backend error after partial output terminates every client protocol",
		Tags:        []string{"protocol-codec", "response-api", "streaming", "failure", "matrix"},
		Fn:          testProtocolCodecChatBackendMidstreamErrorMatrix,
	})
	pkgtestcases.Register("protocol-codec-responses-backend-midstream-error-matrix", pkgtestcases.TestCase{
		Description: "A Responses backend error after partial output terminates every client protocol",
		Tags:        []string{"protocol-codec", "response-api", "streaming", "failure", "matrix"},
		Fn:          testProtocolCodecResponsesBackendMidstreamErrorMatrix,
	})
	pkgtestcases.Register("protocol-codec-anthropic-backend-midstream-error-matrix", pkgtestcases.TestCase{
		Description: "An Anthropic Messages backend error after partial output terminates every client protocol",
		Tags:        []string{"protocol-codec", "response-api", "anthropic", "streaming", "failure", "matrix"},
		Fn:          testProtocolCodecAnthropicBackendMidstreamErrorMatrix,
	})
	pkgtestcases.Register("protocol-codec-chat-backend-tool-lifecycle", pkgtestcases.TestCase{
		Description: "All client protocols preserve buffered and streamed tool calls and results through a Chat Completions backend",
		Tags:        []string{"protocol-codec", "response-api", "tools", "matrix", "streaming"},
		Fn:          testProtocolCodecChatBackendToolLifecycle,
	})
	pkgtestcases.Register("protocol-codec-responses-backend-tool-lifecycle", pkgtestcases.TestCase{
		Description: "All client protocols preserve buffered and streamed tool calls and results through a native Responses backend",
		Tags:        []string{"protocol-codec", "response-api", "tools", "matrix", "streaming"},
		Fn:          testProtocolCodecResponsesBackendToolLifecycle,
	})
	pkgtestcases.Register("protocol-codec-anthropic-backend-tool-lifecycle", pkgtestcases.TestCase{
		Description: "All client protocols preserve buffered and streamed tool calls and results through an Anthropic Messages backend",
		Tags:        []string{"protocol-codec", "response-api", "anthropic", "tools", "matrix", "streaming"},
		Fn:          testProtocolCodecAnthropicBackendToolLifecycle,
	})
	pkgtestcases.Register("protocol-codec-chat-backend-error-matrix", pkgtestcases.TestCase{
		Description: "Provider failures remain failures for every client protocol through a Chat Completions backend",
		Tags:        []string{"protocol-codec", "response-api", "errors", "matrix"},
		Fn:          testProtocolCodecChatBackendErrorMatrix,
	})
	pkgtestcases.Register("protocol-codec-responses-backend-error-matrix", pkgtestcases.TestCase{
		Description: "Provider failures remain failures for every client protocol through a Responses backend",
		Tags:        []string{"protocol-codec", "response-api", "errors", "matrix"},
		Fn:          testProtocolCodecResponsesBackendErrorMatrix,
	})
	pkgtestcases.Register("protocol-codec-anthropic-backend-error-matrix", pkgtestcases.TestCase{
		Description: "Provider failures remain failures for every client protocol through an Anthropic Messages backend",
		Tags:        []string{"protocol-codec", "response-api", "anthropic", "errors", "matrix"},
		Fn:          testProtocolCodecAnthropicBackendErrorMatrix,
	})
}

func testProtocolCodecChatBackendBufferedMatrix(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	return runProtocolCodecBackendBufferedMatrix(ctx, client, opts, chatBackendModel, "openai.chat.v1")
}

func testProtocolCodecResponsesBackendBufferedMatrix(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	return runProtocolCodecBackendBufferedMatrix(ctx, client, opts, nativeResponsesBackendModel, "openai.responses.v1")
}

func testProtocolCodecAnthropicBackendBufferedMatrix(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	return runProtocolCodecBackendBufferedMatrix(ctx, client, opts, "MoM", "anthropic.messages.v1")
}

func runProtocolCodecBackendBufferedMatrix(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
	model string,
	backendFormat string,
) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	checks := []struct {
		name string
		path string
		body any
		want func([]byte) error
	}{
		{
			name: "chat_completions",
			path: "/v1/chat/completions",
			body: map[string]any{
				"model":    model,
				"messages": []map[string]string{{"role": "user", "content": "hello from chat"}},
			},
			want: assertChatCompletionBody,
		},
		{
			name: "anthropic_messages",
			path: "/v1/messages",
			body: map[string]any{
				"model": model, "max_tokens": 64,
				"messages": []map[string]string{{"role": "user", "content": "hello from messages"}},
			},
			want: assertAnthropicBody,
		},
		{
			name: "openai_responses",
			path: "/v1/responses",
			body: map[string]any{
				"model": model, "input": "hello from responses", "store": false,
			},
			want: assertResponsesBody,
		},
	}
	for _, check := range checks {
		body, requestErr := sendProtocolMatrixRequest(ctx, session, check.path, check.body, false)
		if requestErr != nil {
			return fmt.Errorf("%s: %w", check.name, requestErr)
		}
		if shapeErr := check.want(body); shapeErr != nil {
			return fmt.Errorf("%s: %w", check.name, shapeErr)
		}
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"backend_format": backendFormat, "client_formats": len(checks)})
	}
	return nil
}

func testProtocolCodecChatBackendStreamingMatrix(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	return runProtocolCodecBackendStreamingMatrix(ctx, client, opts, chatBackendModel, "openai.chat.v1")
}

func testProtocolCodecResponsesBackendStreamingMatrix(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	return runProtocolCodecBackendStreamingMatrix(ctx, client, opts, nativeResponsesBackendModel, "openai.responses.v1")
}

func testProtocolCodecAnthropicBackendStreamingMatrix(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	return runProtocolCodecBackendStreamingMatrix(ctx, client, opts, "MoM", "anthropic.messages.v1")
}

func runProtocolCodecBackendStreamingMatrix(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
	model string,
	backendFormat string,
) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	chat, err := sendProtocolMatrixRequest(ctx, session, "/v1/chat/completions", map[string]any{
		"model":    model,
		"messages": []map[string]string{{"role": "user", "content": "stream chat"}},
		"stream":   true,
	}, true)
	if err != nil {
		return fmt.Errorf("chat stream: %w", err)
	}
	if !bytes.Contains(chat, []byte("chat.completion.chunk")) || !bytes.Contains(chat, []byte("data: [DONE]")) || bytes.Contains(chat, []byte("response.output_text.delta")) {
		return fmt.Errorf("chat stream leaked the backend protocol: %s", truncateString(string(chat), 500))
	}

	messages, err := sendProtocolMatrixRequest(ctx, session, "/v1/messages", map[string]any{
		"model": model, "max_tokens": 64,
		"messages": []map[string]string{{"role": "user", "content": "stream messages"}},
		"stream":   true,
	}, true)
	if err != nil {
		return fmt.Errorf("messages stream: %w", err)
	}
	if !bytes.Contains(messages, []byte("event: message_start")) || !bytes.Contains(messages, []byte("event: message_stop")) || bytes.Contains(messages, []byte("response.output_text.delta")) {
		return fmt.Errorf("Messages stream leaked the backend protocol: %s", truncateString(string(messages), 500))
	}

	responses, err := sendProtocolMatrixRequest(ctx, session, "/v1/responses", map[string]any{
		"model": model, "input": "stream responses", "stream": true, "store": false,
	}, true)
	if err != nil {
		return fmt.Errorf("Responses stream: %w", err)
	}
	if err := validateResponseAPIStreamingSSEBody(string(responses)); err != nil {
		return fmt.Errorf("Responses stream: %w", err)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"backend_format": backendFormat, "streaming_client_formats": 3})
	}
	return nil
}

func testProtocolCodecIncompleteStreamTerminal(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	return runProtocolCodecIncompleteStreamMatrix(ctx, client, opts, nativeResponsesBackendModel, "openai.responses.v1")
}

func testProtocolCodecChatBackendIncompleteStreamMatrix(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	return runProtocolCodecIncompleteStreamMatrix(ctx, client, opts, chatBackendModel, "openai.chat.v1")
}

func testProtocolCodecAnthropicBackendIncompleteStreamMatrix(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	return runProtocolCodecIncompleteStreamMatrix(ctx, client, opts, "MoM", "anthropic.messages.v1")
}

func testProtocolCodecChatBackendMidstreamErrorMatrix(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	return runProtocolCodecMidstreamErrorMatrix(ctx, client, opts, chatBackendModel, "openai.chat.v1")
}

func testProtocolCodecResponsesBackendMidstreamErrorMatrix(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	return runProtocolCodecMidstreamErrorMatrix(ctx, client, opts, nativeResponsesBackendModel, "openai.responses.v1")
}

func testProtocolCodecAnthropicBackendMidstreamErrorMatrix(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	return runProtocolCodecMidstreamErrorMatrix(ctx, client, opts, "MoM", "anthropic.messages.v1")
}

func runProtocolCodecIncompleteStreamMatrix(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
	model string,
	backendFormat string,
) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	checks := []struct {
		name   string
		path   string
		body   any
		fail   string
		done   string
		detail string
	}{
		{
			name: "chat_completions", path: "/v1/chat/completions",
			body: map[string]any{
				"model": model, "stream": true,
				"messages": []map[string]string{{"role": "user", "content": "__mock_incomplete_stream__"}},
			},
			fail: `"code":"stream_incomplete"`, done: `"finish_reason":"stop"`, detail: "stream_incomplete",
		},
		{
			name: "anthropic_messages", path: "/v1/messages",
			body: map[string]any{
				"model": model, "max_tokens": 16, "stream": true,
				"messages": []map[string]string{{"role": "user", "content": "__mock_incomplete_stream__"}},
			},
			fail: "event: error", done: "event: message_stop", detail: "upstream stream ended before completion",
		},
		{
			name: "openai_responses", path: "/v1/responses",
			body: map[string]any{
				"model": model, "input": "__mock_incomplete_stream__", "stream": true, "store": false,
			},
			fail: "event: error", done: "event: response.completed", detail: "stream_incomplete",
		},
	}
	for _, check := range checks {
		body, requestErr := sendProtocolMatrixRequest(ctx, session, check.path, check.body, true)
		if requestErr != nil {
			return fmt.Errorf("%s: %w", check.name, requestErr)
		}
		stream := string(body)
		if !strings.Contains(stream, check.fail) || !strings.Contains(stream, check.detail) {
			return fmt.Errorf("%s: incomplete stream has no native failure terminal: %s", check.name, truncateString(stream, 800))
		}
		if strings.Contains(stream, check.done) {
			return fmt.Errorf("%s: incomplete stream was reported as successful: %s", check.name, truncateString(stream, 800))
		}
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"backend_format": backendFormat, "failure_client_formats": len(checks)})
	}
	return nil
}

func runProtocolCodecMidstreamErrorMatrix(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
	model string,
	backendFormat string,
) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	checks := []struct {
		name           string
		path           string
		body           any
		failureMarkers []string
		successMarker  string
	}{
		{
			name: "chat_completions", path: "/v1/chat/completions",
			body: map[string]any{
				"model": model, "stream": true,
				"messages": []map[string]string{{"role": "user", "content": "__mock_midstream_error__"}},
			},
			failureMarkers: []string{`"error":`, "mock provider stream failed"},
			successMarker:  "data: [DONE]",
		},
		{
			name: "anthropic_messages", path: "/v1/messages",
			body: map[string]any{
				"model": model, "max_tokens": 16, "stream": true,
				"messages": []map[string]string{{"role": "user", "content": "__mock_midstream_error__"}},
			},
			failureMarkers: []string{"event: error", "mock provider stream failed"},
			successMarker:  "event: message_stop",
		},
		{
			name: "openai_responses", path: "/v1/responses",
			body: map[string]any{
				"model": model, "input": "__mock_midstream_error__", "stream": true, "store": false,
			},
			failureMarkers: []string{"mock provider stream failed"},
			successMarker:  "event: response.completed",
		},
	}
	for _, check := range checks {
		body, requestErr := sendProtocolMatrixRequest(ctx, session, check.path, check.body, true)
		if requestErr != nil {
			return fmt.Errorf("%s: %w", check.name, requestErr)
		}
		stream := string(body)
		for _, marker := range check.failureMarkers {
			if !strings.Contains(stream, marker) {
				return fmt.Errorf("%s: midstream failure marker %q is missing: %s", check.name, marker, truncateString(stream, 1000))
			}
		}
		if count := strings.Count(stream, "mock provider stream failed"); count != 1 {
			return fmt.Errorf("%s: provider failure was emitted %d times, want exactly one: %s", check.name, count, truncateString(stream, 1000))
		}
		if check.name == "openai_responses" &&
			!strings.Contains(stream, "event: error") &&
			!strings.Contains(stream, "event: response.failed") {
			return fmt.Errorf("%s: no Responses failure terminal: %s", check.name, truncateString(stream, 1000))
		}
		if strings.Contains(stream, check.successMarker) {
			return fmt.Errorf("%s: midstream failure was followed by success: %s", check.name, truncateString(stream, 1000))
		}
		partialIndex := strings.Index(stream, "partial")
		failureIndex := strings.Index(stream, "mock provider stream failed")
		if partialIndex < 0 || failureIndex < 0 || partialIndex >= failureIndex {
			return fmt.Errorf("%s: partial output and failure are out of order: %s", check.name, truncateString(stream, 1000))
		}
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"backend_format":                   backendFormat,
			"midstream_failure_client_formats": len(checks),
		})
	}
	return nil
}

func testProtocolCodecChatBackendToolLifecycle(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	return runProtocolCodecToolLifecycle(ctx, client, opts, chatBackendModel, "openai.chat.v1")
}

func testProtocolCodecResponsesBackendToolLifecycle(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	return runProtocolCodecToolLifecycle(ctx, client, opts, nativeResponsesBackendModel, "openai.responses.v1")
}

func testProtocolCodecAnthropicBackendToolLifecycle(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	return runProtocolCodecToolLifecycle(ctx, client, opts, "MoM", "anthropic.messages.v1")
}

func testProtocolCodecChatBackendErrorMatrix(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	return runProtocolCodecErrorMatrix(ctx, client, opts, chatBackendModel, "openai.chat.v1")
}

func testProtocolCodecResponsesBackendErrorMatrix(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	return runProtocolCodecErrorMatrix(ctx, client, opts, nativeResponsesBackendModel, "openai.responses.v1")
}

func testProtocolCodecAnthropicBackendErrorMatrix(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	return runProtocolCodecErrorMatrix(ctx, client, opts, "MoM", "anthropic.messages.v1")
}

func runProtocolCodecErrorMatrix(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
	model string,
	backendFormat string,
) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	checks := []struct {
		name      string
		path      string
		body      any
		anthropic bool
	}{
		{
			name: "chat_completions", path: "/v1/chat/completions",
			body: map[string]any{
				"model":    model,
				"messages": []map[string]string{{"role": "user", "content": "__mock_provider_error__"}},
			},
		},
		{
			name: "anthropic_messages", path: "/v1/messages", anthropic: true,
			body: map[string]any{
				"model": model, "max_tokens": 16,
				"messages": []map[string]string{{"role": "user", "content": "__mock_provider_error__"}},
			},
		},
		{
			name: "openai_responses", path: "/v1/responses",
			body: map[string]any{"model": model, "input": "__mock_provider_error__", "store": false},
		},
	}
	for _, check := range checks {
		result, requestErr := sendProtocolMatrixRaw(ctx, session, check.path, check.body, false, nil)
		if requestErr != nil {
			return fmt.Errorf("%s: %w", check.name, requestErr)
		}
		if result.StatusCode != http.StatusTooManyRequests {
			return fmt.Errorf("%s: status = %d, want 429: %s", check.name, result.StatusCode, truncateString(string(result.Body), 500))
		}
		expectedCode := "rate_limit_exceeded"
		if backendFormat == "anthropic.messages.v1" {
			expectedCode = "rate_limit_error"
		}
		if shapeErr := assertProtocolMatrixRateLimit(result.Body, check.anthropic, expectedCode); shapeErr != nil {
			return fmt.Errorf("%s: %w", check.name, shapeErr)
		}
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"backend_format": backendFormat, "failure_client_formats": len(checks)})
	}
	return nil
}

func assertProtocolMatrixRateLimit(body []byte, anthropic bool, expectedCode string) error {
	var envelope struct {
		Type  string `json:"type"`
		Error struct {
			Type    string `json:"type"`
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		return err
	}
	if envelope.Error.Type != "rate_limit_error" || envelope.Error.Message != "mock provider rate limit" {
		return fmt.Errorf("invalid rate-limit error: %s", truncateString(string(body), 500))
	}
	if anthropic {
		if envelope.Type != "error" {
			return fmt.Errorf("Anthropic error envelope type = %q: %s", envelope.Type, truncateString(string(body), 500))
		}
		return nil
	}
	if envelope.Error.Code != expectedCode {
		return fmt.Errorf("OpenAI error code = %q, want %q: %s", envelope.Error.Code, expectedCode, truncateString(string(body), 500))
	}
	return nil
}

func sendProtocolMatrixRequest(
	ctx context.Context,
	session *fixtures.ServiceSession,
	path string,
	body any,
	stream bool,
) ([]byte, error) {
	return sendProtocolMatrixRequestWithHeaders(ctx, session, path, body, stream, nil)
}

func sendProtocolMatrixRequestWithHeaders(
	ctx context.Context,
	session *fixtures.ServiceSession,
	path string,
	body any,
	stream bool,
	headers map[string]string,
) ([]byte, error) {
	result, err := sendProtocolMatrixRaw(ctx, session, path, body, stream, headers)
	if err != nil {
		return nil, err
	}
	if result.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d: %s", result.StatusCode, truncateString(string(result.Body), 500))
	}
	return result.Body, nil
}

type protocolMatrixHTTPResult struct {
	StatusCode int
	Body       []byte
}

func sendProtocolMatrixRaw(
	ctx context.Context,
	session *fixtures.ServiceSession,
	path string,
	body any,
	stream bool,
	headers map[string]string,
) (protocolMatrixHTTPResult, error) {
	encoded, err := json.Marshal(body)
	if err != nil {
		return protocolMatrixHTTPResult{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, session.BaseURL()+path, bytes.NewReader(encoded))
	if err != nil {
		return protocolMatrixHTTPResult{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	if path == "/v1/messages" {
		req.Header.Set("anthropic-version", "2023-06-01")
	}
	if stream {
		req.Header.Set("Accept", "text/event-stream")
	}
	for key, value := range headers {
		req.Header.Set(key, value)
	}
	resp, err := session.HTTPClient(45 * time.Second).Do(req)
	if err != nil {
		return protocolMatrixHTTPResult{}, err
	}
	defer resp.Body.Close()
	responseBody, readErr := io.ReadAll(resp.Body)
	if readErr != nil {
		return protocolMatrixHTTPResult{}, readErr
	}
	return protocolMatrixHTTPResult{StatusCode: resp.StatusCode, Body: responseBody}, nil
}

func assertChatCompletionBody(body []byte) error {
	var response struct {
		Object  string `json:"object"`
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		return err
	}
	if response.Object != "chat.completion" || len(response.Choices) != 1 || !strings.Contains(response.Choices[0].Message.Content, `"mock":"mock-vllm"`) {
		return fmt.Errorf("invalid Chat Completions response: %s", truncateString(string(body), 500))
	}
	return nil
}

func assertAnthropicBody(body []byte) error {
	var response anthropicMessageResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return err
	}
	return assertAnthropicMessageShape(response)
}

func assertResponsesBody(body []byte) error {
	var response fixtures.ResponseAPIResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return err
	}
	if response.Object != "response" || response.Status != "completed" || !strings.Contains(response.OutputText, `"mock":"mock-vllm"`) {
		return fmt.Errorf("invalid Responses response: %s", truncateString(string(body), 500))
	}
	return nil
}
