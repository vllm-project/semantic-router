package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
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
	protocolCodecChatReply      = `"protocol":"chat_completions"`
	protocolCodecResponsesReply = `"protocol":"responses"`
	protocolCodecAnthropicProbe = "__mock_protocol_matrix__"
	protocolCodecAnthropicReply = `"protocol":"anthropic_messages"`
)

type protocolCodecE2EClient struct {
	name string
	path string
}

var protocolCodecE2EClients = []protocolCodecE2EClient{
	{name: "chat_completions", path: "/v1/chat/completions"},
	{name: "openai_responses", path: "/v1/responses"},
	{name: "anthropic_messages", path: "/v1/messages"},
}

func (client protocolCodecE2EClient) request(model, prompt string, stream bool) map[string]any {
	switch client.path {
	case "/v1/chat/completions":
		return map[string]any{
			"model": model, "stream": stream,
			"messages": []map[string]string{{"role": "user", "content": prompt}},
		}
	case "/v1/responses":
		return map[string]any{
			"model": model, "input": prompt, "store": false, "stream": stream,
		}
	case "/v1/messages":
		return map[string]any{
			"model": model, "max_tokens": 64, "stream": stream,
			"messages": []map[string]string{{"role": "user", "content": prompt}},
		}
	default:
		panic("unregistered protocol codec E2E client path: " + client.path)
	}
}

func (client protocolCodecE2EClient) validateBuffered(body []byte, expectedText string) error {
	switch client.path {
	case "/v1/chat/completions":
		return assertChatCompletionBody(body, expectedText)
	case "/v1/responses":
		return assertResponsesBody(body, expectedText)
	case "/v1/messages":
		return assertAnthropicBody(body, expectedText)
	default:
		return fmt.Errorf("unregistered protocol codec E2E client path %q", client.path)
	}
}

func (client protocolCodecE2EClient) validateStream(body []byte, expectedText string) error {
	switch client.path {
	case "/v1/chat/completions":
		if bytes.Contains(body, []byte("event: response.")) || bytes.Contains(body, []byte("event: message_start")) {
			return fmt.Errorf("Chat stream leaked a backend protocol: %s", truncateString(string(body), 800))
		}
		return assertChatTextStream(body, expectedText)
	case "/v1/responses":
		if err := validateResponseAPIStreamingSSEBody(string(body)); err != nil {
			return err
		}
		text, err := extractProtocolStructuredOutputStreamText(client.path, body)
		if err != nil {
			return err
		}
		if !strings.Contains(text, expectedText) {
			return fmt.Errorf("Responses stream lost backend output: %s", truncateString(string(body), 800))
		}
		return nil
	case "/v1/messages":
		if bytes.Contains(body, []byte("chat.completion.chunk")) || bytes.Contains(body, []byte("event: response.")) || bytes.Contains(body, []byte("data: [DONE]")) {
			return fmt.Errorf("Messages stream leaked a backend protocol: %s", truncateString(string(body), 800))
		}
		return assertAnthropicTextStream(body, expectedText)
	default:
		return fmt.Errorf("unregistered protocol codec E2E client path %q", client.path)
	}
}

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
	return runProtocolCodecBackendBufferedMatrix(
		ctx, client, opts, chatBackendModel, "openai.chat.v1", "buffered protocol matrix", protocolCodecChatReply,
	)
}

func testProtocolCodecResponsesBackendBufferedMatrix(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	return runProtocolCodecBackendBufferedMatrix(
		ctx, client, opts, nativeResponsesBackendModel, "openai.responses.v1", "buffered protocol matrix", protocolCodecResponsesReply,
	)
}

func testProtocolCodecAnthropicBackendBufferedMatrix(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	return runProtocolCodecBackendBufferedMatrix(
		ctx, client, opts, "MoM", "anthropic.messages.v1", protocolCodecAnthropicProbe, protocolCodecAnthropicReply,
	)
}

func runProtocolCodecBackendBufferedMatrix(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
	model string,
	backendFormat string,
	prompt string,
	expectedText string,
) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	var failures []error
	results := make(map[string]string, len(protocolCodecE2EClients))
	for _, clientContract := range protocolCodecE2EClients {
		body, requestErr := sendProtocolMatrixRequest(
			ctx,
			session,
			clientContract.path,
			clientContract.request(model, prompt, false),
			false,
		)
		if requestErr != nil {
			results[clientContract.name] = "failed"
			failures = append(failures, fmt.Errorf("%s: %w", clientContract.name, requestErr))
			continue
		}
		if shapeErr := clientContract.validateBuffered(body, expectedText); shapeErr != nil {
			results[clientContract.name] = "failed"
			failures = append(failures, fmt.Errorf("%s: %w", clientContract.name, shapeErr))
			continue
		}
		results[clientContract.name] = "passed"
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"backend_format": backendFormat,
			"client_formats": len(protocolCodecE2EClients),
			"mode":           "buffered",
			"matrix_cells":   results,
		})
	}
	return errors.Join(failures...)
}

func testProtocolCodecChatBackendStreamingMatrix(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	return runProtocolCodecBackendStreamingMatrix(
		ctx, client, opts, chatBackendModel, "openai.chat.v1", "streaming protocol matrix", protocolCodecChatReply,
	)
}

func testProtocolCodecResponsesBackendStreamingMatrix(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	return runProtocolCodecBackendStreamingMatrix(
		ctx, client, opts, nativeResponsesBackendModel, "openai.responses.v1", "streaming protocol matrix", protocolCodecResponsesReply,
	)
}

func testProtocolCodecAnthropicBackendStreamingMatrix(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	return runProtocolCodecBackendStreamingMatrix(
		ctx, client, opts, "MoM", "anthropic.messages.v1", protocolCodecAnthropicProbe, protocolCodecAnthropicReply,
	)
}

func runProtocolCodecBackendStreamingMatrix(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
	model string,
	backendFormat string,
	prompt string,
	expectedText string,
) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	var failures []error
	results := make(map[string]string, len(protocolCodecE2EClients))
	for _, clientContract := range protocolCodecE2EClients {
		body, requestErr := sendProtocolMatrixRequest(
			ctx,
			session,
			clientContract.path,
			clientContract.request(model, prompt, true),
			true,
		)
		if requestErr != nil {
			results[clientContract.name] = "failed"
			failures = append(failures, fmt.Errorf("%s: %w", clientContract.name, requestErr))
			continue
		}
		if shapeErr := clientContract.validateStream(body, expectedText); shapeErr != nil {
			results[clientContract.name] = "failed"
			failures = append(failures, fmt.Errorf("%s: %w", clientContract.name, shapeErr))
			continue
		}
		results[clientContract.name] = "passed"
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"backend_format":           backendFormat,
			"streaming_client_formats": len(protocolCodecE2EClients),
			"mode":                     "streaming",
			"matrix_cells":             results,
		})
	}
	return errors.Join(failures...)
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

func assertChatCompletionBody(body []byte, expectedText string) error {
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
	if response.Object != "chat.completion" || len(response.Choices) != 1 || !strings.Contains(response.Choices[0].Message.Content, expectedText) {
		return fmt.Errorf("invalid Chat Completions response: %s", truncateString(string(body), 500))
	}
	return nil
}

func assertAnthropicBody(body []byte, expectedText string) error {
	var response anthropicMessageResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return err
	}
	if err := assertAnthropicMessageShape(response); err != nil {
		return err
	}
	var content struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if len(response.Content) != 1 || json.Unmarshal(response.Content[0], &content) != nil ||
		content.Type != "text" || !strings.Contains(content.Text, expectedText) {
		return fmt.Errorf("invalid Anthropic Messages response: %s", truncateString(string(body), 500))
	}
	return nil
}

func assertResponsesBody(body []byte, expectedText string) error {
	var response fixtures.ResponseAPIResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return err
	}
	if response.Object != "response" || response.Status != "completed" || !strings.Contains(response.OutputText, expectedText) {
		return fmt.Errorf("invalid Responses response: %s", truncateString(string(body), 500))
	}
	return nil
}
