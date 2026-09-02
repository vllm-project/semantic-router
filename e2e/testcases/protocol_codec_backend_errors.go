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
