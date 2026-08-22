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

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("anthropic-error-passthrough", pkgtestcases.TestCase{
		Description: "Verify upstream Anthropic error envelopes reach clients on both protocols instead of empty success bodies (anthropic-shim profile)",
		Tags:        []string{"anthropic", "error", "functional"},
		Fn:          testAnthropicErrorPassthrough,
	})
}

// errorSentinel makes the anthropic-shim backend answer with a proper
// Anthropic error envelope (HTTP 429, type rate_limit_error) instead of
// forwarding to the model. See _requests_forced_error in the shim.
const errorSentinel = "__VSR_E2E_FORCE_ERROR__"

// testAnthropicErrorPassthrough asserts the externally visible error
// contract on both client protocols when the Anthropic-format backend
// fails: the upstream status is forwarded and the error body survives
// translation instead of being flattened into an empty success-shaped
// response.
//
//   - /v1/messages clients must receive the Anthropic error envelope with
//     type, message, and request_id intact.
//   - /v1/chat/completions clients must receive an OpenAI-shape error
//     object carrying the same type and message, and no choices array.
//
// Requires the anthropic-shim profile.
func testAnthropicErrorPassthrough(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Anthropic] Testing error-envelope passthrough on both client protocols")
	}

	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	anthropicStatus, anthropicBody, err := sendErrorPassthroughRequest(ctx, localPort, "/v1/messages", map[string]interface{}{
		"model":      "MoM",
		"max_tokens": 32,
		"messages": []map[string]interface{}{
			{"role": "user", "content": errorSentinel},
		},
	})
	if err != nil {
		return fmt.Errorf("anthropic-protocol request failed: %w", err)
	}

	openAIStatus, openAIBody, err := sendErrorPassthroughRequest(ctx, localPort, "/v1/chat/completions", map[string]interface{}{
		"model": "MoM",
		"messages": []map[string]interface{}{
			{"role": "user", "content": errorSentinel},
		},
	})
	if err != nil {
		return fmt.Errorf("openai-protocol request failed: %w", err)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"anthropic_status": anthropicStatus,
			"anthropic_body":   string(anthropicBody),
			"openai_status":    openAIStatus,
			"openai_body":      string(openAIBody),
		})
	}

	if err := assertAnthropicErrorEnvelope(anthropicStatus, anthropicBody); err != nil {
		return fmt.Errorf("anthropic-protocol client: %w", err)
	}
	if err := assertOpenAIErrorBody(openAIStatus, openAIBody); err != nil {
		return fmt.Errorf("openai-protocol client: %w", err)
	}
	return nil
}

func assertAnthropicErrorEnvelope(status int, body []byte) error {
	if status != http.StatusTooManyRequests {
		return fmt.Errorf("expected upstream status 429 forwarded, got %d (body: %s)", status, body)
	}
	var envelope struct {
		Type  string `json:"type"`
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
		RequestID string `json:"request_id"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		return fmt.Errorf("unmarshal envelope: %w (body: %s)", err, body)
	}
	if envelope.Type != "error" {
		return fmt.Errorf("expected type=error envelope, got %q (body: %s)", envelope.Type, body)
	}
	if envelope.Error.Type != "rate_limit_error" {
		return fmt.Errorf("expected error.type=rate_limit_error, got %q", envelope.Error.Type)
	}
	if !strings.Contains(envelope.Error.Message, "synthetic rate limit") {
		return fmt.Errorf("upstream error message lost: %q", envelope.Error.Message)
	}
	if envelope.RequestID != "req_e2e_error_passthrough" {
		return fmt.Errorf("request_id lost: %q", envelope.RequestID)
	}
	return nil
}

func assertOpenAIErrorBody(status int, body []byte) error {
	if status != http.StatusTooManyRequests {
		return fmt.Errorf("expected upstream status 429 forwarded, got %d (body: %s)", status, body)
	}
	var parsed struct {
		Error *struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
		Choices   []interface{} `json:"choices"`
		RequestID *string       `json:"request_id"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return fmt.Errorf("unmarshal error body: %w (body: %s)", err, body)
	}
	if parsed.Error == nil {
		return fmt.Errorf("expected an error object, got none (body: %s)", body)
	}
	if parsed.Error.Type != "rate_limit_error" {
		return fmt.Errorf("expected error.type=rate_limit_error, got %q", parsed.Error.Type)
	}
	if !strings.Contains(parsed.Error.Message, "synthetic rate limit") {
		return fmt.Errorf("upstream error message lost: %q", parsed.Error.Message)
	}
	if len(parsed.Choices) != 0 {
		return fmt.Errorf("error body must not be shaped like a completion (body: %s)", body)
	}
	if parsed.RequestID != nil {
		return fmt.Errorf("OpenAI error bodies must stay spec-pure without request_id (body: %s)", body)
	}
	return nil
}

func sendErrorPassthroughRequest(ctx context.Context, localPort, path string, payload map[string]interface{}) (int, []byte, error) {
	jsonData, err := json.Marshal(payload)
	if err != nil {
		return 0, nil, fmt.Errorf("marshal: %w", err)
	}
	url := fmt.Sprintf("http://localhost:%s%s", localPort, path)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return 0, nil, fmt.Errorf("new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if path == "/v1/messages" {
		req.Header.Set("anthropic-version", "2023-06-01")
	}

	httpClient := &http.Client{Timeout: 120 * time.Second}
	resp, err := httpClient.Do(req)
	if err != nil {
		return 0, nil, fmt.Errorf("do: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return resp.StatusCode, nil, fmt.Errorf("read response body: %w", err)
	}
	return resp.StatusCode, raw, nil
}
