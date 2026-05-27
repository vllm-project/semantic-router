package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("anthropic-messages-response-shape", pkgtestcases.TestCase{
		Description: "Verify non-streaming /v1/messages response is in Anthropic Messages API shape",
		Tags:        []string{"anthropic", "response-shape"},
		Fn:          testAnthropicMessagesResponseShape,
	})
}

// anthropicMessageResponse mirrors the Anthropic Messages API wire shape
// well enough for E2E structural assertions. Free-text content and exact
// usage values are mock-backend dependent and intentionally untyped here.
type anthropicMessageResponse struct {
	ID         string            `json:"id"`
	Type       string            `json:"type"`
	Role       string            `json:"role"`
	Model      string            `json:"model"`
	Content    []json.RawMessage `json:"content"`
	StopReason string            `json:"stop_reason"`
	Usage      json.RawMessage   `json:"usage"`
}

// validAnthropicStopReasons enumerates the stop_reason values defined by
// the Anthropic Messages API. The outbound emitter PR maps OpenAI finish
// reasons into this set; any value outside it indicates the emitter
// either bypassed the mapping or invented a new token.
var validAnthropicStopReasons = map[string]struct{}{
	"end_turn":      {},
	"max_tokens":    {},
	"stop_sequence": {},
	"tool_use":      {},
	"pause_turn":    {},
	"refusal":       {},
}

// testAnthropicMessagesResponseShape asserts the outbound emitter rewrites
// the OpenAI ChatCompletion body the router normalizes to back into the
// Anthropic Messages wire shape, so an Anthropic-SDK client (Claude Code,
// anthropic-sdk-go) can deserialize it without custom adapters.
func testAnthropicMessagesResponseShape(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Anthropic] Verifying response is in Anthropic Messages shape")
	}

	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	resp, err := sendAnthropicMessagesRequest(ctx, anthropicMessagesRequestBody{
		Model:     "MoM",
		MaxTokens: 64,
		Messages: []anthropicMessage{
			{Role: "user", Content: "Say hi in one word."},
		},
	}, localPort)
	if err != nil {
		return fmt.Errorf("anthropic messages request failed: %w", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("expected status 200, got %d: %s", resp.StatusCode, truncateString(string(body), 200))
	}

	var parsed anthropicMessageResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return fmt.Errorf("response body is not Anthropic-shaped JSON: %w (body=%s)",
			err, truncateString(string(body), 200))
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"status_code":    resp.StatusCode,
			"id":             parsed.ID,
			"type":           parsed.Type,
			"role":           parsed.Role,
			"stop_reason":    parsed.StopReason,
			"content_blocks": len(parsed.Content),
			"has_usage":      len(parsed.Usage) > 0,
		})
	}

	return assertAnthropicMessageShape(parsed)
}

// assertAnthropicMessageShape enforces the per-field invariants the
// outbound emitter is responsible for; kept as a helper so the test
// function stays under the project's cyclomatic-complexity ceiling.
func assertAnthropicMessageShape(parsed anthropicMessageResponse) error {
	if parsed.ID == "" {
		return fmt.Errorf("expected non-empty id in Anthropic response")
	}
	if parsed.Type != "message" {
		return fmt.Errorf("expected type=\"message\", got %q", parsed.Type)
	}
	if parsed.Role != "assistant" {
		return fmt.Errorf("expected role=\"assistant\", got %q", parsed.Role)
	}
	// content must always be a (possibly empty) array, never null.
	if parsed.Content == nil {
		return fmt.Errorf("expected content array (may be empty), got null")
	}
	if _, ok := validAnthropicStopReasons[parsed.StopReason]; !ok {
		return fmt.Errorf("stop_reason %q is not a valid Anthropic value", parsed.StopReason)
	}
	// usage block must be present (input_tokens/output_tokens are upstream
	// of vsr but the emitter is responsible for surfacing it).
	if len(parsed.Usage) == 0 {
		return fmt.Errorf("expected usage object in Anthropic response, got empty")
	}
	return nil
}
