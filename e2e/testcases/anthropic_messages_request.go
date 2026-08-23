package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("anthropic-messages-request", pkgtestcases.TestCase{
		Description: "Send POST /v1/messages with an Anthropic Messages body and verify 200 OK and routing",
		Tags:        []string{"anthropic", "functional", "routing"},
		Fn:          testAnthropicMessagesRequest,
	})
}

// testAnthropicMessagesRequest exercises the inbound Anthropic Messages API
// path end-to-end: the Router accepts POST /v1/messages, decodes the request
// into the shared IR, runs routing, and encodes it for an OpenAI-shaped backend.
// We only assert on properties the router controls (status, decision header,
// non-empty body); the free-text response payload is mock-backend dependent.
func testAnthropicMessagesRequest(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Anthropic] Testing POST /v1/messages round-trip")
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
			{Role: "user", Content: "Explain the difference between TCP and UDP."},
		},
	}, localPort)
	if err != nil {
		return fmt.Errorf("anthropic messages request failed: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	decision := resp.Header.Get("x-vsr-selected-decision")
	selectedModel := resp.Header.Get("x-vsr-selected-model")

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"status_code":     resp.StatusCode,
			"response_length": len(body),
			"decision":        decision,
			"selected_model":  selectedModel,
		})
	}

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("expected status 200 for /v1/messages, got %d: %s",
			resp.StatusCode, truncateString(string(body), 200))
	}

	if len(body) == 0 {
		return fmt.Errorf("expected non-empty response body for /v1/messages")
	}

	// The public endpoint must always return well-formed JSON. The dedicated
	// response-shape contract verifies the complete Anthropic envelope.
	var parsed map[string]interface{}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return fmt.Errorf("response body is not valid JSON: %w (body=%s)", err,
			truncateString(string(body), 200))
	}

	// Routing must have happened: at least one of the decision/model headers
	// must be set, otherwise the request bypassed the routing pipeline.
	if decision == "" && selectedModel == "" {
		return fmt.Errorf("expected x-vsr-selected-decision or x-vsr-selected-model header to be set")
	}

	return nil
}

// anthropicMessage is the minimal Anthropic content shape for E2E requests
// (string content only; richer blocks are covered by the codec matrix).
type anthropicMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// anthropicMessagesRequestBody is the minimal POST /v1/messages payload
// used by the Anthropic e2e suite. max_tokens is required by the Anthropic
// Messages API; everything else mirrors the SDK shape.
type anthropicMessagesRequestBody struct {
	Model     string             `json:"model"`
	MaxTokens int                `json:"max_tokens"`
	Messages  []anthropicMessage `json:"messages"`
	System    string             `json:"system,omitempty"`
	Stream    bool               `json:"stream,omitempty"`
}

// sendAnthropicMessagesRequest POSTs an Anthropic Messages body to the
// router's /v1/messages surface and returns the raw response so callers
// can inspect both headers and body.
func sendAnthropicMessagesRequest(
	ctx context.Context,
	body anthropicMessagesRequestBody,
	localPort string,
) (*http.Response, error) {
	jsonData, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal: %w", err)
	}

	url := fmt.Sprintf("http://localhost:%s/v1/messages", localPort)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	// anthropic-version is required by the official Anthropic API. The router
	// does not currently gate on it but real SDK clients always send it.
	req.Header.Set("anthropic-version", "2023-06-01")

	httpClient := &http.Client{Timeout: 30 * time.Second}
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("do: %w", err)
	}
	return resp, nil
}
