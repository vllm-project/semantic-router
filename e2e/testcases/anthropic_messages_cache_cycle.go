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
	pkgtestcases.Register("anthropic-messages-cache-cycle", pkgtestcases.TestCase{
		Description: "Verify cache_creation_input_tokens on first request and cache_read_input_tokens on repeat (anthropic-shim profile)",
		Tags:        []string{"anthropic", "cache", "functional"},
		Fn:          testAnthropicMessagesCacheCycle,
	})
}

// anthropicCacheBlock is the content block shape used in the system array for
// cache-cycle testing. It carries cache_control so the shim treats the
// system prefix as a cacheable boundary.
type anthropicCacheBlock struct {
	Type         string                 `json:"type"`
	Text         string                 `json:"text"`
	CacheControl map[string]interface{} `json:"cache_control,omitempty"`
}

// anthropicCacheRequestBody is the POST /v1/messages payload for cache-cycle
// tests. system is an array (scenario 8) so the shim can detect the
// cache_control marker and synthesise usage counters accordingly.
type anthropicCacheRequestBody struct {
	Model     string                `json:"model"`
	MaxTokens int                   `json:"max_tokens"`
	System    []anthropicCacheBlock `json:"system"`
	Messages  []anthropicMessage    `json:"messages"`
}

// anthropicCacheUsage holds the subset of the usage object the cache-cycle
// test cares about. Other fields are intentionally ignored.
type anthropicCacheUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
}

// anthropicCacheResponse mirrors the Anthropic Messages wire shape for the
// fields needed by cache-cycle assertions.
type anthropicCacheResponse struct {
	ID         string              `json:"id"`
	Type       string              `json:"type"`
	StopReason string              `json:"stop_reason"`
	Usage      anthropicCacheUsage `json:"usage"`
}

// testAnthropicMessagesCacheCycle exercises scenarios 1, 2, and 8:
//
//   - Scenario 8: system is an array with cache_control on at least one block.
//   - Scenario 1: first request with a new cache prefix must set
//     usage.cache_creation_input_tokens > 0 and
//     usage.cache_read_input_tokens == 0.
//   - Scenario 2: second request with the same prefix must set
//     usage.cache_read_input_tokens > 0.
//
// Both requests share the same x-vsr-test-session-id header so the shim's
// per-session tracker sees the prefix repeat. The router's outbound emitter
// must propagate the synthesised usage fields onto the Anthropic-shaped
// response unchanged; any loss or zero-overwrite fails the test.
//
// Requires the anthropic-shim profile.
func testAnthropicMessagesCacheCycle(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Anthropic] Testing cache-cycle assertions on /v1/messages")
	}

	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	sessionID := fmt.Sprintf("cache-cycle-%d", time.Now().UnixNano())

	body := anthropicCacheRequestBody{
		Model:     "MoM",
		MaxTokens: 32,
		System: []anthropicCacheBlock{
			{
				Type: "text",
				Text: "You are a helpful assistant. Answer concisely.",
				CacheControl: map[string]interface{}{
					"type": "ephemeral",
				},
			},
		},
		Messages: []anthropicMessage{
			{Role: "user", Content: "Say one word."},
		},
	}

	resp1, body1, err := sendCacheRequest(ctx, body, sessionID, localPort)
	if err != nil {
		return fmt.Errorf("request 1 failed: %w", err)
	}

	if resp1.StatusCode != http.StatusOK {
		return fmt.Errorf("expected status 200 for request 1, got %d", resp1.StatusCode)
	}
	if body1.Usage.CacheCreationInputTokens <= 0 {
		return fmt.Errorf(
			"scenario 1: expected cache_creation_input_tokens > 0 on first request, got %d",
			body1.Usage.CacheCreationInputTokens,
		)
	}
	if body1.Usage.CacheReadInputTokens != 0 {
		return fmt.Errorf(
			"scenario 1: expected cache_read_input_tokens == 0 on first request, got %d",
			body1.Usage.CacheReadInputTokens,
		)
	}

	resp2, body2, err := sendCacheRequest(ctx, body, sessionID, localPort)
	if err != nil {
		return fmt.Errorf("request 2 failed: %w", err)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"status_code_req1":           resp1.StatusCode,
			"cache_creation_tokens_req1": body1.Usage.CacheCreationInputTokens,
			"cache_read_tokens_req1":     body1.Usage.CacheReadInputTokens,
			"status_code_req2":           resp2.StatusCode,
			"cache_creation_tokens_req2": body2.Usage.CacheCreationInputTokens,
			"cache_read_tokens_req2":     body2.Usage.CacheReadInputTokens,
		})
	}

	if resp2.StatusCode != http.StatusOK {
		return fmt.Errorf("expected status 200 for request 2, got %d", resp2.StatusCode)
	}
	if body2.Usage.CacheReadInputTokens <= 0 {
		return fmt.Errorf(
			"scenario 2: expected cache_read_input_tokens > 0 on repeat request, got %d",
			body2.Usage.CacheReadInputTokens,
		)
	}
	if body2.Usage.CacheCreationInputTokens != 0 {
		return fmt.Errorf(
			"scenario 2: expected cache_creation_input_tokens == 0 on repeat request, got %d",
			body2.Usage.CacheCreationInputTokens,
		)
	}

	return nil
}

// sendCacheRequest posts an anthropicCacheRequestBody to /v1/messages with
// the given session header and returns the parsed response.
func sendCacheRequest(
	ctx context.Context,
	body anthropicCacheRequestBody,
	sessionID string,
	localPort string,
) (*http.Response, anthropicCacheResponse, error) {
	jsonData, err := json.Marshal(body)
	if err != nil {
		return nil, anthropicCacheResponse{}, fmt.Errorf("marshal: %w", err)
	}

	url := fmt.Sprintf("http://localhost:%s/v1/messages", localPort)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, anthropicCacheResponse{}, fmt.Errorf("new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("anthropic-version", "2023-06-01")
	req.Header.Set("x-vsr-test-session-id", sessionID)

	httpClient := &http.Client{Timeout: 120 * time.Second}
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, anthropicCacheResponse{}, fmt.Errorf("do: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return resp, anthropicCacheResponse{}, fmt.Errorf("read response body: %w", err)
	}
	var parsed anthropicCacheResponse
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return resp, anthropicCacheResponse{}, fmt.Errorf(
			"unmarshal response: %w (body=%s)", err, truncateString(string(raw), 200),
		)
	}
	return resp, parsed, nil
}
