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
	pkgtestcases.Register("tool-selection-anthropic", pkgtestcases.TestCase{
		Description: "Native Anthropic /v1/messages tool selection applies the same filter decision as OpenAI chat completions",
		Tags:        []string{"kubernetes", "plugin", "tool-selection", "anthropic"},
		Fn:          testToolSelectionAnthropicE2E,
	})
}

type anthropicE2ETool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema"`
}

func testToolSelectionAnthropicE2E(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] tool_selection native Anthropic /v1/messages filter contract")
	}

	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	minObjectParams := json.RawMessage(`{"type":"object","properties":{}}`)
	payload := anthropicMessagesRequestBody{
		Model:     "MoM",
		MaxTokens: 64,
		Messages: []anthropicMessage{
			{Role: "user", Content: "__TOOL_SELECTION_FILTER__ Will it rain in Seattle this weekend?"},
		},
		Tools: []anthropicE2ETool{
			{Name: "get_weather", Description: "Get current weather information for a location", InputSchema: minObjectParams},
			{Name: "contract_noise_alpha", Description: "Unrelated tool for cataloguing antique spoons", InputSchema: minObjectParams},
			{Name: "contract_noise_beta", Description: "Metadata about underground subway tile patterns", InputSchema: minObjectParams},
		},
		ToolChoice: json.RawMessage(`{"type":"auto"}`),
	}

	resp, err := sendAnthropicToolSelectionRequest(ctx, localPort, payload)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	decision := resp.Header.Get("x-vsr-selected-decision")
	strategy := resp.Header.Get("x-vsr-tools-strategy")
	confidence := resp.Header.Get("x-vsr-tools-confidence")
	latencyMs := resp.Header.Get("x-vsr-tools-latency-ms")
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"status_code":      resp.StatusCode,
			"decision":         decision,
			"tools_strategy":   strategy,
			"tools_confidence": confidence,
			"tools_latency_ms": latencyMs,
		})
	}

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("tool-selection-anthropic: expected status 200, got %d: %s",
			resp.StatusCode, truncateString(string(body), 200))
	}
	if decision != "tool_selection_filter_decision" {
		return fmt.Errorf("tool-selection-anthropic: decision want %q got %q",
			"tool_selection_filter_decision", decision)
	}
	if strategy != "" && strategy != "filter" {
		return fmt.Errorf("tool-selection-anthropic: x-vsr-tools-strategy want %q got %q", "filter", strategy)
	}
	return nil
}

func sendAnthropicToolSelectionRequest(
	ctx context.Context,
	localPort string,
	body anthropicMessagesRequestBody,
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
	req.Header.Set("anthropic-version", "2023-06-01")
	req.Header.Set("x-vsr-debug", "true")

	httpClient := &http.Client{Timeout: 45 * time.Second}
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("do: %w", err)
	}
	return resp, nil
}
