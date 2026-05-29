package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("anthropic-passthrough-openai-regression", pkgtestcases.TestCase{
		Description: "Verify the Anthropic passthrough carrier does not regress OpenAI chat completions requests",
		Tags:        []string{"anthropic", "passthrough", "regression"},
		Fn:          testAnthropicPassthroughOpenAIRegression,
	})
}

// testAnthropicPassthroughOpenAIRegression guards the OpenAI request path
// against regressions introduced by the Anthropic passthrough carrier.
//
// The passthrough machinery (cache_control replay, anthropic-version
// forwarding, image-block preservation, multi-block system reassembly)
// only fires when routing prepares an Anthropic-native upstream request.
// The e2e backend (llm-katan) is OpenAI-shaped, so the positive path is
// not exercisable here without scope-expanding the e2e infrastructure to
// add an Anthropic-native mock — explicitly out of scope for this PR.
//
// What is exercisable: this test sends a standard OpenAI request and
// asserts a clean 200 round-trip with routing headers. That guards
// against the case where the new carrier construction code accidentally
// breaks the dominant OpenAI traffic shape (regression). Per-field
// behavior of the carrier itself (cache_control, image blocks,
// anthropic-version) is covered by unit tests in
// src/semantic-router/pkg/anthropic.
func testAnthropicPassthroughOpenAIRegression(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Anthropic passthrough] OpenAI regression: verifying OpenAI path is unaffected")
	}

	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	chatClient := fixtures.NewChatCompletionsClient(session, 30*time.Second)
	resp, err := chatClient.Create(ctx, fixtures.ChatCompletionsRequest{
		Model: "MoM",
		Messages: []fixtures.ChatMessage{
			{Role: "user", Content: "Briefly describe how DNS resolution works."},
		},
	}, nil)
	if err != nil {
		return fmt.Errorf("openai chat completions request failed: %w", err)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"status_code":     resp.StatusCode,
			"response_length": len(resp.Body),
			"decision":        resp.Headers.Get("x-vsr-selected-decision"),
			"selected_model":  resp.Headers.Get("x-vsr-selected-model"),
		})
	}

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("expected status 200 for openai /v1/chat/completions, got %d: %s",
			resp.StatusCode, truncateString(string(resp.Body), 200))
	}
	if len(resp.Body) == 0 {
		return fmt.Errorf("expected non-empty response body")
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(resp.Body, &parsed); err != nil {
		return fmt.Errorf("response body is not valid JSON: %w", err)
	}

	// At least one routing header must be set; otherwise the request bypassed
	// the routing pipeline and the regression check is meaningless.
	if resp.Headers.Get("x-vsr-selected-decision") == "" &&
		resp.Headers.Get("x-vsr-selected-model") == "" {
		return fmt.Errorf("expected x-vsr-selected-decision or x-vsr-selected-model header to be set")
	}

	return nil
}
