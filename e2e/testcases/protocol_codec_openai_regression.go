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
	pkgtestcases.Register("protocol-codec-openai-regression", pkgtestcases.TestCase{
		Description: "Verify protocol translation does not regress OpenAI chat completions requests",
		Tags:        []string{"protocol", "openai", "regression"},
		Fn:          testProtocolCodecOpenAIRegression,
	})
}

// testProtocolCodecOpenAIRegression keeps the dominant OpenAI request path in
// the same black-box suite as the cross-protocol cases. It proves that codec
// selection and neutral translation do not bypass routing or alter a normal
// OpenAI round trip.
func testProtocolCodecOpenAIRegression(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Protocol codec] OpenAI regression: verifying the native path")
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
