package testcases

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("chat-completions-request", pkgtestcases.TestCase{
		Description: "Send a chat completions request and verify 200 OK response",
		Tags:        []string{"llm", "functional"},
		Fn:          testChatCompletionsRequest,
	})
}

func testChatCompletionsRequest(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	return runChatCompletionsRequest(ctx, client, opts, nil)
}

func runChatCompletionsRequest(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
	headers map[string]string,
) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing chat completions endpoint")
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
			{Role: "user", Content: "Hello, how are you?"},
		},
	}, headers)
	if err != nil {
		return err
	}

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("expected status 200, got %d: %s", resp.StatusCode, string(resp.Body))
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"status_code":     resp.StatusCode,
			"response_length": len(resp.Body),
		})
	}
	return nil
}
