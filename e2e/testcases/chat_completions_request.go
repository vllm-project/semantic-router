package testcases

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("chat-completions-request", pkgtestcases.TestCase{
		Priority:    pkgtestcases.P0,
		Description: "A chat completions request is routed to a selected model and gets a completion back",
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
		Model: "vllm-sr/auto",
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
	// A 200 is not enough: the Router must have selected a model and the
	// client must get a completion back.
	var completion struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := resp.DecodeJSON(&completion); err != nil || len(completion.Choices) == 0 || completion.Choices[0].Message.Content == "" {
		return fmt.Errorf("expected a chat completion with content, got %s", truncateString(string(resp.Body), 300))
	}
	model := resp.Headers.Get("x-vsr-selected-model")
	if model == "" {
		return fmt.Errorf("expected x-vsr-selected-model on a routed response\n%s", formatResponseHeaders(resp.Headers))
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"status_code":    resp.StatusCode,
			"selected_model": model,
		})
	}
	return nil
}
