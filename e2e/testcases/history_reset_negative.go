package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("history-reset-negative", pkgtestcases.TestCase{
		Description: "A configured but disabled history_reset plugin forwards the complete conversation",
		Tags:        []string{"plugin", "context", "history-reset"},
		Fn:          testHistoryResetNegative,
	})
}

// testHistoryResetNegative asserts the disabled-by-default contract of the
// history_reset plugin end to end: the router accepts a decision carrying the
// plugin, and the upstream request still contains every prior turn. The mock
// backend echoes the conversation it actually received, so an HTTP 200 alone
// cannot satisfy this test.
func testHistoryResetNegative(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing disabled history_reset preserves the conversation")
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	conversation := []map[string]string{
		{"role": "user", "content": "__HISTORY_RESET_PROBE__ first question about gardening"},
		{"role": "assistant", "content": "first answer about gardening"},
		{"role": "user", "content": "__HISTORY_RESET_PROBE__ second question about gardening"},
		{"role": "assistant", "content": "second answer about gardening"},
		{"role": "user", "content": "__HISTORY_RESET_PROBE__ now tell me about tax law instead"},
	}

	response, err := sendLocalChatConversation(ctx, localPort, "auto", conversation, 60*time.Second)
	if err != nil {
		return fmt.Errorf("failed to send conversation: %w", err)
	}
	if response.StatusCode != 200 {
		logUnexpectedChatCompletionStatus(opts.Verbose, response, "history-reset-negative")
		return fmt.Errorf("%s", formatUnexpectedChatCompletionStatus(response))
	}

	echo, err := historyResetUpstreamEcho(response.Body)
	if err != nil {
		return err
	}
	if echo.TotalMessages != len(conversation) {
		return fmt.Errorf(
			"upstream received %d messages, want %d: a disabled history_reset must not remove history",
			echo.TotalMessages,
			len(conversation),
		)
	}
	if len(echo.User) != 3 {
		return fmt.Errorf("upstream received %d user turns, want 3", len(echo.User))
	}
	if !strings.Contains(echo.User[0], "first question about gardening") {
		return fmt.Errorf("the oldest user turn was not forwarded, got %q", echo.User[0])
	}
	if !strings.Contains(echo.User[len(echo.User)-1], "tax law") {
		return fmt.Errorf("the live user turn was not forwarded, got %q", echo.User[len(echo.User)-1])
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"messages_sent":     len(conversation),
			"messages_upstream": echo.TotalMessages,
			"user_turns":        len(echo.User),
		})
	}
	if opts.Verbose {
		fmt.Printf("[Test] ✓ upstream received all %d messages\n", echo.TotalMessages)
	}
	return nil
}

// historyResetUpstreamEcho reads the mock backend's JSON echo of the request
// it received from the assistant message of the chat completion.
func historyResetUpstreamEcho(body []byte) (*mockVLLMEcho, error) {
	var completion struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(body, &completion); err != nil {
		return nil, fmt.Errorf("failed to parse chat completion: %w", err)
	}
	if len(completion.Choices) == 0 {
		return nil, fmt.Errorf("chat completion carried no choices: %s", truncateString(string(body), 200))
	}

	echo := &mockVLLMEcho{}
	if err := json.Unmarshal([]byte(completion.Choices[0].Message.Content), echo); err != nil {
		return nil, fmt.Errorf(
			"assistant content is not a mock-vllm echo: %w (content=%q)",
			err,
			truncateString(completion.Choices[0].Message.Content, 200),
		)
	}
	if echo.Mock != "mock-vllm" {
		return nil, fmt.Errorf("unexpected backend marker %q", echo.Mock)
	}
	return echo, nil
}
