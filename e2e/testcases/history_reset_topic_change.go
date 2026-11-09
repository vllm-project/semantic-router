package testcases

import (
	"context"
	"fmt"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("history-reset-topic-change", pkgtestcases.TestCase{
		Description: "An enabled history_reset removes the prior turn on an explicit topic change and keeps it on a continuation or unknown result",
		Tags:        []string{"plugin", "context", "history-reset"},
		Fn:          testHistoryResetTopicChange,
	})
}

const historyResetEnabledProbe = "__HISTORY_RESET_ENABLED_PROBE__"

// testHistoryResetTopicChange drives the enabled path end to end: the
// topic_boundary rule evaluates the conversation, and the route's reset policy
// acts on that result before dispatch. The provider mocker echoes the
// conversation it actually received, so the assertions observe what reached
// the backend rather than the router's own report.
//
// The probe keyword appears only in the live turn. Repeating it in every turn
// would be shared vocabulary between the turns, which the evaluator rightly
// counts as evidence against a topic change.
func testHistoryResetTopicChange(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing enabled history_reset on topic change and continuation")
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	prior := []map[string]string{
		{"role": "user", "content": "Refactor the routing module so plugins load lazily"},
		{"role": "assistant", "content": "Done. The loader now defers plugin initialization."},
	}
	cases := []struct {
		name     string
		live     string
		upstream int
	}{
		{
			name:     "explicit change",
			live:     "Unrelated question: how do I renew a passport? " + historyResetEnabledProbe,
			upstream: 1,
		},
		{
			name:     "continuation",
			live:     "As you said earlier, the loader defers plugin initialization; can it also cache plugins? " + historyResetEnabledProbe,
			upstream: len(prior) + 1,
		},
		{
			// Inconclusive evidence never authorizes removal; this route fails
			// open, so the request keeps its history.
			name:     "unknown",
			live:     "Can you also make the loader thread safe? " + historyResetEnabledProbe,
			upstream: len(prior) + 1,
		},
	}

	details := map[string]interface{}{}
	for _, test := range cases {
		conversation := append(append([]map[string]string(nil), prior...),
			map[string]string{"role": "user", "content": test.live})
		response, sendErr := sendLocalChatConversation(ctx, localPort, "vllm-sr/auto", conversation, 60*time.Second)
		if sendErr != nil {
			return fmt.Errorf("%s: failed to send conversation: %w", test.name, sendErr)
		}
		if response.StatusCode != 200 {
			logUnexpectedChatCompletionStatus(opts.Verbose, response, "history-reset-topic-change")
			return fmt.Errorf("%s: %s", test.name, formatUnexpectedChatCompletionStatus(response))
		}
		echo, echoErr := historyResetUpstreamEcho(response.Body)
		if echoErr != nil {
			return fmt.Errorf("%s: %w", test.name, echoErr)
		}
		if echo.TotalMessages != test.upstream {
			return fmt.Errorf("%s: upstream received %d messages, want %d",
				test.name, echo.TotalMessages, test.upstream)
		}
		if len(echo.User) == 0 || !strings.Contains(echo.User[len(echo.User)-1], historyResetEnabledProbe) {
			return fmt.Errorf("%s: the live user turn was not forwarded, got %q", test.name, echo.User)
		}
		details[test.name] = echo.TotalMessages
		if opts.Verbose {
			fmt.Printf("[Test] ✓ %s: upstream received %d of %d messages\n",
				test.name, echo.TotalMessages, len(conversation))
		}
	}

	if opts.SetDetails != nil {
		opts.SetDetails(details)
	}
	return nil
}
