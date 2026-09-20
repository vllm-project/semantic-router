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
	pkgtestcases.Register("context-dedup-adjacent-turns", pkgtestcases.TestCase{
		Description: "An enabled context_dedup plugin removes the later copy of an adjacent repeated turn and keeps corrections and confirmations",
		Tags:        []string{"plugin", "context", "context-dedup"},
		Fn:          testContextDedupAdjacentTurns,
	})
	pkgtestcases.Register("context-dedup-disabled", pkgtestcases.TestCase{
		Description: "A configured but disabled context_dedup plugin forwards the complete conversation",
		Tags:        []string{"plugin", "context", "context-dedup"},
		Fn:          testContextDedupDisabled,
	})
}

const (
	contextDedupProbe         = "__CONTEXT_DEDUP_PROBE__"
	contextDedupDisabledProbe = "__CONTEXT_DEDUP_DISABLED_PROBE__"
)

// contextDedupScenario is one conversation with the exact upstream shape the
// router must produce for it. The mock backend echoes the conversation it
// actually received, so an HTTP 200 alone cannot satisfy any scenario.
type contextDedupScenario struct {
	name         string
	conversation []map[string]string
	wantRoles    []string
	wantUser     []string
}

func contextDedupScenarios(probe string) []contextDedupScenario {
	user := func(text string) map[string]string {
		return map[string]string{"role": "user", "content": probe + " " + text}
	}
	assistant := func(text string) map[string]string {
		return map[string]string{"role": "assistant", "content": text}
	}
	return []contextDedupScenario{
		{
			name: "adjacent_repeat",
			conversation: []map[string]string{
				user("first question about gardening"), assistant("first answer about gardening"),
				user("first question about gardening"), assistant("first answer about gardening"),
				user("what should I plant next"),
			},
			wantRoles: []string{"user", "assistant", "user"},
			wantUser:  []string{probe + " first question about gardening", probe + " what should I plant next"},
		},
		{
			name: "correction",
			conversation: []map[string]string{
				user("first question about gardening"), assistant("first answer about gardening"),
				user("first question about gardening"), assistant("a corrected answer about gardening"),
				user("what should I plant next"),
			},
			wantRoles: []string{"user", "assistant", "user", "assistant", "user"},
			wantUser: []string{
				probe + " first question about gardening",
				probe + " first question about gardening",
				probe + " what should I plant next",
			},
		},
		{
			name: "confirmation",
			conversation: []map[string]string{
				user("delete the old beds"), assistant("are you sure"),
				user("yes"), user("yes"),
				user("what should I plant next"),
			},
			wantRoles: []string{"user", "assistant", "user", "user", "user"},
			wantUser: []string{
				probe + " delete the old beds",
				probe + " yes",
				probe + " yes",
				probe + " what should I plant next",
			},
		},
	}
}

// testContextDedupAdjacentTurns asserts the enabled contract end to end: the
// later copy of an adjacent repeated turn is removed before dispatch, while a
// corrected reply and a repeated confirmation reach the provider untouched.
func testContextDedupAdjacentTurns(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing enabled context_dedup removes only adjacent repeated turns")
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	details := map[string]interface{}{}
	for _, scenario := range contextDedupScenarios(contextDedupProbe) {
		echo, err := runContextDedupScenario(ctx, localPort, scenario)
		if err != nil {
			return fmt.Errorf("scenario %s: %w", scenario.name, err)
		}
		details[scenario.name+"_messages_sent"] = len(scenario.conversation)
		details[scenario.name+"_messages_upstream"] = echo.TotalMessages
		if opts.Verbose {
			fmt.Printf("[Test] ✓ %s: sent %d messages, upstream received %d\n",
				scenario.name, len(scenario.conversation), echo.TotalMessages)
		}
	}
	if opts.SetDetails != nil {
		opts.SetDetails(details)
	}
	return nil
}

// testContextDedupDisabled asserts the disabled-by-default contract: the
// router accepts a decision carrying the plugin, and an adjacent repeated turn
// still reaches the provider in full.
func testContextDedupDisabled(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing disabled context_dedup preserves the conversation")
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	scenario := contextDedupScenarios(contextDedupDisabledProbe)[0]
	scenario.wantRoles = []string{"user", "assistant", "user", "assistant", "user"}
	scenario.wantUser = []string{
		contextDedupDisabledProbe + " first question about gardening",
		contextDedupDisabledProbe + " first question about gardening",
		contextDedupDisabledProbe + " what should I plant next",
	}
	echo, err := runContextDedupScenario(ctx, localPort, scenario)
	if err != nil {
		return err
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"messages_sent":     len(scenario.conversation),
			"messages_upstream": echo.TotalMessages,
		})
	}
	if opts.Verbose {
		fmt.Printf("[Test] ✓ upstream received all %d messages\n", echo.TotalMessages)
	}
	return nil
}

func runContextDedupScenario(
	ctx context.Context,
	localPort string,
	scenario contextDedupScenario,
) (*mockVLLMEcho, error) {
	response, err := sendLocalChatConversation(ctx, localPort, "auto", scenario.conversation, 60*time.Second)
	if err != nil {
		return nil, fmt.Errorf("failed to send conversation: %w", err)
	}
	if response.StatusCode != 200 {
		return nil, fmt.Errorf("%s", formatUnexpectedChatCompletionStatus(response))
	}
	echo, err := contextDedupUpstreamEcho(response.Body)
	if err != nil {
		return nil, err
	}
	if echo.TotalMessages != len(scenario.wantRoles) {
		return nil, fmt.Errorf("upstream received %d messages, want %d (roles %v)",
			echo.TotalMessages, len(scenario.wantRoles), echo.Roles)
	}
	if strings.Join(echo.Roles, ",") != strings.Join(scenario.wantRoles, ",") {
		return nil, fmt.Errorf("upstream roles %v, want %v", echo.Roles, scenario.wantRoles)
	}
	if strings.Join(echo.User, "\n") != strings.Join(scenario.wantUser, "\n") {
		return nil, fmt.Errorf("upstream user turns %q, want %q", echo.User, scenario.wantUser)
	}
	return echo, nil
}

// contextDedupUpstreamEcho reads the mock backend's JSON echo of the request
// it received from the assistant message of the chat completion.
func contextDedupUpstreamEcho(body []byte) (*mockVLLMEcho, error) {
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
