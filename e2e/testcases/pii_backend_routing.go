package testcases

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("pii-backend-routing", pkgtestcases.TestCase{
		Description: "Verify a remote systemone deployment drives PII routing and fails closed when its answer is unknown",
		Tags:        []string{"systemone", "routing", "pii"},
		Fn:          testPIIBackendRouting,
	})
	pkgtestcases.Register("pii-tool-output-only", pkgtestcases.TestCase{
		Priority:    pkgtestcases.P0,
		Description: "Verify a pii rule with input.request: false reads the tool results and never the user turn",
		Tags:        []string{"systemone", "routing", "pii", "agent"},
		Fn:          testPIIToolOutputOnly,
	})
}

// testPIIBackendRouting drives the PII signal through a remote systemone
// deployment, provider-mocker's System One fixture.
//
// The profile configures no local PII model, so no local model can run and
// produce a span. Every assertion below therefore attributes the decision to
// the remote call rather than merely agreeing with it.
//
// Genuine detections select the higher-priority decision even though it uses
// on_unknown: no_match, and so does an input the deployment could not read in
// full, which a pii rule counts as a detection. A failed call matches the rule
// through its on_unknown: match as an unknown answer, which the no_match
// decision skips and the fail-closed decision takes.
func testPIIBackendRouting(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	cases := []struct {
		name             string
		prompt           string
		expectedDecision string
		why              string
	}{
		{
			name:             "remote span blocks",
			prompt:           "__PII_SPAN__ EMAIL_ADDRESS alice@corp.example please summarise this thread",
			expectedDecision: "block_detected_pii_remote",
			why:              "the deployment returned an EMAIL_ADDRESS span and the rule allows no PII type",
		},
		{
			name:             "unread input is a detection",
			prompt:           "__SYSTEMONE_UNREAD__ unseen text",
			expectedDecision: "block_detected_pii_remote",
			why:              "the deployment could not read the whole input, and a pii rule matches what it could not scan despite on_unknown: no_match",
		},
		{
			name:             "no span routes normally",
			prompt:           "please summarise this thread",
			expectedDecision: "default-route",
			why:              "the deployment returned no spans, so nothing was denied",
		},
		{
			name:             "failed call fails closed",
			prompt:           "__SYSTEMONE_UNAVAILABLE__ please summarise this thread",
			expectedDecision: "block_pii_remote",
			why:              "a 503 leaves no answer, and the rule's on_unknown: match reaches the fail-closed decision",
		},
	}

	observed := make(map[string]string, len(cases))
	for _, tc := range cases {
		resp, err := sendLocalChatCompletion(ctx, localPort, "vllm-sr/auto", tc.prompt, 30*time.Second)
		if err != nil {
			return fmt.Errorf("remote PII request (%s): %w", tc.name, err)
		}
		if resp.StatusCode != http.StatusOK {
			return fmt.Errorf("remote PII request (%s): %s", tc.name, formatUnexpectedChatCompletionStatus(resp))
		}

		decision := resp.Headers.Get("x-vsr-selected-decision")
		observed[tc.name] = decision

		if opts.Verbose {
			fmt.Printf("[Test] %s -> decision=%q (%s)\n", tc.name, decision, tc.why)
		}

		if decision != tc.expectedDecision {
			return fmt.Errorf(
				"%s selected decision %q, want %q - %s. With no local PII model configured, a wrong verdict here means the systemone deployment did not take effect",
				tc.name, decision, tc.expectedDecision, tc.why)
		}
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"remote_deployment": "provider-mocker /v1/systemone (System One fixture)",
			"local_pii_model":   "none, so no local model can produce a span",
			"span_returned":     observed["remote span blocks"],
			"unread_input":      observed["unread input is a detection"],
			"no_span":           observed["no span routes normally"],
			"failed_call":       observed["failed call fails closed"],
		})
	}
	return nil
}

// testPIIToolOutputOnly checks the profile's pii_tool_output rule, which reads
// the tool results with the user turn left out, the way an agent's privacy
// rule must: an agent's user turn carries its harness's own instructions.
// Personal data in the user turn alone reaches pii_deny_all and never this
// rule, and personal data in a tool result reaches this rule while the user
// turn stays clean.
func testPIIToolOutputOnly(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	toolTurn := func(user, toolOutput string) []map[string]interface{} {
		return []map[string]interface{}{
			{"role": "user", "content": user},
			{"role": "assistant", "content": nil, "tool_calls": []map[string]interface{}{{
				"id": "call_1", "type": "function",
				"function": map[string]interface{}{"name": "read_file", "arguments": `{"path":"owners.txt"}`},
			}}},
			{"role": "tool", "tool_call_id": "call_1", "content": toolOutput},
		}
	}
	cases := []struct {
		name     string
		messages any
		want     string
		why      string
	}{
		{
			name:     "personal data in the user turn alone",
			messages: []map[string]string{{"role": "user", "content": "__PII_SPAN__ EMAIL_ADDRESS alice@corp.example summarise this thread"}},
			want:     "block_detected_pii_remote",
			why:      "pii_deny_all reads the user turn; pii_tool_output has no tool result to read, so it is not asked",
		},
		{
			name:     "personal data in a tool result",
			messages: toolTurn("summarise the owners file", "__PII_SPAN__ EMAIL_ADDRESS bob@corp.example owns this file"),
			want:     "block_tool_output_pii",
			why:      "pii_tool_output reads the tool result, which carries the span",
		},
		{
			name:     "a clean tool result",
			messages: toolTurn("summarise the owners file", "nobody owns this file"),
			want:     "default-route",
			why:      "neither rule finds a span",
		},
		{
			name:     "personal data in the user turn beside a clean tool result",
			messages: toolTurn("__PII_SPAN__ EMAIL_ADDRESS alice@corp.example summarise the owners file", "nobody owns this file"),
			want:     "block_detected_pii_remote",
			why:      "pii_tool_output is asked about the clean tool result alone, so only pii_deny_all, which reads the user turn, matches",
		},
	}
	observed := make(map[string]string, len(cases))
	for _, tc := range cases {
		resp, err := sendLocalChatConversation(ctx, localPort, "vllm-sr/auto", tc.messages, 30*time.Second)
		if err != nil {
			return fmt.Errorf("tool output PII request (%s): %w", tc.name, err)
		}
		if resp.StatusCode != http.StatusOK {
			return fmt.Errorf("tool output PII request (%s): %s", tc.name, formatUnexpectedChatCompletionStatus(resp))
		}
		decision := resp.Headers.Get("x-vsr-selected-decision")
		observed[tc.name] = decision
		if decision != tc.want {
			return fmt.Errorf("%s selected decision %q, want %q - %s", tc.name, decision, tc.want, tc.why)
		}
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"decisions": observed})
	}
	return nil
}
