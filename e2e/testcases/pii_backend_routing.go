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
