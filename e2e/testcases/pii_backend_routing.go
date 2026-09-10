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
		Description: "Verify a remote token_spans.v1 backend drives PII routing and fails closed under on_error: block",
		Tags:        []string{"classifier", "http-classify", "routing", "pii"},
		Fn:          testPIIBackendRouting,
	})
}

// testPIIBackendRouting drives the PII signal through a remote token_spans.v1
// service.
//
// The profile configures no local PII model, so candle token classification
// cannot run and cannot produce a span. Every assertion below therefore
// attributes the decision to the remote call rather than merely agreeing with
// it.
//
// Three requests: one where the mock returns a span, one where it returns
// none, and one where it returns a 200 carrying an error member. The last is
// the on_error contract: a response the token_spans decoder rejects is not a
// clean "no PII" result, and with on_error: block it must close the request
// exactly like a real detection.
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
			expectedDecision: "block_pii_remote",
			why:              "the backend returned an EMAIL_ADDRESS span and the rule allows no PII type",
		},
		{
			name:             "no span routes normally",
			prompt:           "please summarise this thread",
			expectedDecision: "default-route",
			why:              "the backend returned an empty spans array, so nothing was denied",
		},
		{
			name:             "rejected response fails closed",
			prompt:           "__PII_ERROR__ please summarise this thread",
			expectedDecision: "block_pii_remote",
			why:              "a 200 carrying an error member is a backend failure, and on_error: block matches it as classification_error",
		},
	}

	observed := make(map[string]string, len(cases))
	for _, tc := range cases {
		resp, err := sendLocalChatCompletion(ctx, localPort, "auto", tc.prompt, 30*time.Second)
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
				"%s selected decision %q, want %q - %s. With no local PII model configured, a wrong verdict here means the token_spans.v1 backend did not take effect",
				tc.name, decision, tc.expectedDecision, tc.why)
		}
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"remote_classifier": "mock-pii-spans /classify (token_spans.v1)",
			"local_pii_model":   "none, so local token classification cannot produce a span",
			"span_returned":     observed["remote span blocks"],
			"no_span":           observed["no span routes normally"],
			"rejected_response": observed["rejected response fails closed"],
		})
	}
	return nil
}
