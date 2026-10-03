package testcases

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const systemOneClassifierDecision = "systemone_classifier_risky"

func init() {
	pkgtestcases.Register("systemone-classifier-routing", pkgtestcases.TestCase{
		Description: "Verify a SystemOne Choice answer drives a generic classifier decision",
		Tags:        []string{"classifier", "systemone", "routing"},
		Fn:          testSystemOneClassifierRouting,
	})
}

// testSystemOneClassifierRouting exercises the router's own request at the HTTP
// boundary. The mock endpoint rejects any request that is not a valid SystemOne
// Choice over the declared labels, so a router that stops sending one fails here
// rather than being answered anyway.
func testSystemOneClassifierRouting(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	cases := []struct {
		name      string
		prompt    string
		wantMatch bool
	}{
		{
			name:      "risky choice matches",
			prompt:    "__SYSTEMONE_BACKEND_RISKY__ verify the typed decision path",
			wantMatch: true,
		},
		{
			name:      "safe choice does not match",
			prompt:    "verify the typed decision path with an ordinary request",
			wantMatch: false,
		},
	}

	decisions := make(map[string]string, len(cases))
	for _, testCase := range cases {
		response, err := sendLocalChatCompletion(ctx, localPort, "auto", testCase.prompt, 30*time.Second)
		if err != nil {
			return fmt.Errorf("%s: %w", testCase.name, err)
		}
		if response.StatusCode != http.StatusOK {
			return fmt.Errorf("%s: %s", testCase.name, formatUnexpectedChatCompletionStatus(response))
		}
		decision := response.Headers.Get("x-vsr-selected-decision")
		decisions[testCase.name] = decision
		matched := decision == systemOneClassifierDecision
		if matched != testCase.wantMatch {
			return fmt.Errorf(
				"%s: x-vsr-selected-decision=%q, expected systemone match=%t",
				testCase.name,
				decision,
				testCase.wantMatch,
			)
		}
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"routing_cases":  len(cases),
			"routing_passed": len(cases),
			"decisions":      decisions,
		})
	}
	return nil
}
