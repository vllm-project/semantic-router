package testcases

import (
	"context"
	"fmt"
	"net/http"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("complexity-backend-routing", pkgtestcases.TestCase{
		Description: "Verify a remote score.v1 backend drives complexity routing through per-rule boundaries",
		Tags:        []string{"classifier", "http-classify", "routing", "complexity"},
		Fn:          testComplexityBackendRouting,
	})
}

// testComplexityBackendRouting drives the complexity signal through a remote
// score.v1 scorer.
//
// The profile declares no hard/easy candidates, so local prototype scoring has
// nothing to compare against and cannot produce a verdict at all. Every
// assertion below therefore attributes the routing decision to the remote
// call rather than merely agreeing with it.
//
// The two requests differ only in the score they pin, and the profile's two
// rules read that one score through different boundaries. A score of 0.90 is
// past needs_reasoning's hard boundary but inside extreme's medium band, while
// 0.99 clears both - so the same signal reaches two different decisions.
func testComplexityBackendRouting(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	cases := []struct {
		score            string
		expectedDecision string
		why              string
	}{
		{
			score:            "0.90",
			expectedDecision: "complexity_backend_hard",
			why:              "past needs_reasoning's hard_above (0.80) but inside extreme's medium band",
		},
		{
			score:            "0.99",
			expectedDecision: "complexity_backend_extreme",
			why:              "past both hard boundaries, so the stricter rule's higher-priority decision wins",
		},
	}

	observed := make(map[string]string, len(cases))
	for _, tc := range cases {
		prompt := fmt.Sprintf("__COMPLEXITY_SCORE__ %s route this request", tc.score)

		resp, err := sendLocalChatCompletion(ctx, localPort, "auto", prompt, 30*time.Second)
		if err != nil {
			return fmt.Errorf("remote complexity request (score %s): %w", tc.score, err)
		}
		if resp.StatusCode != http.StatusOK {
			return fmt.Errorf("remote complexity request (score %s): %s", tc.score, formatUnexpectedChatCompletionStatus(resp))
		}

		decision := resp.Headers.Get("x-vsr-selected-decision")
		observed[tc.score] = decision

		if opts.Verbose {
			fmt.Printf("[Test] score=%s -> decision=%q (%s)\n", tc.score, decision, tc.why)
		}

		if decision == "default-route" {
			return fmt.Errorf(
				"score %s fell through to default-route: the remote scorer produced no complexity verdict. "+
					"With no local candidates configured, that means the score.v1 backend did not take effect",
				tc.score)
		}
		if decision != tc.expectedDecision {
			return fmt.Errorf(
				"score %s selected decision %q, want %q - %s",
				tc.score, decision, tc.expectedDecision, tc.why)
		}
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"remote_scorer":    "mock-difficulty-scorer /classify (score.v1)",
			"local_candidates": "none, so the local prototype path cannot produce a verdict",
			"score_0.90":       observed["0.90"],
			"score_0.99":       observed["0.99"],
		})
	}
	return nil
}
