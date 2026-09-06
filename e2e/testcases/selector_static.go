package testcases

import (
	"context"
	"fmt"
	"net/http"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

const staticSelectorRequestCount = 3

func init() {
	pkgtestcases.Register("selector-static", pkgtestcases.TestCase{
		Description: "Verify the static selector deterministically chooses its first configured candidate",
		Tags:        []string{"selection", "static", "selector-conformance"},
		Fn:          testStaticSelector,
	})
}

func testStaticSelector(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	for requestNumber := 1; requestNumber <= staticSelectorRequestCount; requestNumber++ {
		response, err := sendLocalChatCompletion(
			ctx,
			localPort,
			"auto",
			fmt.Sprintf("selector-static-contract request %d", requestNumber),
			30*time.Second,
		)
		if err != nil {
			return fmt.Errorf("static selector request %d: %w", requestNumber, err)
		}
		if response.StatusCode != http.StatusOK {
			return fmt.Errorf("static selector request %d: %s", requestNumber, formatUnexpectedChatCompletionStatus(response))
		}
		if got := response.Headers.Get("x-vsr-selected-decision"); got != "selector_static" {
			return fmt.Errorf("static selector request %d: x-vsr-selected-decision=%q, want selector_static", requestNumber, got)
		}
		if got := response.Headers.Get("x-vsr-selected-algorithm"); got != "static" {
			return fmt.Errorf("static selector request %d: x-vsr-selected-algorithm=%q, want static", requestNumber, got)
		}
		if got := response.Headers.Get("x-vsr-selected-model"); got != "static-first" {
			return fmt.Errorf("static selector request %d: x-vsr-selected-model=%q, want static-first", requestNumber, got)
		}
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"algorithm":       "static",
			"expected_model":  "static-first",
			"successful_runs": staticSelectorRequestCount,
		})
	}
	return nil
}
