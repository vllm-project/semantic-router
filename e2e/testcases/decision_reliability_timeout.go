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
	pkgtestcases.Register("decision-reliability-timeout", pkgtestcases.TestCase{
		Description: "A decision's total_timeout reaches Envoy's router per request: Envoy answers 504 at that " +
			"timeout, while a decision without one waits for the slow backend",
		Tags: []string{"reliability", "functional"},
		Fn:   testDecisionReliabilityTimeout,
	})
}

// Bounds for a backend that answers after six seconds, behind a decision
// whose reliability block sets total_timeout: 2s.
const (
	decisionTimeout     = 2 * time.Second
	decisionTimeoutSlop = 3 * time.Second
	slowBackendDelay    = 6 * time.Second
)

func testDecisionReliabilityTimeout(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	chat := fixtures.NewChatCompletionsClient(session, 30*time.Second)
	ask := func(prompt string) (int, time.Duration, []byte, error) {
		started := time.Now()
		resp, sendErr := chat.Create(ctx, fixtures.ChatCompletionsRequest{
			Model:    "MoM",
			Messages: []fixtures.ChatMessage{{Role: "user", Content: prompt}},
		}, nil)
		if sendErr != nil {
			return 0, 0, nil, sendErr
		}
		return resp.StatusCode, time.Since(started), resp.Body, nil
	}

	boundedStatus, boundedElapsed, body, err := ask("reliability-probe-bounded: answer when you can")
	if err != nil {
		return err
	}
	if boundedStatus != http.StatusGatewayTimeout ||
		boundedElapsed < decisionTimeout || boundedElapsed >= decisionTimeout+decisionTimeoutSlop {
		return fmt.Errorf("bounded decision: status %d after %v (%s), want Envoy's 504 at the decision's %v",
			boundedStatus, boundedElapsed, body, decisionTimeout)
	}

	controlStatus, controlElapsed, body, err := ask("reliability-probe-control: answer when you can")
	if err != nil {
		return err
	}
	if controlStatus != http.StatusOK || controlElapsed < slowBackendDelay-time.Second {
		return fmt.Errorf("control decision: status %d after %v (%s), want the slow backend's 200",
			controlStatus, controlElapsed, body)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"bounded_status":     boundedStatus,
			"bounded_elapsed_ms": boundedElapsed.Milliseconds(),
			"control_status":     controlStatus,
			"control_elapsed_ms": controlElapsed.Milliseconds(),
		})
	}
	return nil
}
