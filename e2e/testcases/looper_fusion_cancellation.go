package testcases

import (
	"context"
	"errors"
	"fmt"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const (
	looperFusionCancelProbeKeyword  = "__LOOPER_FUSION_CANCEL_PROBE__"
	looperFusionJudgeModel          = "fusion-judge"
	looperFusionFallbackModel       = "fusion-fallback-target"
	looperFusionSlowPanelModel      = "fusion-panel-slow"
	looperFusionBrokenFallbackModel = "fusion-fallback-broken"
	looperFusionTinyWindowModel     = "fusion-fallback-tiny-window"
)

func init() {
	pkgtestcases.Register("looper-fusion-quorum-caller-cancellation", pkgtestcases.TestCase{
		Description: "Abandon a Fusion panel on caller cancellation without spending fallback budget",
		Tags:        []string{"kubernetes", "routing", "looper", "fusion"},
		Fn:          testLooperFusionQuorumCallerCancellation,
	})
}

// testLooperFusionQuorumCallerCancellation covers caller cancellation, which is
// a different contract from the internal round timeout.
//
// On a round timeout the client is still waiting, so the fallback runs and
// answers. When the caller disconnects there is nobody left to answer, and the
// policy deliberately skips the fallback rather than spend more budget. The
// decision behind this case has no round_timeout_seconds, so cancellation is the
// only thing that can end its panel.
func testLooperFusionQuorumCallerCancellation(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	counters, err := openFusionBackendCounters(ctx, client, opts)
	if err != nil {
		return err
	}
	defer counters.close()
	if err := counters.reset(ctx); err != nil {
		return err
	}

	requestCtx, cancelRequest := context.WithCancel(ctx)
	defer cancelRequest()

	requestErr := make(chan error, 1)
	go func() {
		_, sendErr := sendLocalChatCompletion(
			requestCtx, localPort, "MoM", looperFusionCancelProbeKeyword, 60*time.Second)
		requestErr <- sendErr
	}()

	// Cancel only once the panel is demonstrably in flight. A fixed sleep would
	// be flaky in both directions: too short and the test would cancel before
	// routing even began, proving nothing; too long and the panel could finish
	// first.
	if err := counters.waitForDispatch(ctx, looperFusionSlowPanelModel, 30*time.Second); err != nil {
		return fmt.Errorf("looper-fusion-quorum-caller-cancellation: %w", err)
	}
	cancelRequest()

	select {
	case sendErr := <-requestErr:
		if sendErr == nil {
			return fmt.Errorf("looper-fusion-quorum-caller-cancellation: cancelled request unexpectedly succeeded")
		}
		// net/http wraps the cause, so match on the sentinel rather than text.
		if !errors.Is(sendErr, context.Canceled) {
			return fmt.Errorf("looper-fusion-quorum-caller-cancellation: want context.Canceled, got %w", sendErr)
		}
	case <-time.After(30 * time.Second):
		return fmt.Errorf("looper-fusion-quorum-caller-cancellation: cancelled request never returned")
	}

	// The slow panel model sleeps well past this point. Waiting past the moment
	// the panel would otherwise have finished is what makes a zero count mean
	// "never dispatched" rather than "not dispatched yet".
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(10 * time.Second):
	}

	return counters.requireCounts(ctx, "looper-fusion-quorum-caller-cancellation", map[string]int{
		looperFusionSlowPanelModel: 1,
		looperFusionFallbackModel:  0,
		looperFusionJudgeModel:     0,
	})
}
