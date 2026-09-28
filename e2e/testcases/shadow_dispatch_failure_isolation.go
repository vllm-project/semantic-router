package testcases

import (
	"context"
	"fmt"
	"slices"
	"strconv"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("shadow-dispatch-fail-open-timeout", pkgtestcases.TestCase{
		Description: "A held shadow times out while the exact primary response completes within two seconds, then the same shadow route recovers",
		Tags:        []string{"router-replay", "shadow-dispatch", "failure-isolation"},
		Fn:          testShadowDispatchFailOpenTimeout,
	})
	pkgtestcases.Register("shadow-dispatch-fail-open-malformed-response", pkgtestcases.TestCase{
		Description: "An invalid shadow JSON response shape leaves the exact primary response intact and the same shadow route recovers",
		Tags:        []string{"router-replay", "shadow-dispatch", "failure-isolation"},
		Fn:          testShadowDispatchFailOpenMalformedResponse,
	})
	pkgtestcases.Register("shadow-dispatch-fail-open-queue-full", pkgtestcases.TestCase{
		Description: "Three primary requests finish within two seconds each while one shadow runs, one queues, and one drops without a Replay outcome",
		Tags:        []string{"router-replay", "shadow-dispatch", "failure-isolation"},
		Fn:          testShadowDispatchFailOpenQueueFull,
	})
}

func testShadowDispatchFailOpenTimeout(
	ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions,
) (result error) {
	run, err := openShadowDispatchRun(ctx, client, opts)
	if err != nil {
		return err
	}
	defer run.Close()
	defer run.releaseOnExit(ctx, "timeout", &result)
	if err = run.control(ctx, "timeout", "reset", "hold"); err != nil {
		return err
	}
	sessionID, err := run.primary(ctx, "vllm-sr/shadow-timeout", shadowPrimaryCeiling)
	if err != nil {
		return err
	}
	if _, err = run.waitProvider(ctx, "timeout", 1, 1); err != nil {
		return err
	}
	record, err := run.replay(ctx, sessionID, 1)
	if err != nil {
		return err
	}
	if err = requireShadowOutcome(record, "openai/shadow-timeout", "failed", "timeout"); err != nil {
		return err
	}
	latency, err := strconv.ParseInt(record.Outcomes[0].Metadata["latency_ms"], 10, 64)
	if err != nil || latency < shadowPrimaryCeiling.Milliseconds() {
		return fmt.Errorf("shadow timeout must outlast the primary ceiling, got latency_ms=%q",
			record.Outcomes[0].Metadata["latency_ms"])
	}
	if err = run.control(ctx, "timeout", "release", ""); err != nil {
		return err
	}
	if _, err = run.waitProvider(ctx, "timeout", 1, 0); err != nil {
		return err
	}
	recovery, err := run.primary(ctx, "vllm-sr/shadow-timeout", shadowPrimaryCeiling)
	if err != nil {
		return err
	}
	recovered, err := run.replay(ctx, recovery, 1)
	if err != nil {
		return err
	}
	if err = requireShadowOutcome(recovered, "openai/shadow-timeout", "completed", "completed"); err != nil {
		return err
	}
	_, err = run.waitProvider(ctx, "timeout", 2, 0)
	return err
}

func testShadowDispatchFailOpenMalformedResponse(
	ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions,
) (result error) {
	run, err := openShadowDispatchRun(ctx, client, opts)
	if err != nil {
		return err
	}
	defer run.Close()
	defer run.releaseOnExit(ctx, "malformed", &result)
	if err = run.control(ctx, "malformed", "reset", "malformed"); err != nil {
		return err
	}
	sessionID, err := run.primary(ctx, "vllm-sr/shadow-malformed", shadowPrimaryCeiling)
	if err != nil {
		return err
	}
	record, err := run.replay(ctx, sessionID, 1)
	if err != nil {
		return err
	}
	if err = requireShadowOutcome(record, "openai/shadow-malformed", "failed", "malformed_response"); err != nil {
		return err
	}
	if record.Outcomes[0].Metadata["status_code"] != "200" {
		return fmt.Errorf("malformed response must reach the JSON decoder after HTTP 200: %v", record.Outcomes[0].Metadata)
	}
	if _, err = run.waitProvider(ctx, "malformed", 1, 0); err != nil {
		return err
	}
	if err = run.control(ctx, "malformed", "release", ""); err != nil {
		return err
	}
	recovery, err := run.primary(ctx, "vllm-sr/shadow-malformed", shadowPrimaryCeiling)
	if err != nil {
		return err
	}
	recovered, err := run.replay(ctx, recovery, 1)
	if err != nil {
		return err
	}
	if err = requireShadowOutcome(recovered, "openai/shadow-malformed", "completed", "completed"); err != nil {
		return err
	}
	_, err = run.waitProvider(ctx, "malformed", 2, 0)
	return err
}

func testShadowDispatchFailOpenQueueFull(
	ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions,
) (result error) {
	run, err := openShadowDispatchRun(ctx, client, opts)
	if err != nil {
		return err
	}
	defer run.Close()
	defer run.releaseOnExit(ctx, "queue", &result)
	if err = run.control(ctx, "queue", "reset", "hold"); err != nil {
		return err
	}
	before, err := run.queueMetrics(ctx)
	if err != nil {
		return err
	}
	if before.Inflight != 0 || before.Queued != 0 {
		return fmt.Errorf("shadow queue must be idle before the scenario: %+v", before)
	}
	run.details["queue_before"] = before
	first, err := run.primary(ctx, "vllm-sr/shadow-queue", shadowPrimaryCeiling)
	if err != nil {
		return err
	}
	if _, err = run.waitProvider(ctx, "queue", 1, 1); err != nil {
		return err
	}
	if err = run.waitQueue(ctx, "running", shadowQueueMetrics{Inflight: 1, Drops: before.Drops}); err != nil {
		return err
	}
	queued, err := run.primary(ctx, "vllm-sr/shadow-queue", shadowPrimaryCeiling)
	if err != nil {
		return err
	}
	if err = run.waitQueue(ctx, "queued", shadowQueueMetrics{Inflight: 1, Queued: 1, Drops: before.Drops}); err != nil {
		return err
	}
	dropped, err := run.primary(ctx, "vllm-sr/shadow-queue", shadowPrimaryCeiling)
	if err != nil {
		return err
	}
	if err = run.waitQueue(ctx, "dropped", shadowQueueMetrics{Inflight: 1, Queued: 1, Drops: before.Drops + 1}); err != nil {
		return err
	}
	if _, err = run.waitProvider(ctx, "queue", 1, 1); err != nil {
		return err
	}
	if err = run.control(ctx, "queue", "release", ""); err != nil {
		return err
	}
	state, err := run.waitProvider(ctx, "queue", 2, 0)
	if err != nil {
		return err
	}
	if err = run.waitQueue(ctx, "drained", shadowQueueMetrics{Drops: before.Drops + 1}); err != nil {
		return err
	}
	var expectedRequestIDs []string
	for _, sessionID := range []string{first, queued} {
		record, replayErr := run.replay(ctx, sessionID, 1)
		if replayErr != nil {
			return replayErr
		}
		if err = requireShadowOutcome(record, "openai/shadow-queue", "completed", "completed"); err != nil {
			return err
		}
		expectedRequestIDs = append(expectedRequestIDs, record.Outcomes[0].Metadata["shadow_request_id"])
	}
	if expectedRequestIDs[0] == expectedRequestIDs[1] || !slices.Equal(state.RequestIDs, expectedRequestIDs) {
		return fmt.Errorf("provider shadow request IDs=%v, want distinct admitted requests %v", state.RequestIDs, expectedRequestIDs)
	}
	if _, err = run.replay(ctx, dropped, 0); err != nil {
		return fmt.Errorf("queue-full primary must retain its Replay row without a shadow outcome: %w", err)
	}
	recovery, err := run.primary(ctx, "vllm-sr/shadow-queue", shadowPrimaryCeiling)
	if err != nil {
		return err
	}
	recovered, err := run.replay(ctx, recovery, 1)
	if err != nil {
		return err
	}
	if err = requireShadowOutcome(recovered, "openai/shadow-queue", "completed", "completed"); err != nil {
		return err
	}
	if _, err = run.waitProvider(ctx, "queue", 3, 0); err != nil {
		return err
	}
	return run.waitQueue(ctx, "recovered", shadowQueueMetrics{Drops: before.Drops + 1})
}
