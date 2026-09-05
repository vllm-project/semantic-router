package extproc

import (
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

// memoryPersistenceReceipt holds only submission-time metadata and the
// concurrency-safe recorder; asynchronous reports never read RequestContext.
type memoryPersistenceReceipt struct {
	requestID   string
	decisionKey string
	replayID    string
	recorder    *routerreplay.Recorder
}

func (r *OpenAIRouter) snapshotMemoryPersistenceReceipt(ctx *RequestContext) memoryPersistenceReceipt {
	recorder := ctx.RouterReplayRecorder
	if recorder == nil {
		recorder = r.ReplayRecorder
	}
	return memoryPersistenceReceipt{
		requestID:   ctx.RequestID,
		decisionKey: requestDecisionStateKey(ctx),
		replayID:    ctx.RouterReplayID,
		recorder:    recorder,
	}
}

func (r *OpenAIRouter) recordMemoryPersistenceOutcome(
	ctx *RequestContext, status,
	reason string,
	failOpen bool,
	cause error,
) {
	r.snapshotMemoryPersistenceReceipt(ctx).record(status, reason, failOpen, cause)
}

func (receipt memoryPersistenceReceipt) record(status, reason string, failOpen bool, cause error) {
	if status != "scheduled" {
		metrics.RecordPluginExecution("memory_persistence", receipt.decisionKey, status, 0)
	}
	receipt.appendReplayOutcome(status, reason, failOpen)
	if status == "scheduled" {
		return
	}

	if !failOpen && cause == nil && status != "rejected" {
		return
	}

	fields := map[string]interface{}{
		"request_id": receipt.requestID,
		"status":     status,
		"reason":     reason,
		"fail_open":  failOpen,
	}
	if cause != nil {
		fields["error_class"] = fmt.Sprintf("%T", cause)
	}
	logging.ComponentWarnEvent("extproc", "memory_persistence_outcome", fields)
}

func (receipt memoryPersistenceReceipt) appendReplayOutcome(
	status string,
	reason string,
	failOpen bool,
) {
	if receipt.replayID == "" {
		return
	}
	recorder := receipt.recorder
	if recorder == nil {
		return
	}
	phase := "terminal"
	if status == "scheduled" {
		phase = "scheduled"
	}
	if err := recorder.AppendOutcome(receipt.replayID, routerreplay.Outcome{
		Timestamp: time.Now().UTC(),
		Source:    "router",
		Target:    "router",
		TargetRef: "memory_persistence",
		Verdict:   status,
		Reason:    reason,
		Metadata: map[string]string{
			"kind":      "memory_persistence_receipt",
			"phase":     phase,
			"fail_open": fmt.Sprintf("%t", failOpen),
		},
	}); err != nil {
		logging.ComponentWarnEvent("extproc", "memory_persistence_replay_outcome_failed", map[string]interface{}{
			"request_id": receipt.requestID,
			"replay_id":  receipt.replayID,
			"status":     status,
			"reason":     reason,
			"phase":      phase,
			"error":      err.Error(),
		})
	}
}
