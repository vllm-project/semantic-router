package extproc

import (
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
)

func TestInferenceSettlementStopsAdmissionHeartbeat(t *testing.T) {
	called := make(chan struct{}, 8)
	fake := &fakeInferenceAccess{heartbeat: func(_ accessruntime.Admission) (quotaruntime.AdmissionHeartbeatResult, error) {
		called <- struct{}{}
		return quotaruntime.AdmissionHeartbeatResult{Deadline: time.Now().UTC().Add(time.Minute)}, nil
	}}
	router := &OpenAIRouter{Config: inferenceTestConfig(t), InferenceAccess: fake}
	ctx := admittedInferenceTestContext("internal-model")
	ctx.InferenceAccess.admission.Result.PlanDigest = "heartbeat-plan"
	router.startInferenceAdmissionHeartbeatEvery(ctx, *ctx.InferenceAccess.admission, 2*time.Millisecond)
	select {
	case <-called:
	case <-time.After(time.Second):
		t.Fatal("admission heartbeat did not run")
	}
	if err := router.settleNoBackendInference(ctx, 200, "cache_short_circuit"); err != nil {
		t.Fatalf("settleNoBackendInference() error = %v", err)
	}
	ctx.InferenceAccess.mu.Lock()
	done := ctx.InferenceAccess.heartbeatDone
	finalized := ctx.InferenceAccess.finalized
	ctx.InferenceAccess.mu.Unlock()
	if done != nil || !finalized {
		t.Fatalf("settlement did not close heartbeat lifecycle: done=%v finalized=%v", done, finalized)
	}
	count := fake.heartbeatCount()
	time.Sleep(6 * time.Millisecond)
	if after := fake.heartbeatCount(); after != count {
		t.Fatalf("heartbeat continued after settlement: before=%d after=%d", count, after)
	}
}

func TestInferenceAdmissionLeaseCoversEveryPinnedModelTimeout(t *testing.T) {
	router := &OpenAIRouter{Config: inferenceTestConfig(t)}
	params := router.Config.ModelConfig["internal-model"]
	params.Execution.RequestTimeout = "20m"
	params.Execution.StreamTimeout = "2h"
	router.Config.ModelConfig["internal-model"] = params

	got := router.inferenceAdmissionLease([]string{"internal-model"})
	want := 2*time.Hour + inferenceAdmissionLeaseHeadroom
	if got != want {
		t.Fatalf("inferenceAdmissionLease() = %s, want %s", got, want)
	}
	if minimum := router.inferenceAdmissionLease([]string{"missing-model"}); minimum != inferenceAdmissionMinimumLease {
		t.Fatalf("minimum inferenceAdmissionLease() = %s, want %s", minimum, inferenceAdmissionMinimumLease)
	}
}
