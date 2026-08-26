package extproc

import (
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/usageledger"
)

func TestSignedEmptyDispatchOutcomeSettlesBackendAndServedUsageKnownZero(t *testing.T) {
	fake := &fakeInferenceAccess{}
	router := &OpenAIRouter{Config: inferenceTestConfig(t), InferenceAccess: fake}
	ctx := admittedInferenceTestContext("internal-model")
	limit, err := quota.ParseQuotaInteger("100")
	if err != nil {
		t.Fatal(err)
	}
	ctx.InferenceAccess.admission.Rules = []quotaruntime.RuleBinding{{
		BindingID: "00000000-0000-4000-8000-000000000201",
		Rule: quota.RateLimitRule{
			ID: "00000000-0000-4000-8000-000000000202", Metric: quota.MetricServedTotalTokens,
			Algorithm: quota.AlgorithmSlidingLog, Accounting: quota.AccountingResponseActual,
			Enforcement: quota.EnforcementEnforce, WholeLimit: &limit, Window: time.Minute,
		},
	}}
	if err := router.beginPrimaryInferenceDispatch(ctx.TraceContext, ctx, "internal-model"); err != nil {
		t.Fatal(err)
	}
	markPrimaryCapabilityIssued(ctx)
	if err := applyDispatchOutcome(ctx, dispatchOutcomeForTest(ctx, 0)); err != nil {
		t.Fatalf("apply signed empty outcome: %v", err)
	}

	if err := router.completeAndSettlePrimaryInference(ctx, responseUsageMetrics{}, 503); err != nil {
		t.Fatalf("settle pre-dispatch terminal: %v", err)
	}
	if len(fake.settlements) != 1 {
		t.Fatalf("settlements = %d, want one", len(fake.settlements))
	}
	settlement := fake.settlements[0]
	if !settlement.Aggregate.Input.Complete || !settlement.Aggregate.Input.Value.IsZero() ||
		!settlement.Aggregate.Output.Complete || !settlement.Aggregate.Output.Value.IsZero() ||
		!settlement.Aggregate.Total.Complete || !settlement.Aggregate.Total.Value.IsZero() ||
		!settlement.Aggregate.Cost.Complete || !settlement.Aggregate.Cost.Value.IsZero() ||
		!settlement.Aggregate.ServedInput.Complete || !settlement.Aggregate.ServedInput.Value.IsZero() ||
		!settlement.Aggregate.ServedOutput.Complete || !settlement.Aggregate.ServedOutput.Value.IsZero() ||
		!settlement.Aggregate.ServedTotal.Complete || !settlement.Aggregate.ServedTotal.Value.IsZero() ||
		settlement.FenceID != "" {
		t.Fatalf("pre-dispatch aggregate = %+v fence=%q", settlement.Aggregate, settlement.FenceID)
	}
	if settlement.EventEvidenceState != usageledger.EvidenceKnown {
		t.Fatalf("event evidence state = %q, want %q", settlement.EventEvidenceState, usageledger.EvidenceKnown)
	}
	event, err := usageledger.DecodeTerminalEvent(settlement.Event)
	if err != nil {
		t.Fatal(err)
	}
	if event.EvidenceState != usageledger.EvidenceKnown || event.Fence != nil ||
		len(event.Dispatches) != 1 || event.Dispatches[0].UsageState != usageledger.UsageKnownZero ||
		!event.Served.InputKnown || event.Served.InputTokens != "0" ||
		!event.Served.OutputKnown || event.Served.OutputTokens != "0" {
		t.Fatalf("pre-dispatch terminal event = %+v", event)
	}
	ctx.DispatchState.mu.Lock()
	noDispatchProven := ctx.DispatchState.noDispatchProven
	ctx.DispatchState.mu.Unlock()
	if !noDispatchProven {
		t.Fatal("signed empty outcome did not retain no-dispatch proof")
	}
}
