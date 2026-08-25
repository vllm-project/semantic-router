package extproc

import (
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/usageledger"
)

func TestCacheHitSettlementKeepsBackendZeroAndChargesAuthoritativeServedTokens(t *testing.T) {
	fake := &fakeInferenceAccess{}
	router := &OpenAIRouter{Config: inferenceTestConfig(t), InferenceAccess: fake}
	ctx := admittedInferenceTestContext("internal-model")
	ctx.VSRCacheHit = true
	recordCacheHitSettlementUsage(ctx, responseUsageMetrics{
		promptTokens: 7, promptTokensReported: true,
		completionTokens: 3, completionTokensReported: true,
		totalTokens: 10, totalTokensReported: true,
	})
	if err := router.settleNoBackendInference(ctx, 200, "cache_short_circuit"); err != nil {
		t.Fatalf("settleNoBackendInference() error = %v", err)
	}
	if len(fake.settlements) != 1 {
		t.Fatalf("settlements = %d, want one", len(fake.settlements))
	}
	settlement := fake.settlements[0]
	if !settlement.Aggregate.Total.Complete || !settlement.Aggregate.Total.Value.IsZero() ||
		!settlement.Aggregate.Cost.Complete || !settlement.Aggregate.Cost.Value.IsZero() ||
		!settlement.Aggregate.ServedInput.Complete || settlement.Aggregate.ServedInput.Value.String() != "7" ||
		!settlement.Aggregate.ServedOutput.Complete || settlement.Aggregate.ServedOutput.Value.String() != "3" ||
		!settlement.Aggregate.ServedTotal.Complete || settlement.Aggregate.ServedTotal.Value.String() != "10" ||
		settlement.FenceID != "" {
		t.Fatalf("cache-hit aggregate = %+v fence=%q", settlement.Aggregate, settlement.FenceID)
	}
	event, err := usageledger.DecodeTerminalEvent(settlement.Event)
	if err != nil {
		t.Fatal(err)
	}
	if len(event.Dispatches) != 1 || event.Dispatches[0].UsageState != usageledger.UsageKnownZero ||
		event.Dispatches[0].Cost.Numerator != "0" || event.Served.InputTokens != "7" ||
		event.Served.OutputTokens != "3" || !event.Served.InputKnown || !event.Served.OutputKnown {
		t.Fatalf("cache-hit terminal event = %+v", event)
	}
}

func TestCacheHitMissingAuthoritativeUsageFencesOnlyServedQuota(t *testing.T) {
	fake := &fakeInferenceAccess{}
	router := &OpenAIRouter{Config: inferenceTestConfig(t), InferenceAccess: fake}
	ctx := admittedInferenceTestContext("internal-model")
	ctx.VSRCacheHit = true
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
	recordCacheHitSettlementUsage(ctx, responseUsageMetrics{
		invalid: true, invalidReason: "cache_authoritative_usage_missing",
	})
	if err := router.settleNoBackendInference(ctx, 200, "cache_short_circuit"); err != nil {
		t.Fatalf("settleNoBackendInference() error = %v", err)
	}
	settlement := fake.settlements[0]
	if !settlement.Aggregate.Total.Complete || !settlement.Aggregate.Total.Value.IsZero() ||
		!settlement.Aggregate.Cost.Complete || !settlement.Aggregate.Cost.Value.IsZero() ||
		settlement.Aggregate.ServedTotal.Complete || settlement.FenceID == "" {
		t.Fatalf("missing cache evidence aggregate = %+v fence=%q", settlement.Aggregate, settlement.FenceID)
	}
	if settlement.Aggregate.Metric(quota.MetricServedTotalTokens).Reason == "" {
		t.Fatal("missing served usage did not retain an unknown reason")
	}
}
