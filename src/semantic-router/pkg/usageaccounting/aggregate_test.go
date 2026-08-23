package usageaccounting

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
)

func TestAggregatorSumsEveryDispatchAndPinnedCost(t *testing.T) {
	pricing, testAggregatorSumsEveryDispatchAndPinnedCostErr := CompilePricing(PricingInput{Currency: "USD", Input: stringPointer("1"), Output: stringPointer("2")})
	if testAggregatorSumsEveryDispatchAndPinnedCostErr != nil {
		t.Fatal(testAggregatorSumsEveryDispatchAndPinnedCostErr)
	}
	aggregator := NewAggregator()
	for _, dispatch := range []DispatchUsage{
		{DispatchID: "dispatch-b", ModelID: "model-2", ModelRevision: 4, State: EvidenceKnownActual, Pricing: pricing, Usage: ActualUsage{InputTotal: mustQuantity(t, "20"), InputKnown: true, Output: mustQuantity(t, "3"), OutputKnown: true}},
		{DispatchID: "dispatch-a", ModelID: "model-1", ModelRevision: 2, State: EvidenceKnownActual, Pricing: pricing, Usage: ActualUsage{InputTotal: mustQuantity(t, "10"), InputKnown: true, Output: mustQuantity(t, "2"), OutputKnown: true}},
	} {
		if err := aggregator.RecordDispatch(dispatch); err != nil {
			t.Fatal(err)
		}
	}
	if err := aggregator.SetServedUsage(ServedUsage{Input: mustQuantity(t, "8"), InputKnown: true, Output: mustQuantity(t, "2"), OutputKnown: true}); err != nil {
		t.Fatal(err)
	}
	result, testAggregatorSumsEveryDispatchAndPinnedCostErr := aggregator.Finalize()
	if testAggregatorSumsEveryDispatchAndPinnedCostErr != nil {
		t.Fatal(testAggregatorSumsEveryDispatchAndPinnedCostErr)
	}
	if result.Input.Value.String() != "30" || result.Output.Value.String() != "5" || result.Total.Value.String() != "35" {
		t.Fatalf("backend totals = %+v", result)
	}
	if result.ServedTotal.Value.String() != "10" || result.Cost.Value.String() != "40000000000" {
		t.Fatalf("served/cost totals = %+v", result)
	}
	if result.KnownDispatches.String() != "2" || result.IncompleteDispatches.String() != "0" || len(result.Digest) != 64 {
		t.Fatalf("evidence summary = %+v", result)
	}
}

func TestAggregatorNeverTreatsUnknownAsZero(t *testing.T) {
	pricing, _ := CompilePricing(PricingInput{Currency: "USD"})
	aggregator := NewAggregator()
	if err := aggregator.RecordDispatch(DispatchUsage{DispatchID: "dispatch-1", ModelID: "model-1", ModelRevision: 1, State: EvidenceUnknown, Pricing: pricing, Reason: "provider_usage_missing"}); err != nil {
		t.Fatal(err)
	}
	result, err := aggregator.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	for _, metric := range []quota.Metric{quota.MetricInputTokens, quota.MetricOutputTokens, quota.MetricTotalTokens, quota.MetricCost} {
		if value := result.Metric(metric); value.Complete || value.Reason == "" {
			t.Fatalf("metric %s silently became complete: %+v", metric, value)
		}
	}
	if result.IncompleteDispatches.String() != "1" {
		t.Fatalf("incomplete dispatches = %s", result.IncompleteDispatches.String())
	}
}

func TestAggregatorDigestIsIndependentOfParallelCompletionOrder(t *testing.T) {
	pricing, _ := CompilePricing(PricingInput{Currency: "USD", Input: stringPointer("0"), Output: stringPointer("0")})
	dispatches := []DispatchUsage{
		{DispatchID: "a", ModelID: "m", ModelRevision: 1, State: EvidenceKnownActual, Pricing: pricing, Usage: ActualUsage{InputKnown: true, OutputKnown: true}},
		{DispatchID: "b", ModelID: "m", ModelRevision: 1, State: EvidenceKnownActual, Pricing: pricing, Usage: ActualUsage{InputKnown: true, OutputKnown: true}},
	}
	left, right := NewAggregator(), NewAggregator()
	for _, value := range dispatches {
		_ = left.RecordDispatch(value)
	}
	for index := len(dispatches) - 1; index >= 0; index-- {
		_ = right.RecordDispatch(dispatches[index])
	}
	leftResult, _ := left.Finalize()
	rightResult, _ := right.Finalize()
	if leftResult.Digest != rightResult.Digest {
		t.Fatalf("digest depends on completion order: %s != %s", leftResult.Digest, rightResult.Digest)
	}
}
