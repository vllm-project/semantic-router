package routerreplay

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func TestRecorderUpdateUsageCostClonesStoredValues(t *testing.T) {
	recorder := NewRecorder(store.NewMemoryStore(10, 0))
	recordID, err := recorder.AddRecord(RoutingRecord{
		ID:        "replay-usage-1",
		Decision:  "decision-a",
		RequestID: "req-1",
	})
	if err != nil {
		t.Fatalf("failed to add record: %v", err)
	}

	promptTokens := 120
	completionTokens := 45
	totalTokens := 165
	actualCost := 0.0012
	baselineCost := 0.0033
	costSavings := 0.0021
	currency := "USD"
	baselineModel := "premium-model"
	usage := UsageCost{
		PromptTokens:     &promptTokens,
		CompletionTokens: &completionTokens,
		TotalTokens:      &totalTokens,
		ActualCost:       &actualCost,
		BaselineCost:     &baselineCost,
		CostSavings:      &costSavings,
		Currency:         &currency,
		BaselineModel:    &baselineModel,
	}

	if err := recorder.UpdateUsageCost(recordID, usage); err != nil {
		t.Fatalf("failed to update usage cost: %v", err)
	}

	promptTokens = 999
	completionTokens = 999
	totalTokens = 1998
	actualCost = 9.9
	baselineCost = 19.9
	costSavings = 10.0
	currency = "CNY"
	baselineModel = "mutated-model"

	record, found := recorder.GetRecord(recordID)
	if !found {
		t.Fatal("expected to retrieve updated replay record")
	}

	assertIntPtr(t, record.PromptTokens, 120, "prompt tokens")
	assertIntPtr(t, record.CompletionTokens, 45, "completion tokens")
	assertIntPtr(t, record.TotalTokens, 165, "total tokens")
	assertFloatPtr(t, record.ActualCost, 0.0012, "actual cost")
	assertFloatPtr(t, record.BaselineCost, 0.0033, "baseline cost")
	assertFloatPtr(t, record.CostSavings, 0.0021, "cost savings")
	assertStringPtr(t, record.Currency, "USD", "currency")
	assertStringPtr(t, record.BaselineModel, "premium-model", "baseline model")
}

func assertIntPtr(t *testing.T, value *int, expected int, label string) {
	t.Helper()
	if value == nil || *value != expected {
		t.Fatalf("expected %s=%d, got %#v", label, expected, value)
	}
}

func assertFloatPtr(t *testing.T, value *float64, expected float64, label string) {
	t.Helper()
	if value == nil || *value != expected {
		t.Fatalf("expected %s=%.4f, got %#v", label, expected, value)
	}
}

func assertStringPtr(t *testing.T, value *string, expected string, label string) {
	t.Helper()
	if value == nil || *value != expected {
		t.Fatalf("expected %s=%q, got %#v", label, expected, value)
	}
}
