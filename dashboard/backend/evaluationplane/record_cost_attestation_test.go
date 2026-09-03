package evaluationplane

import (
	"math"
	"strings"
	"testing"
)

func TestRecordCostReducerMatchesPythonBuildCostsSemantics(t *testing.T) {
	reducer := recordCostReducer{}
	rows := []executionRecordEvidence{
		{RuntimeCost: floatPointer(0), EvaluationCost: floatPointer(0.25), InputTokens: int64Pointer(2)},
		{RuntimeCost: floatPointer(0.5), CapacityTCO: floatPointer(2), OutputTokens: int64Pointer(3), GPUSeconds: floatPointer(4), EnergyKWh: floatPointer(5)},
		{InputTokens: int64Pointer(0), GPUSeconds: floatPointer(0)},
	}
	for _, row := range rows {
		if err := reducer.observe(row); err != nil {
			t.Fatal(err)
		}
	}
	actual := reducer.finalize()
	assertOptionalReducedFloat(t, actual.RuntimeAmount, 0.5)
	assertOptionalReducedFloat(t, actual.EvaluationOverheadAmount, 0.25)
	assertOptionalReducedFloat(t, actual.CapacityTCOAmount, 2)
	assertOptionalReducedFloat(t, actual.GPUSeconds, 4)
	assertOptionalReducedFloat(t, actual.EnergyKWh, 5)
	if actual.InputTokens != 2 || actual.OutputTokens != 3 {
		t.Fatalf("token totals=%d/%d", actual.InputTokens, actual.OutputTokens)
	}
}

func TestRecordCostReducerChecksIntegerAndFloatOverflow(t *testing.T) {
	maximum := int64(math.MaxInt64)
	reducer := recordCostReducer{}
	if err := reducer.observe(executionRecordEvidence{InputTokens: &maximum}); err != nil {
		t.Fatal(err)
	}
	one := int64(1)
	if err := reducer.observe(executionRecordEvidence{InputTokens: &one}); err == nil || !strings.Contains(err.Error(), "overflows") {
		t.Fatalf("integer overflow error=%v", err)
	}

	large := math.MaxFloat64
	floatReducer := recordCostReducer{}
	if err := floatReducer.observe(executionRecordEvidence{RuntimeCost: &large}); err != nil {
		t.Fatal(err)
	}
	if err := floatReducer.observe(executionRecordEvidence{RuntimeCost: &large}); err == nil || !strings.Contains(err.Error(), "not finite") {
		t.Fatalf("float overflow error=%v", err)
	}
}

func TestValidateServerReducedCostsRejectsForgedLedger(t *testing.T) {
	expected := recordCostAttestation{
		RuntimeAmount: floatPointer(1), InputTokens: 2, OutputTokens: 3,
		EvaluationOverheadAmount: floatPointer(0.5), CapacityTCOAmount: floatPointer(4),
		GPUSeconds: floatPointer(5), EnergyKWh: floatPointer(6),
	}
	valid := CostLedgers{
		Runtime:            CostAmount{Amount: floatPointer(1), Currency: "USD", InputTokens: 2, OutputTokens: 3},
		EvaluationOverhead: CostAmount{Amount: floatPointer(0.5), Currency: "USD"},
		CapacityTCO:        CostAmount{Amount: floatPointer(4), Currency: "USD", GPUSeconds: 5, EnergyKWh: 6},
	}
	if err := validateServerReducedCosts(valid, expected); err != nil {
		t.Fatal(err)
	}
	valid.Runtime.Amount = floatPointer(2)
	if err := validateServerReducedCosts(valid, expected); err == nil || !strings.Contains(err.Error(), "amount does not match") {
		t.Fatalf("forged ledger error=%v", err)
	}
}

func assertOptionalReducedFloat(t *testing.T, actual *float64, expected float64) {
	t.Helper()
	if actual == nil || !reducedFloatsEqual(*actual, expected) {
		t.Fatalf("value=%v, want %v", actual, expected)
	}
}
