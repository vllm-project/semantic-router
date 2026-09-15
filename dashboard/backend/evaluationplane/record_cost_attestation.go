package evaluationplane

import (
	"fmt"
	"math"
)

type optionalOrderedFloatSum struct {
	value   float64
	present bool
}

func (sum *optionalOrderedFloatSum) add(value *float64, field string) error {
	if value == nil {
		return nil
	}
	next := sum.value + *value
	if !finiteFloat(next) {
		return fmt.Errorf("%s aggregate is not finite", field)
	}
	sum.value = next
	sum.present = true
	return nil
}

func (sum optionalOrderedFloatSum) pointer() *float64 {
	if !sum.present {
		return nil
	}
	value := sum.value
	return &value
}

type recordCostAttestation struct {
	RuntimeAmount            *float64
	EvaluationOverheadAmount *float64
	CapacityTCOAmount        *float64
	InputTokens              int64
	OutputTokens             int64
	GPUSeconds               *float64
	EnergyKWh                *float64
}

type recordCostReducer struct {
	runtimeAmount            optionalOrderedFloatSum
	evaluationOverheadAmount optionalOrderedFloatSum
	capacityTCOAmount        optionalOrderedFloatSum
	inputTokens              int64
	outputTokens             int64
	gpuSeconds               optionalOrderedFloatSum
	energyKWh                optionalOrderedFloatSum
}

func (reducer *recordCostReducer) observe(record executionRecordEvidence) error {
	for _, field := range []struct {
		name  string
		value *float64
		sum   *optionalOrderedFloatSum
	}{
		{name: "runtime_cost", value: record.RuntimeCost, sum: &reducer.runtimeAmount},
		{name: "evaluation_cost", value: record.EvaluationCost, sum: &reducer.evaluationOverheadAmount},
		{name: "capacity_tco", value: record.CapacityTCO, sum: &reducer.capacityTCOAmount},
		{name: "gpu_seconds", value: record.GPUSeconds, sum: &reducer.gpuSeconds},
		{name: "energy_kwh", value: record.EnergyKWh, sum: &reducer.energyKWh},
	} {
		if err := field.sum.add(field.value, field.name); err != nil {
			return err
		}
	}
	var err error
	reducer.inputTokens, err = checkedNonnegativeInt64Sum(reducer.inputTokens, record.InputTokens, "input_tokens")
	if err != nil {
		return err
	}
	reducer.outputTokens, err = checkedNonnegativeInt64Sum(reducer.outputTokens, record.OutputTokens, "output_tokens")
	return err
}

func checkedNonnegativeInt64Sum(total int64, value *int64, field string) (int64, error) {
	if value == nil {
		return total, nil
	}
	if *value > math.MaxInt64-total {
		return 0, fmt.Errorf("%s aggregate overflows int64", field)
	}
	return total + *value, nil
}

func (reducer recordCostReducer) finalize() recordCostAttestation {
	return recordCostAttestation{
		RuntimeAmount:            reducer.runtimeAmount.pointer(),
		EvaluationOverheadAmount: reducer.evaluationOverheadAmount.pointer(),
		CapacityTCOAmount:        reducer.capacityTCOAmount.pointer(),
		InputTokens:              reducer.inputTokens,
		OutputTokens:             reducer.outputTokens,
		GPUSeconds:               reducer.gpuSeconds.pointer(),
		EnergyKWh:                reducer.energyKWh.pointer(),
	}
}

func validateServerReducedCosts(actual CostLedgers, expected recordCostAttestation) error {
	if err := validateServerCostAmount("runtime", actual.Runtime, expected.RuntimeAmount, expected.InputTokens, expected.OutputTokens, nil, nil); err != nil {
		return err
	}
	if err := validateServerCostAmount("evaluation_overhead", actual.EvaluationOverhead, expected.EvaluationOverheadAmount, 0, 0, nil, nil); err != nil {
		return err
	}
	return validateServerCostAmount("capacity_tco", actual.CapacityTCO, expected.CapacityTCOAmount, 0, 0, expected.GPUSeconds, expected.EnergyKWh)
}

func validateServerCostAmount(
	name string,
	actual CostAmount,
	expectedAmount *float64,
	expectedInputTokens, expectedOutputTokens int64,
	expectedGPUSeconds, expectedEnergyKWh *float64,
) error {
	if actual.Currency != "USD" {
		return fmt.Errorf("%w: %s cost currency must be USD", ErrInvalid, name)
	}
	if !reducedOptionalFloatEqual(actual.Amount, expectedAmount) {
		return fmt.Errorf("%w: %s cost amount does not match records", ErrInvalid, name)
	}
	if actual.InputTokens != expectedInputTokens || actual.OutputTokens != expectedOutputTokens {
		return fmt.Errorf("%w: %s token totals do not match records", ErrInvalid, name)
	}
	if !reducedOptionalScalarEqual(actual.GPUSeconds, expectedGPUSeconds) || !reducedOptionalScalarEqual(actual.EnergyKWh, expectedEnergyKWh) {
		return fmt.Errorf("%w: %s resource totals do not match records", ErrInvalid, name)
	}
	return nil
}

func reducedOptionalFloatEqual(actual, expected *float64) bool {
	if (actual == nil) != (expected == nil) {
		return false
	}
	return actual == nil || reducedFloatsEqual(*actual, *expected)
}

// CostAmount uses scalar zero values for optional resource totals, so an
// absent worker value and an explicit zero share one decoded representation.
func reducedOptionalScalarEqual(actual float64, expected *float64) bool {
	if expected == nil {
		return actual == 0
	}
	return reducedFloatsEqual(actual, *expected)
}
