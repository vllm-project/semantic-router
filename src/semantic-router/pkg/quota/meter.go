package quota

import (
	"errors"
	"fmt"
)

var ErrInvalidMeter = errors.New("invalid quota meter")

// Completeness states whether a live meter includes every started dispatch.
type Completeness string

const (
	CompletenessComplete Completeness = "complete"
	CompletenessPartial  Completeness = "partial"
	CompletenessUnknown  Completeness = "unknown"
)

// CapacityState is the public admission-facing interpretation of a meter.
type CapacityState string

const (
	CapacityAvailable CapacityState = "available"
	CapacityExhausted CapacityState = "exhausted"
	CapacityOverLimit CapacityState = "over_limit"
	CapacityFenced    CapacityState = "fenced"
	CapacityUnknown   CapacityState = "unknown"
)

// MeterSnapshot is one atomic live counter observation. Limit and Used are
// whole units for every metric except cost, where both are exact 10^15-scaled
// currency quantities.
type MeterSnapshot struct {
	Counter              CounterIdentity
	Metric               Metric
	Enforcement          Enforcement
	Limit                QuotaInteger
	Used                 QuotaInteger
	Currency             string
	KnownDispatches      QuotaInteger
	IncompleteDispatches QuotaInteger
	FenceOpen            bool
}

// PublicMeter is the precision-safe wire representation. Remaining is always
// present: complete meters contain a canonical string and incomplete meters
// contain null. Overage is present only in over_limit state.
type PublicMeter struct {
	BindingID            string        `json:"bindingId"`
	RuleID               string        `json:"ruleId"`
	Metric               Metric        `json:"metric"`
	Enforcement          Enforcement   `json:"enforcement"`
	Limit                string        `json:"limit"`
	Used                 string        `json:"used"`
	Remaining            *string       `json:"remaining"`
	Overage              *string       `json:"overage,omitempty"`
	Currency             string        `json:"currency,omitempty"`
	Completeness         Completeness  `json:"completeness"`
	KnownDispatches      string        `json:"knownDispatches"`
	IncompleteDispatches string        `json:"incompleteDispatches"`
	CapacityState        CapacityState `json:"capacityState"`
}

// NewPublicMeter derives completeness, remaining, overage, and capacity state
// without approximation.
func NewPublicMeter(snapshot MeterSnapshot) (PublicMeter, error) {
	if err := snapshot.Counter.Validate(); err != nil {
		return PublicMeter{}, fmt.Errorf("%w: %w", ErrInvalidMeter, err)
	}
	if !snapshot.Metric.valid() {
		return PublicMeter{}, fmt.Errorf("%w: unsupported metric %q", ErrInvalidMeter, snapshot.Metric)
	}
	if !snapshot.Enforcement.valid() {
		return PublicMeter{}, fmt.Errorf("%w: unsupported enforcement %q", ErrInvalidMeter, snapshot.Enforcement)
	}
	if snapshot.Limit.IsZero() {
		return PublicMeter{}, fmt.Errorf("%w: limit must be positive", ErrInvalidMeter)
	}
	if snapshot.Metric == MetricCost {
		if !validCurrencyCode(snapshot.Currency) {
			return PublicMeter{}, fmt.Errorf("%w: cost requires a three-letter uppercase currency", ErrInvalidMeter)
		}
	} else if snapshot.Currency != "" {
		return PublicMeter{}, fmt.Errorf("%w: currency is valid only for cost", ErrInvalidMeter)
	}
	if snapshot.FenceOpen {
		if snapshot.Enforcement != EnforcementEnforce {
			return PublicMeter{}, fmt.Errorf("%w: a shadow meter cannot have an enforcement fence", ErrInvalidMeter)
		}
	}

	meter := PublicMeter{
		BindingID:            snapshot.Counter.BindingID,
		RuleID:               snapshot.Counter.RuleID,
		Metric:               snapshot.Metric,
		Enforcement:          snapshot.Enforcement,
		Currency:             snapshot.Currency,
		KnownDispatches:      snapshot.KnownDispatches.String(),
		IncompleteDispatches: snapshot.IncompleteDispatches.String(),
	}
	meter.Limit, meter.Used = publicMeterValues(snapshot.Metric, snapshot.Limit, snapshot.Used)
	meter.Completeness = meterCompleteness(snapshot.KnownDispatches, snapshot.IncompleteDispatches)

	// A corrected fence remains an admission barrier until the durable ledger
	// commit completes. Its incomplete count may already be zero during that
	// interval, but public capacity must remain fenced and undisclosed.
	if snapshot.FenceOpen {
		meter.CapacityState = CapacityFenced
		return meter, nil
	}
	if !snapshot.IncompleteDispatches.IsZero() {
		if snapshot.Enforcement == EnforcementEnforce {
			meter.CapacityState = CapacityFenced
		} else {
			meter.CapacityState = CapacityUnknown
		}
		return meter, nil
	}

	comparison := snapshot.Used.Compare(snapshot.Limit)
	switch {
	case comparison < 0:
		remaining, _ := snapshot.Limit.Sub(snapshot.Used)
		value := publicMeterValue(snapshot.Metric, remaining)
		meter.Remaining = &value
		meter.CapacityState = CapacityAvailable
	case comparison == 0:
		value := "0"
		meter.Remaining = &value
		meter.CapacityState = CapacityExhausted
	default:
		overage, _ := snapshot.Used.Sub(snapshot.Limit)
		remaining := "0"
		overageValue := publicMeterValue(snapshot.Metric, overage)
		meter.Remaining = &remaining
		meter.Overage = &overageValue
		meter.CapacityState = CapacityOverLimit
	}
	return meter, nil
}

func meterCompleteness(known, incomplete QuotaInteger) Completeness {
	if incomplete.IsZero() {
		return CompletenessComplete
	}
	if known.IsZero() {
		return CompletenessUnknown
	}
	return CompletenessPartial
}

func publicMeterValues(metric Metric, limit, used QuotaInteger) (string, string) {
	return publicMeterValue(metric, limit), publicMeterValue(metric, used)
}

func publicMeterValue(metric Metric, value QuotaInteger) string {
	if metric == MetricCost {
		return NewCurrencyDecimalFromScaled(value).String()
	}
	return value.String()
}

func validCurrencyCode(value string) bool {
	if len(value) != 3 {
		return false
	}
	for index := range value {
		if value[index] < 'A' || value[index] > 'Z' {
			return false
		}
	}
	return true
}
