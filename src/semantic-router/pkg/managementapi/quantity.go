package managementapi

import (
	"encoding/json"
	"fmt"
	"math/big"
	"regexp"
	"strings"
)

var (
	wholeQuantityPattern   = regexp.MustCompile(`^(0|[1-9][0-9]*)$`)
	decimalQuantityPattern = regexp.MustCompile(`^(0|[1-9][0-9]*)(\.[0-9]+)?$`)
	currencyDecimalPattern = regexp.MustCompile(`^(0|[1-9][0-9]*)(\.[0-9]{1,15})?$`)
	currencyCodePattern    = regexp.MustCompile(`^[A-Z]{3}$`)
)

func ParseWholeQuantity(value string) (WholeQuantity, error) {
	if !wholeQuantityPattern.MatchString(value) {
		return "", fmt.Errorf("whole quantity must be a canonical non-negative integer string")
	}
	return WholeQuantity(value), nil
}

func ParseDecimalQuantity(value string) (DecimalQuantity, error) {
	if !decimalQuantityPattern.MatchString(value) {
		return "", fmt.Errorf("decimal quantity must be a canonical non-negative plain decimal string")
	}
	return DecimalQuantity(value), nil
}

func ParseCurrencyDecimal(value string) (CurrencyDecimal, error) {
	if !currencyDecimalPattern.MatchString(value) {
		return "", fmt.Errorf("currency decimal must be canonical, non-negative, and contain at most 15 fractional digits")
	}
	return CurrencyDecimal(value), nil
}

func (q WholeQuantity) MarshalJSON() ([]byte, error) {
	if _, err := ParseWholeQuantity(string(q)); err != nil {
		return nil, err
	}
	return json.Marshal(string(q))
}

func (q *WholeQuantity) UnmarshalJSON(data []byte) error {
	var value string
	if err := json.Unmarshal(data, &value); err != nil {
		return fmt.Errorf("whole quantity must be a JSON string: %w", err)
	}
	parsed, err := ParseWholeQuantity(value)
	if err != nil {
		return err
	}
	*q = parsed
	return nil
}

func (q DecimalQuantity) MarshalJSON() ([]byte, error) {
	if _, err := ParseDecimalQuantity(string(q)); err != nil {
		return nil, err
	}
	return json.Marshal(string(q))
}

func (q *DecimalQuantity) UnmarshalJSON(data []byte) error {
	var value string
	if err := json.Unmarshal(data, &value); err != nil {
		return fmt.Errorf("decimal quantity must be a JSON string: %w", err)
	}
	parsed, err := ParseDecimalQuantity(value)
	if err != nil {
		return err
	}
	*q = parsed
	return nil
}

func (q CurrencyDecimal) MarshalJSON() ([]byte, error) {
	if _, err := ParseCurrencyDecimal(string(q)); err != nil {
		return nil, err
	}
	return json.Marshal(string(q))
}

func (q *CurrencyDecimal) UnmarshalJSON(data []byte) error {
	var value string
	if err := json.Unmarshal(data, &value); err != nil {
		return fmt.Errorf("currency decimal must be a JSON string: %w", err)
	}
	parsed, err := ParseCurrencyDecimal(value)
	if err != nil {
		return err
	}
	*q = parsed
	return nil
}

func (summary CostSummary) Validate() error {
	if !currencyCodePattern.MatchString(summary.Currency) {
		return fmt.Errorf("cost currency must be an ISO-4217 uppercase code")
	}
	if _, err := ParseCurrencyDecimal(string(summary.KnownAmount)); err != nil {
		return err
	}
	known, err := quantityInteger(summary.KnownDispatches)
	if err != nil {
		return fmt.Errorf("known dispatches: %w", err)
	}
	incomplete, err := quantityInteger(summary.IncompleteDispatches)
	if err != nil {
		return fmt.Errorf("incomplete dispatches: %w", err)
	}
	zero := big.NewInt(0)
	switch summary.Completeness {
	case CostComplete:
		if known.Cmp(zero) <= 0 || incomplete.Cmp(zero) != 0 {
			return fmt.Errorf("complete cost requires known dispatches and no incomplete dispatches")
		}
	case CostPartial:
		if known.Cmp(zero) <= 0 || incomplete.Cmp(zero) <= 0 {
			return fmt.Errorf("partial cost requires both known and incomplete dispatches")
		}
	case CostUnknown:
		if known.Cmp(zero) != 0 || incomplete.Cmp(zero) <= 0 {
			return fmt.Errorf("unknown cost requires only incomplete dispatches")
		}
	default:
		return fmt.Errorf("unsupported cost completeness %q", summary.Completeness)
	}
	return nil
}

func (summary CostSummary) MarshalJSON() ([]byte, error) {
	if err := summary.Validate(); err != nil {
		return nil, err
	}
	type wire CostSummary
	return json.Marshal(wire(summary))
}

func (summary *CostSummary) UnmarshalJSON(data []byte) error {
	type wire CostSummary
	var decoded wire
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	parsed := CostSummary(decoded)
	if err := parsed.Validate(); err != nil {
		return err
	}
	*summary = parsed
	return nil
}

func (meter QuotaMeter) Validate() error {
	if !oneOf(meter.Enforcement, "enforce", "shadow") {
		return fmt.Errorf("unsupported quota enforcement %q", meter.Enforcement)
	}
	if !oneOf(meter.Accounting, "request", "response_actual") {
		return fmt.Errorf("unsupported quota accounting %q", meter.Accounting)
	}
	if !oneOf(meter.Algorithm, "sliding_log", "calendar_window", "token_bucket", "gcra", "concurrency") {
		return fmt.Errorf("unsupported quota algorithm %q", meter.Algorithm)
	}
	if !oneOf(meter.Metric, "requests", "input_tokens", "output_tokens", "total_tokens", "served_input_tokens", "served_output_tokens", "served_total_tokens", "cost", "concurrent_requests") {
		return fmt.Errorf("unsupported quota metric %q", meter.Metric)
	}
	known, err := quantityInteger(meter.KnownDispatches)
	if err != nil {
		return fmt.Errorf("known dispatches: %w", err)
	}
	incomplete, err := quantityInteger(meter.IncompleteDispatches)
	if err != nil {
		return fmt.Errorf("incomplete dispatches: %w", err)
	}
	zero := big.NewInt(0)
	switch meter.Completeness {
	case "complete":
		if incomplete.Cmp(zero) != 0 {
			return fmt.Errorf("complete quota meter cannot contain incomplete dispatches")
		}
	case "partial":
		if known.Cmp(zero) <= 0 || incomplete.Cmp(zero) <= 0 {
			return fmt.Errorf("partial quota meter requires known and incomplete dispatches")
		}
	case "unknown":
		if known.Cmp(zero) != 0 || incomplete.Cmp(zero) <= 0 {
			return fmt.Errorf("unknown quota meter requires only incomplete dispatches")
		}
	default:
		return fmt.Errorf("unsupported quota completeness %q", meter.Completeness)
	}
	switch meter.CapacityState {
	case "available", "exhausted", "over_limit":
		if meter.Completeness != "complete" || meter.Remaining == nil {
			return fmt.Errorf("known capacity requires a complete meter and remaining capacity")
		}
	case "fenced":
		if meter.Enforcement != "enforce" || meter.Completeness == "complete" || meter.Remaining != nil || len(meter.ActiveFenceIDs) == 0 {
			return fmt.Errorf("fenced capacity requires enforced incomplete usage and an active fence")
		}
	case "unknown":
		if meter.Enforcement != "shadow" || meter.Completeness == "complete" || meter.Remaining != nil || len(meter.ActiveFenceIDs) != 0 {
			return fmt.Errorf("unknown capacity requires shadow incomplete usage without a fence")
		}
	default:
		return fmt.Errorf("unsupported quota capacity state %q", meter.CapacityState)
	}
	if meter.Remaining == nil && meter.CapacityState != "fenced" && meter.CapacityState != "unknown" {
		return fmt.Errorf("complete quota meter must include remaining capacity")
	}
	if meter.Remaining != nil && (meter.CapacityState == "fenced" || meter.CapacityState == "unknown") {
		return fmt.Errorf("incomplete quota meter must not claim remaining capacity")
	}
	if meter.CapacityState == "over_limit" && meter.Overage == nil {
		return fmt.Errorf("over-limit quota meter requires overage")
	}
	if meter.CapacityState != "over_limit" && meter.Overage != nil {
		return fmt.Errorf("overage is valid only for over-limit quota meters")
	}
	if meter.Metric == "cost" {
		if !currencyCodePattern.MatchString(meter.Currency) {
			return fmt.Errorf("cost quota meter requires an ISO-4217 uppercase currency")
		}
		for _, quantity := range meterQuantities(meter) {
			name, value := quantity.name, quantity.value
			if _, err := ParseCurrencyDecimal(string(value)); err != nil {
				return fmt.Errorf("quota %s: %w", name, err)
			}
		}
		return validateMeterCapacity(meter)
	}
	if strings.TrimSpace(meter.Currency) != "" {
		return fmt.Errorf("non-cost quota meter must not carry currency")
	}
	for _, quantity := range meterQuantities(meter) {
		name, value := quantity.name, quantity.value
		if _, err := ParseWholeQuantity(string(value)); err != nil {
			return fmt.Errorf("quota %s: %w", name, err)
		}
	}
	return validateMeterCapacity(meter)
}

func validateMeterCapacity(meter QuotaMeter) error {
	limit, ok := new(big.Rat).SetString(string(meter.Limit))
	if !ok || limit.Sign() <= 0 {
		return fmt.Errorf("quota limit must be positive")
	}
	used, ok := new(big.Rat).SetString(string(meter.Used))
	if !ok {
		return fmt.Errorf("quota used is invalid")
	}
	if meter.Completeness != "complete" {
		return nil
	}
	remaining, ok := new(big.Rat).SetString(string(*meter.Remaining))
	if !ok {
		return fmt.Errorf("quota remaining is invalid")
	}
	difference := new(big.Rat).Sub(new(big.Rat).Set(limit), used)
	switch meter.CapacityState {
	case "available":
		if difference.Sign() <= 0 || remaining.Cmp(difference) != 0 {
			return fmt.Errorf("available quota remaining must equal limit minus used")
		}
	case "exhausted":
		if difference.Sign() != 0 || remaining.Sign() != 0 {
			return fmt.Errorf("exhausted quota must have used equal limit and zero remaining")
		}
	case "over_limit":
		overage, ok := new(big.Rat).SetString(string(*meter.Overage))
		if !ok || difference.Sign() >= 0 || remaining.Sign() != 0 || overage.Cmp(new(big.Rat).Neg(difference)) != 0 {
			return fmt.Errorf("over-limit quota must expose exact positive overage and zero remaining")
		}
	}
	return nil
}

func oneOf(value string, candidates ...string) bool {
	for _, candidate := range candidates {
		if value == candidate {
			return true
		}
	}
	return false
}

func (meter QuotaMeter) MarshalJSON() ([]byte, error) {
	if err := meter.Validate(); err != nil {
		return nil, err
	}
	type wire QuotaMeter
	return json.Marshal(wire(meter))
}

func (meter *QuotaMeter) UnmarshalJSON(data []byte) error {
	type wire QuotaMeter
	var decoded wire
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	parsed := QuotaMeter(decoded)
	if err := parsed.Validate(); err != nil {
		return err
	}
	*meter = parsed
	return nil
}

type namedMeterQuantity struct {
	name  string
	value DecimalQuantity
}

func meterQuantities(meter QuotaMeter) []namedMeterQuantity {
	quantities := []namedMeterQuantity{
		{name: "limit", value: meter.Limit},
		{name: "used", value: meter.Used},
	}
	if meter.Remaining != nil {
		quantities = append(quantities, namedMeterQuantity{name: "remaining", value: *meter.Remaining})
	}
	if meter.Overage != nil {
		quantities = append(quantities, namedMeterQuantity{name: "overage", value: *meter.Overage})
	}
	return quantities
}

func quantityInteger(quantity WholeQuantity) (*big.Int, error) {
	if _, err := ParseWholeQuantity(string(quantity)); err != nil {
		return nil, err
	}
	value := new(big.Int)
	value.SetString(string(quantity), 10)
	return value, nil
}
