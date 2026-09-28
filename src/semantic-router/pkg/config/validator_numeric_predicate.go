package config

import (
	"fmt"
	"math"
)

func validateNumericPredicateContract(
	predicate *NumericPredicate,
) error {
	if predicate == nil {
		return nil
	}
	if !numericPredicateIsFinite(predicate) {
		return fmt.Errorf("predicate values must be finite")
	}
	if structurePredicateComparatorCount(predicate) == 0 {
		return fmt.Errorf("predicate must set at least one comparator")
	}
	if predicate.GT != nil && predicate.GTE != nil {
		return fmt.Errorf("predicate cannot set both gt and gte")
	}
	if predicate.LT != nil && predicate.LTE != nil {
		return fmt.Errorf("predicate cannot set both lt and lte")
	}
	if numericPredicateRangeIsEmpty(predicate) {
		return fmt.Errorf("predicate defines an empty numeric range")
	}
	return nil
}

func numericPredicateIsFinite(predicate *NumericPredicate) bool {
	for _, value := range []*float64{
		predicate.GT,
		predicate.GTE,
		predicate.LT,
		predicate.LTE,
	} {
		if value != nil && (math.IsNaN(*value) || math.IsInf(*value, 0)) {
			return false
		}
	}
	return true
}

func numericPredicateRangeIsEmpty(predicate *NumericPredicate) bool {
	lower, lowerStrict, hasLower := numericPredicateLowerBound(predicate)
	upper, upperStrict, hasUpper := numericPredicateUpperBound(predicate)
	if !hasLower || !hasUpper {
		return false
	}
	if lower > upper {
		return true
	}
	return lower == upper && (lowerStrict || upperStrict)
}

func numericPredicateLowerBound(
	predicate *NumericPredicate,
) (value float64, strict bool, ok bool) {
	if predicate.GT != nil {
		return *predicate.GT, true, true
	}
	if predicate.GTE != nil {
		return *predicate.GTE, false, true
	}
	return 0, false, false
}

func numericPredicateUpperBound(
	predicate *NumericPredicate,
) (value float64, strict bool, ok bool) {
	if predicate.LT != nil {
		return *predicate.LT, true, true
	}
	if predicate.LTE != nil {
		return *predicate.LTE, false, true
	}
	return 0, false, false
}
