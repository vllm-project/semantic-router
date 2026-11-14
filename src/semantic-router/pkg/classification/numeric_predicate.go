package classification

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

// predicateMatches reports whether value satisfies every bound of predicate;
// without one, whether value is positive.
func predicateMatches(value float64, predicate *config.NumericPredicate) bool {
	if predicate == nil {
		return value > 0
	}
	if predicate.GT != nil && !(value > *predicate.GT) {
		return false
	}
	if predicate.GTE != nil && !(value >= *predicate.GTE) {
		return false
	}
	if predicate.LT != nil && !(value < *predicate.LT) {
		return false
	}
	if predicate.LTE != nil && !(value <= *predicate.LTE) {
		return false
	}
	return true
}
