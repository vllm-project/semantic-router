package config

import (
	"math"
	"time"
)

// DurationFromSeconds converts a configured second count, including fractions,
// into a duration. 0.8 becomes 800ms. Non-positive, NaN, and infinite values
// return 0 so callers can apply their own default.
func DurationFromSeconds(seconds float64) time.Duration {
	if seconds <= 0 || math.IsNaN(seconds) || math.IsInf(seconds, 0) {
		return 0
	}
	return time.Duration(math.Round(seconds * float64(time.Second)))
}
