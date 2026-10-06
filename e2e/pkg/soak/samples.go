package soak

import "time"

// FilterSamples returns the samples whose phase satisfies match.
func FilterSamples(samples []Sample, match func(Sample) bool) []Sample {
	out := make([]Sample, 0, len(samples))
	for _, s := range samples {
		if match(s) {
			out = append(out, s)
		}
	}
	return out
}

// TailMean averages a sample field over the trailing window, clamped so short
// quick-mode phases still yield a baseline.
func TailMean(samples []Sample, window time.Duration, field func(Sample) float64) float64 {
	if len(samples) == 0 {
		return 0
	}
	span := samples[len(samples)-1].TS.Sub(samples[0].TS)
	if window <= 0 || window > span/2 {
		window = span / 2
	}
	cutoff := samples[len(samples)-1].TS.Add(-window)
	var sum float64
	var n int
	for _, s := range samples {
		if !s.TS.Before(cutoff) {
			sum += field(s)
			n++
		}
	}
	if n == 0 {
		return field(samples[len(samples)-1])
	}
	return sum / float64(n)
}

// MaxField returns the maximum of a sample field, or 0 for an empty slice.
func MaxField(samples []Sample, field func(Sample) float64) float64 {
	var max float64
	for i, s := range samples {
		v := field(s)
		if i == 0 || v > max {
			max = v
		}
	}
	return max
}

// MinField returns the minimum of a sample field, or 0 for an empty slice.
func MinField(samples []Sample, field func(Sample) float64) float64 {
	var min float64
	for i, s := range samples {
		v := field(s)
		if i == 0 || v < min {
			min = v
		}
	}
	return min
}
