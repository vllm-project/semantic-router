package soak

import (
	"math"
	"testing"
	"time"
)

const mb = 1024 * 1024

var base = time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

func rssSamples(step time.Duration, valuesMB ...float64) []Sample {
	out := make([]Sample, len(valuesMB))
	for i, v := range valuesMB {
		out[i] = Sample{TS: base.Add(time.Duration(i) * step), RSS: v * mb}
	}
	return out
}

func approx(t *testing.T, got, want, tol float64, what string) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Errorf("%s = %v, want %v (+/- %v)", what, got, want, tol)
	}
}

func TestTailMeanClampsToShortPhases(t *testing.T) {
	samples := rssSamples(5*time.Second, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)
	got := TailMean(samples, 60*time.Second, func(s Sample) float64 { return s.RSS }) / mb
	if got <= 500 {
		t.Errorf("TailMean = %v MB, want a tail-weighted value above 500", got)
	}

	single := rssSamples(time.Second, 700)
	approx(t, TailMean(single, time.Minute, func(s Sample) float64 { return s.RSS })/mb, 700, 0.01, "single sample")
	if got := TailMean(nil, time.Minute, func(s Sample) float64 { return s.RSS }); got != 0 {
		t.Errorf("TailMean(nil) = %v, want 0", got)
	}
}
