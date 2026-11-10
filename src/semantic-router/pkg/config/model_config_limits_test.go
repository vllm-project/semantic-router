package config

import (
	"math"
	"testing"
	"time"
)

func TestDurationFromSecondsKeepsFractions(t *testing.T) {
	cases := []struct {
		name    string
		seconds float64
		want    time.Duration
	}{
		{name: "fractional", seconds: 0.8, want: 800 * time.Millisecond},
		{name: "integer", seconds: 5, want: 5 * time.Second},
		{name: "zero", seconds: 0, want: 0},
		{name: "negative", seconds: -1, want: 0},
		{name: "not a number", seconds: math.NaN(), want: 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := DurationFromSeconds(tc.seconds); got != tc.want {
				t.Fatalf("DurationFromSeconds(%v) = %v, want %v", tc.seconds, got, tc.want)
			}
		})
	}
}
