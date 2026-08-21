package accesscontrol

import (
	"testing"
	"time"
)

func TestResolveUsageGranularity(t *testing.T) {
	now := time.Date(2026, time.August, 21, 12, 0, 0, 0, time.UTC)
	tests := []struct {
		name      string
		requested string
		duration  time.Duration
		want      string
	}{
		{name: "auto minute", duration: 4 * time.Hour, want: "minute"},
		{name: "auto hour", duration: 7 * 24 * time.Hour, want: "hour"},
		{name: "auto day", duration: 31 * 24 * time.Hour, want: "day"},
		{name: "explicit minute", requested: "minute", duration: 24 * time.Hour, want: "minute"},
		{name: "minute bounded", requested: "minute", duration: 7 * 24 * time.Hour, want: "hour"},
		{name: "hour bounded", requested: "hour", duration: 120 * 24 * time.Hour, want: "day"},
		{name: "explicit day", requested: "day", duration: time.Hour, want: "day"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := resolveUsageGranularity(test.requested, now.Add(-test.duration), now)
			if got.name != test.want {
				t.Fatalf("granularity = %q, want %q", got.name, test.want)
			}
		})
	}
}
