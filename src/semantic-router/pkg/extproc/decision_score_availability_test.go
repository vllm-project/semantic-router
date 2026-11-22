package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

func TestDecisionScoreAvailabilityReachesTraceAndReplay(t *testing.T) {
	for _, tc := range []struct {
		name      string
		value     float64
		available bool
	}{
		{"structural-match", 1, false},
		{"reported-zero", 0, true},
		{"model-score", 0.91, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := debugRequestContext()
			ctx.VSRSelectedDecisionName, ctx.VSRSelectedDecisionConfidence, ctx.VSRSelectedDecisionConfidenceScored = "route", tc.value, tc.available
			if _, found := traceMembersOf(t, responseTrace(ctx))[headers.TraceConfidence]; found != tc.available {
				t.Fatalf("trace score availability = %v, want %v", found, tc.available)
			}
			record := buildReplayRoutingRecord(ctx, "vllm-sr/auto", "backend", "route")
			if record.ConfidenceScoreAvailable != tc.available {
				t.Fatal("replay lost decision score availability")
			}
		})
	}
}
