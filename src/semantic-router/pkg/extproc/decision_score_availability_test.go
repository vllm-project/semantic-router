package extproc

import (
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

func TestDecisionScoreAvailabilityReachesHeadersReplayAndTools(t *testing.T) {
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
			ctx := &RequestContext{VSRSelectedDecisionName: "route", VSRSelectedDecisionConfidence: tc.value, VSRSelectedDecisionConfidenceScored: tc.available}
			builder := newResponseHeaderMutationBuilder()
			addFinalDecisionHeaders(builder, ctx)
			var looperHeaders []*core.HeaderValueOption
			appendLooperRoutingFacts(&looperHeaders, nil, ctx)
			for _, values := range [][]*core.HeaderValueOption{builder.setHeaders, looperHeaders} {
				found := false
				for _, value := range values {
					if value.GetHeader().GetKey() == headers.VSRSelectedConfidence {
						found = true
					}
				}
				if found != tc.available {
					t.Fatalf("header score availability = %v, want %v", found, tc.available)
				}
			}
			record := buildReplayRoutingRecord(ctx, "auto", "backend", "route")
			if record.ConfidenceScoreAvailable != tc.available {
				t.Fatal("replay lost decision score availability")
			}
			input := newToolRetrievalInput("query", "", 1, 1, ctx, nil)
			if input.DecisionConfidenceAvailable != tc.available || (!tc.available && input.DecisionConfidence != 0) {
				t.Fatal("structural confidence became tool model evidence")
			}
		})
	}
}

func TestToolCategoryThresholdRequiresReportedConfidence(t *testing.T) {
	enabled, threshold := true, float32(0.5)
	advanced := &config.AdvancedToolFilteringConfig{UseCategoryFilter: &enabled, CategoryConfidenceThreshold: &threshold}
	ctx := &RequestContext{VSRSelectedCategory: "math", VSRSelectedDecisionConfidence: 1}
	if got := resolveCategory(advanced, ctx); got != "" {
		t.Fatal("unscored structural match satisfied model confidence threshold")
	}
	ctx.VSRSelectedDecisionConfidenceScored = true
	if got := resolveCategory(advanced, ctx); got != "math" {
		t.Fatal("reported model score did not satisfy category threshold")
	}
}
