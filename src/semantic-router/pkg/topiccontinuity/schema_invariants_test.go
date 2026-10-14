package topiccontinuity

import (
	"math"
	"testing"
)

// TestNewResultRejectsInvariantViolations checks that the single constructor
// turns every schema violation into a sanitized internal error.
func TestNewResultRejectsInvariantViolations(t *testing.T) {
	valid := resultInput{
		signal: "s", class: ClassChange, reason: ReasonExplicitChange, confidence: 0.9,
		coverage: CoverageFull, scope: EvidenceScope{
			AssistantIncluded:      true,
			ExcludedContentPresent: true,
		},
	}
	if got := newResult(valid); got.Reason != ReasonExplicitChange {
		t.Fatalf("valid input rejected: %+v", got)
	}
	mutations := map[string]func(*resultInput){
		"nan confidence":               func(in *resultInput) { in.confidence = math.NaN() },
		"confidence above one":         func(in *resultInput) { in.confidence = 1.5 },
		"change without full coverage": func(in *resultInput) { in.coverage = CoverageWindow },
		"weak explicit change":         func(in *resultInput) { in.confidence = 0.5 },
		"disjoint above cap":           func(in *resultInput) { in.reason = ReasonDisjoint; in.confidence = 0.7 },
		"reason of another class":      func(in *resultInput) { in.reason = ReasonReference },
		"undeclared reason":            func(in *resultInput) { in.reason = "change_maybe" },
		"undeclared coverage":          func(in *resultInput) { in.coverage = "most" },
		"unknown with confidence": func(in *resultInput) {
			in.class, in.reason, in.confidence = ClassUnknown, ReasonInconclusive, 0.2
		},
	}
	for name, mutate := range mutations {
		in := valid
		mutate(&in)
		got := newResult(in)
		assertInvariants(t, got)
		if got.Reason != ReasonInternalError || got.Scope.ExcludedContentPresent || got.Coverage != CoveragePartial {
			t.Fatalf("%s: not sanitized: %+v", name, got)
		}
		if !got.Scope.AssistantIncluded {
			t.Fatalf("%s: AssistantIncluded should follow the policy", name)
		}
	}
}
