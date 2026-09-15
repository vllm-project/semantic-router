package dsl

import (
	"strings"
	"testing"
)

func TestAlgorithmMinimumCandidatesRoundTrip(t *testing.T) {
	input := `
SIGNAL domain test { description: "test" }
ROUTE panel {
  PRIORITY 1
  WHEN domain("test")
  MODEL "m1", "m2", "m3"
  ALGORITHM fusion {
    minimum_candidates: 3
    min_successful_responses: 2
  }
}
`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	if got := cfg.Decisions[0].Algorithm.MinimumCandidates; got != 3 {
		t.Fatalf("minimum_candidates = %d, want 3", got)
	}
	output, err := Decompile(cfg)
	if err != nil {
		t.Fatalf("decompile: %v", err)
	}
	if !strings.Contains(output, "minimum_candidates: 3") {
		t.Fatalf("decompiled DSL omitted minimum_candidates:\n%s", output)
	}
}

func TestAlgorithmMinimumCandidatesConstraint(t *testing.T) {
	diagnostics, parseErrors := Validate(`
ROUTE invalid {
  PRIORITY 1
  ALGORITHM static { minimum_candidates: 0 }
}
`)
	if len(parseErrors) > 0 {
		t.Fatalf("parse errors: %v", parseErrors)
	}
	for _, diagnostic := range diagnostics {
		if strings.Contains(diagnostic.Message, "minimum_candidates must be >= 1") {
			return
		}
	}
	t.Fatalf("missing minimum_candidates constraint diagnostic: %v", diagnostics)
}
