package dsl

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// multiEmitProjectionDSL declares a mapping with method: "multi_emit" and two
// deliberately overlapping bands so the method carries meaning end-to-end.
const multiEmitProjectionDSL = `
SIGNAL keyword reasoning_request_markers {
  operator: "OR"
  keywords: ["reason carefully"]
}

PROJECTION score difficulty_score {
  method: "weighted_sum"
  inputs: [
    { type: "keyword", name: "reasoning_request_markers", weight: 0.6 }
  ]
}

PROJECTION mapping difficulty_band {
  source: "difficulty_score"
  method: "multi_emit"
  outputs: [
    { name: "balance_medium", lt: 0.7 },
    { name: "balance_reasoning", gte: 0.5 }
  ]
}

ROUTE reasoning_route {
  PRIORITY 200
  WHEN projection("balance_reasoning")
  MODEL "m1"
}

ROUTE medium_route {
  PRIORITY 100
  WHEN projection("balance_medium")
  MODEL "m2"
}
`

func TestCompileAcceptsMultiEmitProjectionMapping(t *testing.T) {
	cfg, errs := Compile(multiEmitProjectionDSL)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	if len(cfg.Projections.Mappings) != 1 {
		t.Fatalf("expected 1 projection mapping, got %d", len(cfg.Projections.Mappings))
	}
	if got := cfg.Projections.Mappings[0].Method; got != config.ProjectionMappingMethodMultiEmit {
		t.Fatalf("mapping method = %q, want %q", got, config.ProjectionMappingMethodMultiEmit)
	}
}

func TestMultiEmitProjectionMappingSurvivesRoundTrip(t *testing.T) {
	cfg, errs := Compile(multiEmitProjectionDSL)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}

	dslText, err := DecompileRouting(cfg)
	if err != nil {
		t.Fatalf("DecompileRouting error: %v", err)
	}
	if !strings.Contains(dslText, `method: "multi_emit"`) {
		t.Fatalf("decompiled DSL missing multi_emit method:\n%s", dslText)
	}

	roundTripped, compileErrs := Compile(dslText)
	if len(compileErrs) > 0 {
		t.Fatalf("re-compile errors: %v", compileErrs)
	}
	if len(roundTripped.Projections.Mappings) != 1 {
		t.Fatalf("expected 1 mapping after round-trip, got %d", len(roundTripped.Projections.Mappings))
	}
	if got := roundTripped.Projections.Mappings[0].Method; got != config.ProjectionMappingMethodMultiEmit {
		t.Fatalf("round-tripped mapping method = %q, want %q", got, config.ProjectionMappingMethodMultiEmit)
	}
}
