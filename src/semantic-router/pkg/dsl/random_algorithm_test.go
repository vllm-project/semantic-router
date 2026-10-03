package dsl

import (
	"strings"
	"testing"
)

const randomAlgorithmDSL = `
SIGNAL domain other { description: "other" }
ROUTE random_route {
  PRIORITY 122
  WHEN domain("other")
  MODEL "qwen3-8b", "qwen3-32b"
  ALGORITHM random {}
}
`

func TestRandomAlgorithmCompiles(t *testing.T) {
	cfg, errs := Compile(randomAlgorithmDSL)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	if got := cfg.Decisions[0].Algorithm.Type; got != "random" {
		t.Fatalf("algorithm type = %q, want %q", got, "random")
	}
}

func TestRandomAlgorithmRoundTrips(t *testing.T) {
	cfg, errs := Compile(randomAlgorithmDSL)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	output, err := Decompile(cfg)
	if err != nil {
		t.Fatalf("decompile: %v", err)
	}
	// Blockless algorithms emit bare, with no field block, exactly like static.
	if !strings.Contains(output, "ALGORITHM random\n") {
		t.Fatalf("decompiled DSL omitted the random algorithm:\n%s", output)
	}
	if strings.Contains(output, "ALGORITHM random {") {
		t.Fatalf("decompiled DSL gave random a spurious field block:\n%s", output)
	}

	recompiled, errs := Compile(output)
	if len(errs) > 0 {
		t.Fatalf("recompile errors: %v", errs)
	}
	if got := recompiled.Decisions[0].Algorithm.Type; got != "random" {
		t.Fatalf("round-tripped algorithm type = %q, want %q", got, "random")
	}
}

func TestRandomAlgorithmValidates(t *testing.T) {
	diagnostics, parseErrors := Validate(randomAlgorithmDSL)
	if len(parseErrors) > 0 {
		t.Fatalf("parse errors: %v", parseErrors)
	}
	for _, diagnostic := range diagnostics {
		if strings.Contains(diagnostic.Message, "random") {
			t.Fatalf("unexpected diagnostic for random algorithm: %s", diagnostic.Message)
		}
	}
}

func TestRandomAlgorithmUnknownTypeStillRejected(t *testing.T) {
	_, errs := Compile(strings.Replace(randomAlgorithmDSL, "ALGORITHM random", "ALGORITHM bogus", 1))
	for _, err := range errs {
		if strings.Contains(err.Error(), `unknown algorithm type "bogus"`) {
			return
		}
	}
	t.Fatalf("missing unknown algorithm type error: %v", errs)
}
