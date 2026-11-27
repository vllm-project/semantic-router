package dsl

import (
	"reflect"
	"testing"
)

func TestCompileASTSeedsModelSelectionDefaults(t *testing.T) {
	cfg, errs := CompileAST(&Program{})
	if len(errs) > 0 {
		t.Fatalf("CompileAST errors: %v", errs)
	}
	if cfg == nil {
		t.Fatal("expected router config from CompileAST")
	}
	if !cfg.ModelSelection.Enabled {
		t.Fatalf("expected CompileAST to enable model selection, got %+v", cfg.ModelSelection)
	}
}

func TestEscalateAnswerCheckRoundTrip(t *testing.T) {
	cfg, errs := Compile(`
ROUTE checked {
  MODEL "small", "large"
  ALGORITHM escalate {
    on_error: "fail"
    order: "cost"
    accept: { systemone: { gte: 0.8, timeout_ms: 5000 } }
  }
}`)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}

	dslText, err := DecompileRouting(cfg)
	if err != nil {
		t.Fatal(err)
	}
	roundTripped, errs := Compile(dslText)
	if len(errs) > 0 {
		t.Fatalf("round-trip compile errors: %v", errs)
	}
	if !reflect.DeepEqual(cfg.Decisions[0].Algorithm, roundTripped.Decisions[0].Algorithm) {
		t.Fatalf("escalate changed during round trip:\n%s\n%#v", dslText, roundTripped.Decisions[0].Algorithm)
	}
	if check := roundTripped.Decisions[0].Algorithm.Escalate.Accept.SystemOne; check == nil || *check.GTE != 0.8 || check.TimeoutMs != 5000 {
		t.Fatalf("accept.systemone = %+v", check)
	}
}
