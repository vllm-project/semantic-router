package dsl

import "testing"

func TestCompileASTSeedsModelSelectionDefaults(t *testing.T) {
	cfg, errs := CompileAST(&Program{})
	if len(errs) > 0 {
		t.Fatalf("CompileAST errors: %v", errs)
	}
	if cfg == nil {
		t.Fatal("expected router config from CompileAST")
	}
	if !cfg.ModelSelection.RouterDC.UseQueryContrastive ||
		!cfg.ModelSelection.RouterDC.UseModelContrastive ||
		!cfg.ModelSelection.RouterDC.UseCapabilities {
		t.Fatalf("expected CompileAST to seed RouterDC defaults, got %+v", cfg.ModelSelection.RouterDC)
	}
	if !cfg.ModelSelection.Hybrid.NormalizeScores {
		t.Fatalf("expected CompileAST to seed Hybrid defaults, got %+v", cfg.ModelSelection.Hybrid)
	}
}

func TestConfidenceMaxResponseBytesRoundTrip(t *testing.T) {
	cfg, errs := Compile(`
ROUTE confidence {
  MODEL "small", "large"
  ALGORITHM confidence {
    confidence_method: "automix_entailment"
    verifier_server_url: "https://verifier.example.com"
    max_response_bytes: 1234
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
	if got := roundTripped.Decisions[0].Algorithm.Confidence.MaxResponseBytes; got != 1234 {
		t.Fatalf("max_response_bytes = %d, want 1234", got)
	}
}
