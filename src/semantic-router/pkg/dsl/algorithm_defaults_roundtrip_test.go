package dsl

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

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

// Covers supported-field parity only, not rejection of unsupported fields.
func TestAutoMixSupportedFieldsRoundTrip(t *testing.T) {
	cfg, errs := Compile(`
ROUTE automix_route {
  MODEL "small", "large"
  ALGORITHM automix {
    verification_threshold: 0.82
    max_escalations: 3
    cost_aware_routing: true
    cost_quality_tradeoff: 0.35
    discount_factor: 0.9
    use_logprob_verification: true
  }
}`)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	want := config.AutoMixSelectionConfig{
		VerificationThreshold:  0.82,
		MaxEscalations:         3,
		CostAwareRouting:       true,
		CostQualityTradeoff:    0.35,
		DiscountFactor:         0.9,
		UseLogprobVerification: true,
	}
	if got := *cfg.Decisions[0].Algorithm.AutoMix; got != want {
		t.Fatalf("compiled AutoMix = %+v, want %+v", got, want)
	}

	dslText, err := DecompileRouting(cfg)
	if err != nil {
		t.Fatal(err)
	}
	roundTripped, errs := Compile(dslText)
	if len(errs) > 0 {
		t.Fatalf("round-trip compile errors: %v", errs)
	}
	if got := *roundTripped.Decisions[0].Algorithm.AutoMix; got != want {
		t.Fatalf("round-tripped AutoMix = %+v, want %+v", got, want)
	}
}
