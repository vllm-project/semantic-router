package config

import (
	"strings"
	"testing"
)

// A score.v1 rule stops contributing a real confidence, so any decision gated
// on it leaves confidence-based ranking. That is a deliberate consequence of
// the contract, not an error - but it changes how decisions are ordered, so it
// must not be silent.
func TestComplexityBackendAdvisories_WarnsThatScoreLeavesConfidenceRanking(t *testing.T) {
	cfg := complexityBackendConfig(&RemoteClassifierBackend{
		Protocol: RemoteClassifierProtocolHTTPClassify,
		Contract: RemoteClassifierContractScore,
		Model:    "difficulty-scorer",
	})

	advisories := ComplexityBackendAdvisories(cfg)

	if len(advisories) == 0 {
		t.Fatal("expected an advisory when a score backend is configured")
	}
	joined := strings.Join(advisories, "\n")
	for _, want := range []string{"confidence", RemoteClassifierContractScore} {
		if !strings.Contains(joined, want) {
			t.Errorf("advisory %q should mention %q", joined, want)
		}
	}
}

// The label contract reports the winning label's probability, so ranking is
// unaffected and there is nothing to warn about.
func TestComplexityBackendAdvisories_SilentForTheLabelContract(t *testing.T) {
	cfg := complexityBackendConfig(&RemoteClassifierBackend{
		Protocol: RemoteClassifierProtocolHTTPClassify,
		Contract: RemoteClassifierContractLabelDistribution,
		Model:    "difficulty-scorer",
	})

	if advisories := ComplexityBackendAdvisories(cfg); len(advisories) != 0 {
		t.Fatalf("label_distribution.v1 reports a real confidence, got advisories: %v", advisories)
	}
}

// Candidate lists are unreachable once a backend produces the score. That is
// not an error - a config being migrated will legitimately still carry them -
// but leaving it unsaid means a user can edit examples for hours with no
// effect.
func TestComplexityBackendAdvisories_WarnsThatCandidatesAreUnreachable(t *testing.T) {
	cfg := complexityBackendConfig(&RemoteClassifierBackend{
		Protocol: RemoteClassifierProtocolHTTPClassify,
		Contract: RemoteClassifierContractScore,
		Model:    "difficulty-scorer",
	})
	cfg.ComplexityRules = []ComplexityRule{{
		Name:      "needs_reasoning",
		HardAbove: floatPtr(0.85),
		EasyBelow: floatPtr(0.60),
		Hard:      ComplexityCandidates{Candidates: []string{"solve this step by step"}},
	}}

	advisories := ComplexityBackendAdvisories(cfg)

	joined := strings.Join(advisories, "\n")
	if !strings.Contains(joined, "needs_reasoning") {
		t.Errorf("advisory %q should name the rule still carrying candidates", joined)
	}
}

// Without a backend the local path runs and every candidate list matters, so
// neither advisory applies.
func TestComplexityBackendAdvisories_SilentWithoutABackend(t *testing.T) {
	cfg := complexityBackendConfig(nil)
	cfg.ComplexityRules = []ComplexityRule{{
		Name:      "needs_reasoning",
		Threshold: 0.10,
		Hard:      ComplexityCandidates{Candidates: []string{"solve this step by step"}},
	}}

	if advisories := ComplexityBackendAdvisories(cfg); len(advisories) != 0 {
		t.Fatalf("the local path warrants no advisories, got: %v", advisories)
	}
}

func TestComplexityBackendAdvisories_NilConfigIsSilent(t *testing.T) {
	if advisories := ComplexityBackendAdvisories(nil); len(advisories) != 0 {
		t.Fatalf("a nil config warrants no advisories, got: %v", advisories)
	}
}
