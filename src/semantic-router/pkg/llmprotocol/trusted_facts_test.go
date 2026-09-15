package llmprotocol

import "testing"

func TestEvaluateTrustedFacts(t *testing.T) {
	allowed := []TrustedStage{TrustedStageCandidate, TrustedStageFinal}
	cases := []struct {
		name string
		in   TrustedFacts
		want TrustedOutcome
	}{
		{"disabled allows", TrustedFacts{Enforcement: TrustedDisabled}, TrustedAllow},
		{"empty enforcement observes", TrustedFacts{}, TrustedObserve},
		{"advisory observes", TrustedFacts{Enforcement: TrustedAdvisory, Capable: true, Authorized: true, Available: true, Stage: TrustedStageCandidate, AllowedStages: allowed}, TrustedObserve},
		{"authoritative allows when capable, authorized, allowed, and fresh", TrustedFacts{Enforcement: TrustedAuthoritative, Capable: true, Authorized: true, Available: true, Stage: TrustedStageFinal, AllowedStages: allowed}, TrustedAllow},
		{"authoritative narrows when only availability is stale", TrustedFacts{Enforcement: TrustedAuthoritative, Capable: true, Authorized: true, Available: false, Stage: TrustedStageCandidate, AllowedStages: allowed}, TrustedNarrow},
		{"authoritative denies without capability", TrustedFacts{Enforcement: TrustedAuthoritative, Capable: false, Authorized: true, Available: true, Stage: TrustedStageCandidate, AllowedStages: allowed}, TrustedDeny},
		{"authoritative denies without authorization", TrustedFacts{Enforcement: TrustedAuthoritative, Capable: true, Authorized: false, Available: true, Stage: TrustedStageCandidate, AllowedStages: allowed}, TrustedDeny},
		// Runtime evidence alone must never authorize, even when fresh.
		{"authoritative denies runtime-only sources when fresh", TrustedFacts{Enforcement: TrustedAuthoritative, Capable: true, Authorized: false, Available: true, Stage: TrustedStageVerifier, AllowedStages: []TrustedStage{TrustedStageVerifier}}, TrustedDeny},
		{"authoritative denies unknown stage", TrustedFacts{Enforcement: TrustedAuthoritative, Capable: true, Authorized: true, Available: true, Stage: "unknown", AllowedStages: allowed}, TrustedDeny},
		{"authoritative denies disallowed stage", TrustedFacts{Enforcement: TrustedAuthoritative, Capable: true, Authorized: true, Available: true, Stage: TrustedStageFinal, AllowedStages: []TrustedStage{TrustedStageCandidate}}, TrustedDeny},
		{"authoritative denies empty allowed stages", TrustedFacts{Enforcement: TrustedAuthoritative, Capable: true, Authorized: true, Available: true, Stage: TrustedStageCandidate}, TrustedDeny},
		{"unknown enforcement denies", TrustedFacts{Enforcement: "bogus", Capable: true, Authorized: true, Available: true, Stage: TrustedStageCandidate, AllowedStages: allowed}, TrustedDeny},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := EvaluateTrustedFacts(tc.in); got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
}

func TestTrustedSourceAuthorizes(t *testing.T) {
	if !TrustedSourceOperatorPolicy.Authorizes() {
		t.Fatal("operator-policy must authorize")
	}
	for _, s := range []TrustedSource{TrustedSourceGatewayAttested, TrustedSourceRuntimeFresh, "prompt", "client-metadata", ""} {
		if s.Authorizes() {
			t.Fatalf("%q must not authorize by declaration", s)
		}
	}
}
