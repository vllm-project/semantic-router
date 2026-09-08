package llmprotocol

import "testing"

func TestEvaluateTrustedFacts(t *testing.T) {
	cases := []struct {
		name string
		in   TrustedFacts
		want TrustedOutcome
	}{
		{"disabled allows", TrustedFacts{Enforcement: TrustedDisabled}, TrustedAllow},
		{"advisory observes", TrustedFacts{Enforcement: TrustedAdvisory, TrustSources: []string{"operator-policy"}, Stage: TrustedStageCandidate, Fresh: true}, TrustedObserve},
		{"authoritative allows when fresh", TrustedFacts{Enforcement: TrustedAuthoritative, TrustSources: []string{"operator-policy"}, Stage: TrustedStageFinal, Fresh: true}, TrustedAllow},
		{"authoritative narrows when stale", TrustedFacts{Enforcement: TrustedAuthoritative, TrustSources: []string{"gateway-attested"}, Stage: TrustedStageVerifier, Fresh: false}, TrustedNarrow},
		{"authoritative denies without source", TrustedFacts{Enforcement: TrustedAuthoritative, Stage: TrustedStageCandidate, Fresh: true}, TrustedDeny},
		{"authoritative denies unknown stage", TrustedFacts{Enforcement: TrustedAuthoritative, TrustSources: []string{"operator-policy"}, Stage: "unknown", Fresh: true}, TrustedDeny},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := EvaluateTrustedFacts(tc.in); got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
}
