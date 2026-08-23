package backendinvoker

import (
	"strings"
	"testing"
	"time"
)

func TestDispatchOutcomeIsSignedBoundedAndTamperEvident(t *testing.T) {
	now := time.Unix(1_800_000_000, 0).UTC()
	keyring := SigningKeyring{
		ActiveVersion: "v1", Keys: map[string][]byte{"v1": []byte(strings.Repeat("k", 32))},
		MaxLifetime: time.Minute,
	}
	outcome := testDispatchOutcome(now)
	token, err := keyring.SignOutcome(outcome, now)
	if err != nil {
		t.Fatal(err)
	}
	verified, err := keyring.VerifyOutcome(token, outcome.Audience, now.Add(time.Second))
	if err != nil || verified.SelectedDispatchID != "dispatch-1" || len(verified.Attempted) != 2 {
		t.Fatalf("verified outcome = %+v, %v", verified, err)
	}
	if _, err := keyring.Verify(token, outcome.Audience, now); err == nil {
		t.Fatal("dispatch outcome was accepted as a request capability")
	}
	tampered := []byte(token)
	tampered[len(tampered)-1] ^= 1
	if _, err := keyring.VerifyOutcome(string(tampered), outcome.Audience, now); err == nil {
		t.Fatal("tampered dispatch outcome unexpectedly verified")
	}
	if _, err := keyring.VerifyOutcome(token, "other", now); err == nil {
		t.Fatal("dispatch outcome audience mismatch unexpectedly verified")
	}
}

func TestDispatchOutcomeRejectsInvalidSelectedAndUnsafeFallback(t *testing.T) {
	now := time.Unix(1_800_000_000, 0).UTC()
	keyring := SigningKeyring{
		ActiveVersion: "v1", Keys: map[string][]byte{"v1": []byte(strings.Repeat("k", 32))},
		MaxLifetime: time.Minute,
	}
	for name, mutate := range map[string]func(*DispatchOutcome){
		"selected non-terminal":   func(outcome *DispatchOutcome) { outcome.SelectedDispatchID = "dispatch-0" },
		"fallback after response": func(outcome *DispatchOutcome) { outcome.Attempted[1].FallbackTrigger = FallbackUnavailable },
		"too many attempts":       func(outcome *DispatchOutcome) { outcome.Attempted[0].AttemptCount = 7 },
	} {
		t.Run(name, func(t *testing.T) {
			outcome := testDispatchOutcome(now)
			mutate(&outcome)
			if _, err := keyring.SignOutcome(outcome, now); err == nil {
				t.Fatal("invalid dispatch outcome unexpectedly signed")
			}
		})
	}
}

func testDispatchOutcome(now time.Time) DispatchOutcome {
	return DispatchOutcome{
		NamespaceID: "namespace", QuotaPartition: "partition", PublicationID: "publication",
		RuntimeEpoch: 2, RoutingRevision: 3, RoutingDigest: strings.Repeat("d", 64),
		AdmissionID: "admission", AdmissionDigest: strings.Repeat("b", 64),
		RequestID: "request", RequestDigest: RequestDigest("POST", "/v1/chat/completions", "", []byte(`{}`)),
		Attempted: []DispatchOutcomeCandidate{
			{
				DispatchID: "dispatch-0", DispatchType: "primary", Ordinal: 0,
				DispatchPlanDigest: strings.Repeat("a", 64), ModelID: "model-0", ModelRevision: 1,
				State: AttemptKnownZero, FallbackTrigger: FallbackUnavailable, AttemptCount: 2,
			},
			{
				DispatchID: "dispatch-1", DispatchType: "primary", Ordinal: 1,
				DispatchPlanDigest: strings.Repeat("c", 64), ModelID: "model-1", ModelRevision: 2,
				Priority: 1, State: AttemptResponseStarted, AttemptCount: 1,
			},
		},
		SelectedDispatchID: "dispatch-1", Audience: "backend-invoker",
		IssuedAt: now.Unix(), ExpiresAt: now.Add(30 * time.Second).Unix(),
	}
}
