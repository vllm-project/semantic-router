package dispatchauthority

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
)

func TestRuntimeVerifiesOutcomeAgainstExactGeneration(t *testing.T) {
	now := time.Unix(1_900_000_000, 0).UTC()
	options := testIssuerOptions(now)
	authority, err := newMeteredAuthority(
		&recordingPreparer{prepared: testPreparedIdentity()},
		options,
	)
	if err != nil {
		t.Fatal(err)
	}
	runtime := &Runtime{metered: authority}
	defer runtime.Close()

	outcome := testSignedOutcome(now, options.Audience)
	token, err := options.Keyring.SignOutcome(outcome, now)
	if err != nil {
		t.Fatal(err)
	}
	expected := OutcomeVerificationRequest{
		Generation: testGrantVerification("request-1").Generation,
		RequestID:  "request-1",
	}
	verified, err := runtime.VerifyDispatchOutcome(context.Background(), token, expected)
	if err != nil || verified.SelectedDispatchID != "dispatch-1" {
		t.Fatalf("VerifyDispatchOutcome() = %+v, %v", verified, err)
	}

	wrongGeneration := expected
	wrongGeneration.Generation.RuntimeEpoch++
	if _, err := runtime.VerifyDispatchOutcome(context.Background(), token, wrongGeneration); err == nil {
		t.Fatal("VerifyDispatchOutcome() accepted a different generation")
	}
	tampered := token[:len(token)-1] + "A"
	if _, err := runtime.VerifyDispatchOutcome(context.Background(), tampered, expected); err == nil {
		t.Fatal("VerifyDispatchOutcome() accepted a tampered token")
	}
	invalidRequest := expected
	invalidRequest.RequestID = strings.Repeat("x", 257)
	if _, err := runtime.VerifyDispatchOutcome(context.Background(), token, invalidRequest); err == nil {
		t.Fatal("VerifyDispatchOutcome() accepted an unbounded request identity")
	}
}

func testSignedOutcome(now time.Time, audience string) backendinvoker.DispatchOutcome {
	return backendinvoker.DispatchOutcome{
		NamespaceID: "namespace-1", QuotaPartition: "partition-1",
		PublicationID: "publication-1", RuntimeEpoch: 2,
		RoutingRevision: 29, RoutingDigest: strings.Repeat("d", 64),
		AdmissionID: "admission-1", AdmissionDigest: strings.Repeat("b", 64),
		RequestID: "request-1",
		RequestDigest: backendinvoker.RequestDigest(
			"POST", "/v1/chat/completions", "", []byte(`{}`),
		),
		Attempted: []backendinvoker.DispatchOutcomeCandidate{{
			DispatchID: "dispatch-1", DispatchType: primaryDispatchType, Ordinal: 3,
			DispatchPlanDigest: strings.Repeat("a", 64), ModelID: "model-1",
			ModelRevision: 1, Priority: 0,
			State: backendinvoker.AttemptResponseStarted, AttemptCount: 1,
		}},
		SelectedDispatchID: "dispatch-1", Audience: audience,
		IssuedAt: now.Unix(), ExpiresAt: now.Add(30 * time.Second).Unix(),
	}
}
