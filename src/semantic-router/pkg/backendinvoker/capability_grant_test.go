package backendinvoker

import (
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestDispatchGrantMintsOnlyMatchingExactCapability(t *testing.T) {
	now := time.Unix(1_800_000_000, 0).UTC()
	issuer, err := NewCapabilityIssuer(CapabilityIssuerOptions{
		Audience: "vllm-sr.backend-dispatch",
		Keyring: SigningKeyring{ActiveVersion: "v1", Keys: map[string][]byte{
			"v1": []byte(strings.Repeat("k", 32)),
		}, MaxLifetime: time.Minute},
		Lifetime: 30 * time.Second, Now: func() time.Time { return now },
	})
	if err != nil {
		t.Fatal(err)
	}
	defer issuer.Close()
	grantToken, err := issuer.IssueGrant(completeTestGrantIssue(DispatchGrantIssueRequest{
		NamespaceID: "namespace", QuotaPartition: "partition", RoutingRevision: 4,
		AdmissionID: "admission", AdmissionDigest: strings.Repeat("b", 64),
		Candidates: []DispatchCandidate{{
			DispatchID: "dispatch", DispatchType: "looper", Ordinal: 2,
			DispatchPlanDigest: strings.Repeat("a", 64), ModelID: "model-id", ModelRevision: 7,
		}},
	}))
	if err != nil {
		t.Fatal(err)
	}
	verified, err := issuer.VerifyGrant(grantToken)
	if err != nil {
		t.Fatal(err)
	}
	body := []byte(`{"model":"logical-model"}`)
	capabilityToken, err := issuer.IssueFromGrant(verified, DispatchFinalRequest{
		Method: "POST", Path: "/v1/chat/completions", WireFormat: llmprotocol.OpenAIChatV1, Body: body,
	})
	if err != nil {
		t.Fatal(err)
	}
	capability, err := issuer.keyring.Verify(capabilityToken, "vllm-sr.backend-dispatch", now)
	if err != nil {
		t.Fatal(err)
	}
	if len(capability.Candidates) != 1 || capability.Candidates[0].DispatchID != "dispatch" ||
		capability.AdmissionDigest != strings.Repeat("b", 64) ||
		capability.RequestDigest != RequestDigest("POST", "/v1/chat/completions", "", body) {
		t.Fatalf("capability = %+v", capability)
	}
}

func TestDispatchGrantIsDistinctAudienceBoundAndTamperEvident(t *testing.T) {
	now := time.Unix(1_800_000_000, 0).UTC()
	keyring := SigningKeyring{ActiveVersion: "v1", Keys: map[string][]byte{
		"v1": []byte(strings.Repeat("k", 32)),
	}, MaxLifetime: time.Minute}
	grant := completeTestGrant(DispatchGrant{
		NamespaceID: "namespace", QuotaPartition: "partition", RoutingRevision: 4,
		AdmissionID: "admission", AdmissionDigest: strings.Repeat("b", 64),
		Candidates: []DispatchCandidate{{
			DispatchID: "dispatch", DispatchType: "looper", Ordinal: 2,
			DispatchPlanDigest: strings.Repeat("a", 64), ModelID: "model-id", ModelRevision: 7,
		}},
		Audience: "vllm-sr.backend-dispatch", IssuedAt: now.Unix(), ExpiresAt: now.Add(30 * time.Second).Unix(),
	})
	token, err := keyring.signGrant(grant, now)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := keyring.Verify(token, grant.Audience, now); err == nil {
		t.Fatal("dispatch grant was accepted as a backend capability")
	}
	if _, err := keyring.verifyGrant(token, "other", now); err == nil {
		t.Fatal("dispatch grant audience mismatch unexpectedly accepted")
	}
	tampered := []byte(token)
	tampered[len(tampered)-1] ^= 1
	if _, err := keyring.verifyGrant(string(tampered), grant.Audience, now); err == nil {
		t.Fatal("tampered dispatch grant unexpectedly accepted")
	}
}
