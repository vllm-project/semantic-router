package backendinvoker

import (
	"strings"
	"testing"
	"time"
)

func TestCapabilityIssuerBindsExactRequestAndOwnsKeys(t *testing.T) {
	now := time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC)
	source := []byte(strings.Repeat("k", 32))
	issuer, err := NewCapabilityIssuer(CapabilityIssuerOptions{
		Audience: "vllm-sr.backend-dispatch",
		Keyring: SigningKeyring{
			ActiveVersion: "v1", Keys: map[string][]byte{"v1": source}, MaxLifetime: 30 * time.Second,
		},
		Lifetime: 10 * time.Second,
		Now:      func() time.Time { return now },
	})
	if err != nil {
		t.Fatal(err)
	}
	for index := range source {
		source[index] = 0
	}
	body := []byte(`{"model":"model-a"}`)
	token, err := issuer.Issue(completeTestCapabilityIssue(CapabilityIssueRequest{
		NamespaceID: "namespace-a", QuotaPartition: "partition-a", RoutingRevision: 7,
		AdmissionID: "admission-a", AdmissionDigest: strings.Repeat("b", 64),
		Candidates: []DispatchCandidate{testDispatchCandidate("dispatch-a", "model-a", 3)},
		Method:     "POST", Path: "/v1/chat/completions", Body: body,
	}))
	if err != nil {
		t.Fatal(err)
	}
	capability, err := issuer.keyring.Verify(token, "vllm-sr.backend-dispatch", now)
	if err != nil {
		t.Fatal(err)
	}
	if capability.QuotaPartition != "partition-a" || capability.RequestDigest != RequestDigest("POST", "/v1/chat/completions", "", body) {
		t.Fatalf("capability = %+v", capability)
	}
	if err := issuer.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := issuer.Issue(CapabilityIssueRequest{}); err == nil {
		t.Fatal("closed issuer accepted a request")
	}
}

func TestCapabilityIssuerRejectsLifetimeAndIncompleteIssue(t *testing.T) {
	keyring := SigningKeyring{
		ActiveVersion: "v1", Keys: map[string][]byte{"v1": []byte(strings.Repeat("k", 32))},
		MaxLifetime: 5 * time.Second,
	}
	if _, err := NewCapabilityIssuer(CapabilityIssuerOptions{
		Audience: "vllm-sr.backend-dispatch", Keyring: keyring, Lifetime: 6 * time.Second,
	}); err == nil {
		t.Fatal("issuer accepted a lifetime beyond the verifier bound")
	}
	issuer, err := NewCapabilityIssuer(CapabilityIssuerOptions{
		Audience: "vllm-sr.backend-dispatch", Keyring: keyring, Lifetime: time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer issuer.Close()
	if _, err := issuer.Issue(CapabilityIssueRequest{}); err == nil {
		t.Fatal("issuer accepted an incomplete dispatch")
	}
}
