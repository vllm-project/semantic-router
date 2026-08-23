package backendinvoker

import (
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testsupport/signedtoken"
)

func TestDispatchTokensRejectNonCanonicalSignatureEncoding(t *testing.T) {
	now := time.Unix(1_800_000_000, 0).UTC()
	keyring := SigningKeyring{
		ActiveVersion: "v1",
		Keys:          map[string][]byte{"v1": []byte(strings.Repeat("k", 32))},
		MaxLifetime:   time.Minute,
	}
	capability := completeTestCapability(DispatchCapability{
		NamespaceID: "namespace", QuotaPartition: "partition", RoutingRevision: 1,
		AdmissionID: "admission", AdmissionDigest: strings.Repeat("b", 64),
		Candidates:    []DispatchCandidate{testDispatchCandidate("dispatch", "model", 1)},
		RequestDigest: RequestDigest("POST", "/v1/chat/completions", "", []byte(`{}`)),
		Method:        "POST", Path: "/v1/chat/completions", Audience: "backend-invoker",
		IssuedAt: now.Unix(), ExpiresAt: now.Add(30 * time.Second).Unix(),
	})
	capabilityToken, err := keyring.Sign(capability, now)
	if err != nil {
		t.Fatal(err)
	}
	grant := completeTestGrant(DispatchGrant{
		NamespaceID: "namespace", QuotaPartition: "partition", RoutingRevision: 1,
		AdmissionID: "admission", AdmissionDigest: strings.Repeat("b", 64),
		Candidates: []DispatchCandidate{testDispatchCandidate("dispatch", "model", 1)},
		Audience:   "backend-invoker", IssuedAt: now.Unix(), ExpiresAt: now.Add(30 * time.Second).Unix(),
	})
	grantToken, err := keyring.signGrant(grant, now)
	if err != nil {
		t.Fatal(err)
	}
	outcome := testDispatchOutcome(now)
	outcomeToken, err := keyring.SignOutcome(outcome, now)
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name   string
		token  string
		verify func(string) error
	}{
		{name: "capability", token: capabilityToken, verify: func(token string) error {
			_, err := keyring.Verify(token, capability.Audience, now)
			return err
		}},
		{name: "grant", token: grantToken, verify: func(token string) error {
			_, err := keyring.verifyGrant(token, grant.Audience, now)
			return err
		}},
		{name: "outcome", token: outcomeToken, verify: func(token string) error {
			_, err := keyring.VerifyOutcome(token, outcome.Audience, now)
			return err
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if err := test.verify(test.token); err != nil {
				t.Fatal(err)
			}
			if err := test.verify(signedtoken.Alias(t, test.token)); err == nil {
				t.Fatal("non-canonical signature encoding was accepted")
			}
		})
	}
}
