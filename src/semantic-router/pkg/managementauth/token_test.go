package managementauth

import (
	"crypto/ed25519"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

func testTokenCodec(t *testing.T) TokenCodec {
	t.Helper()
	oldPublic, _, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatal(err)
	}
	activePublic, activePrivate, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatal(err)
	}
	return TokenCodec{
		Keyring: securitykeyring.Signing{
			ActiveVersion: "sig-2",
			Private:       map[string]ed25519.PrivateKey{"sig-2": activePrivate},
			Public:        map[string]ed25519.PublicKey{"sig-1": oldPublic, "sig-2": activePublic},
		},
		Issuer: "vllm-sr", Audience: "vllm-sr-management", MaxSkew: 5 * time.Second,
	}
}

func validHumanClaims(now time.Time) Claims {
	return Claims{
		Issuer: "vllm-sr", Subject: "principal-1", SessionID: "session-1", TokenID: "token-1",
		Audience: "vllm-sr-management", IssuedAt: now.Unix(), ExpiresAt: now.Add(15 * time.Minute).Unix(),
		AuthSourceKind: "oidc", AuthSourceID: "issuer-1", EvidenceKind: EvidenceHuman,
		Human: &HumanEvidence{AuthenticationTime: now.Add(-time.Minute).Unix(), AAL: "aal2", AMR: []string{"pwd", "otp"}},
	}
}

func TestTokenCodecRoundTrip(t *testing.T) {
	now := time.Date(2026, 8, 22, 1, 2, 3, 0, time.UTC)
	codec := testTokenCodec(t)
	token, err := codec.Issue(validHumanClaims(now))
	if err != nil {
		t.Fatalf("Issue() error = %v", err)
	}
	claims, err := codec.Verify(token, now.Add(time.Minute))
	if err != nil {
		t.Fatalf("Verify() error = %v", err)
	}
	if claims.Subject != "principal-1" || claims.EvidenceKind != EvidenceHuman || claims.Human == nil {
		t.Fatalf("Verify() claims = %+v", claims)
	}
}

func TestTokenCodecSupportsWorkloadEvidence(t *testing.T) {
	now := time.Date(2026, 8, 22, 1, 2, 3, 0, time.UTC)
	codec := testTokenCodec(t)
	claims := validHumanClaims(now)
	claims.AuthSourceKind = "service_credential"
	claims.AuthSourceID = "service-credential-1"
	claims.EvidenceKind = EvidenceWorkload
	claims.Human = nil
	claims.Workload = &WorkloadEvidence{Class: "workload_strong", SourceAssuredAt: now.Add(-time.Hour).Unix()}
	token, err := codec.Issue(claims)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := codec.Verify(token, now); err != nil {
		t.Fatalf("Verify() workload error = %v", err)
	}
}

func TestTokenCodecRejectsTamperingExpiryAudienceAndMixedEvidence(t *testing.T) {
	now := time.Date(2026, 8, 22, 1, 2, 3, 0, time.UTC)
	codec := testTokenCodec(t)
	valid := validHumanClaims(now)
	token, err := codec.Issue(valid)
	if err != nil {
		t.Fatal(err)
	}
	last := "A"
	if strings.HasSuffix(token, last) {
		last = "B"
	}
	cases := []struct {
		name  string
		token string
		now   time.Time
	}{
		{"tampered", token[:len(token)-1] + last, now},
		{"expired", token, now.Add(time.Hour)},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			if _, err := codec.Verify(test.token, test.now); !errors.Is(err, ErrInvalidToken) {
				t.Fatalf("Verify() error = %v", err)
			}
		})
	}

	wrongAudience := codec
	wrongAudience.Audience = "other"
	if _, err := wrongAudience.Verify(token, now); !errors.Is(err, ErrInvalidToken) {
		t.Fatalf("Verify() wrong audience error = %v", err)
	}

	mixed := valid
	mixed.Workload = &WorkloadEvidence{Class: "workload_strong", SourceAssuredAt: now.Unix()}
	if _, err := codec.Issue(mixed); !errors.Is(err, ErrInvalidToken) {
		t.Fatalf("Issue() mixed evidence error = %v", err)
	}
}
