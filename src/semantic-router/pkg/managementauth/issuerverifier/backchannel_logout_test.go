package issuerverifier

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"testing"
	"time"
)

func TestBackchannelLogoutVerifierAcceptsSIDAndSubjectSelectors(t *testing.T) {
	now := time.Date(2026, 8, 23, 2, 3, 4, 0, time.UTC)
	public, private, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	verifier := testVerifier(t, public)
	for _, selector := range []map[string]any{
		{"sid": "issuer-session-42"},
		{"sub": "user-42"},
		{"sid": "issuer-session-42", "sub": "user-42"},
	} {
		claims := validBackchannelClaims(now)
		delete(claims, "sid")
		for key, value := range selector {
			claims[key] = value
		}
		identity, err := verifier.VerifyBackchannelLogout(
			context.Background(), testIssuerID, signedAssertion(t, private, claims), now,
		)
		if err != nil {
			t.Fatal(err)
		}
		if identity.IssuerID != testIssuerID || identity.TokenID != "logout-1" ||
			identity.IssuerSessionID != stringValueAny(selector["sid"]) ||
			identity.Subject != stringValueAny(selector["sub"]) || identity.ClaimsDigest == [32]byte{} {
			t.Fatalf("verified logout identity = %+v", identity)
		}
	}
}

func TestBackchannelLogoutVerifierRejectsLogoutTokenConfusion(t *testing.T) {
	now := time.Date(2026, 8, 23, 2, 3, 4, 0, time.UTC)
	public, private, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name   string
		mutate func(map[string]any)
	}{
		{name: "missing selector", mutate: func(claims map[string]any) { delete(claims, "sid") }},
		{name: "nonce forbidden", mutate: func(claims map[string]any) { claims["nonce"] = "challenge" }},
		{name: "missing event", mutate: func(claims map[string]any) { delete(claims, "events") }},
		{name: "wrong event", mutate: func(claims map[string]any) { claims["events"] = map[string]any{"other": map[string]any{}} }},
		{name: "event payload", mutate: func(claims map[string]any) {
			claims["events"] = map[string]any{backchannelLogoutEvent: map[string]any{"unexpected": true}}
		}},
		{name: "missing jti", mutate: func(claims map[string]any) { delete(claims, "jti") }},
		{name: "expired", mutate: func(claims map[string]any) { claims["exp"] = now.Add(-time.Minute).Unix() }},
	} {
		t.Run(test.name, func(t *testing.T) {
			claims := validBackchannelClaims(now)
			test.mutate(claims)
			if _, err := testVerifier(t, public).VerifyBackchannelLogout(
				context.Background(), testIssuerID, signedAssertion(t, private, claims), now,
			); err == nil {
				t.Fatal("VerifyBackchannelLogout accepted an invalid token")
			}
		})
	}
}

func validBackchannelClaims(now time.Time) map[string]any {
	return map[string]any{
		"iss": "https://issuer.example", "aud": ManagementAudience,
		"iat": now.Unix(), "exp": now.Add(5 * time.Minute).Unix(), "jti": "logout-1",
		"sid":    "issuer-session-42",
		"events": map[string]any{backchannelLogoutEvent: map[string]any{}},
	}
}

func stringValueAny(value any) string {
	text, _ := value.(string)
	return text
}
