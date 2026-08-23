package issuerverifier

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
)

const testIssuerID = "11111111-1111-4111-8111-111111111111"

type repositoryStub struct {
	issuer TrustedIssuer
	err    error
}

func (repository repositoryStub) LoadActive(context.Context, string) (TrustedIssuer, error) {
	return repository.issuer, repository.err
}

type keySourceStub struct {
	keys KeySet
	err  error
}

func (source keySourceStub) Keys(context.Context, TrustedIssuer) (KeySet, error) {
	return source.keys, source.err
}

func TestVerifierAcceptsBoundEdDSAAssertion(t *testing.T) {
	now := time.Date(2026, 8, 23, 1, 2, 3, 0, time.UTC)
	sourceExpiresAt := now.Add(12 * time.Hour)
	public, private, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	verifier := testVerifier(t, public)
	token := signedAssertion(t, private, map[string]any{
		"iss": "https://issuer.example", "aud": []string{ManagementAudience},
		"sub": "user-42", "iat": now.Add(-time.Second).Unix(), "exp": now.Add(5 * time.Minute).Unix(),
		routerSourceExpiryClaim: sourceExpiresAt.Unix(),
		"jti":                   "assertion-1", "nonce": "challenge-nonce", "sid": "issuer-session-1",
		"auth_time": now.Add(-time.Minute).Unix(), "aal": "aal2", "amr": []string{"pwd", "mfa"},
		"email": "person@example.com", "email_verified": true, "name": "Person",
	})
	identity, err := verifier.Verify(context.Background(), testIssuerID,
		managementauth.SubjectTokenRouterAssertion, token, now)
	if err != nil {
		t.Fatal(err)
	}
	if identity.IssuerID != testIssuerID || identity.Subject != "user-42" ||
		identity.Nonce != "challenge-nonce" || identity.AAL != "aal2" ||
		identity.VerifiedEmail != "person@example.com" || identity.DisplayName != "Person" ||
		identity.IssuerSessionID == nil || *identity.IssuerSessionID != "issuer-session-1" ||
		len(identity.AMR) != 2 || identity.AMR[0] != "mfa" || identity.AMR[1] != "pwd" ||
		!identity.EvidenceExpiresAt.Equal(sourceExpiresAt) {
		t.Fatalf("verified identity = %+v", identity)
	}
}

func TestVerifierKeepsOIDCEvidenceBoundToTokenExpiry(t *testing.T) {
	now := time.Date(2026, 8, 23, 1, 2, 3, 0, time.UTC)
	tokenExpiresAt := now.Add(5 * time.Minute)
	public, private, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	verifier := testVerifier(t, public)
	token := signedAssertion(t, private, map[string]any{
		"iss": "https://issuer.example", "aud": ManagementAudience, "sub": "oidc-user",
		"iat": now.Unix(), "exp": tokenExpiresAt.Unix(), "jti": "oidc-token-1",
		"nonce": "challenge-nonce", "auth_time": now.Unix(), "aal": "aal2", "amr": []string{"pwd"},
	})
	identity, err := verifier.Verify(
		context.Background(), testIssuerID, managementauth.SubjectTokenOIDCIDToken, token, now,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !identity.EvidenceExpiresAt.Equal(tokenExpiresAt) {
		t.Fatalf("EvidenceExpiresAt = %v, want %v", identity.EvidenceExpiresAt, tokenExpiresAt)
	}
}

func TestVerifierRejectsTrustBoundaryViolations(t *testing.T) {
	now := time.Date(2026, 8, 23, 1, 2, 3, 0, time.UTC)
	public, private, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	valid := map[string]any{
		"iss": "https://issuer.example", "aud": ManagementAudience,
		"sub": "user-42", "iat": now.Unix(), "exp": now.Add(5 * time.Minute).Unix(),
		routerSourceExpiryClaim: now.Add(12 * time.Hour).Unix(),
		"jti":                   "assertion-1", "nonce": "challenge-nonce", "auth_time": now.Unix(),
		"aal": "aal2", "amr": []string{"pwd"},
	}
	for _, test := range []struct {
		name   string
		mutate func(map[string]any)
	}{
		{name: "wrong issuer", mutate: func(claims map[string]any) { claims["iss"] = "https://attacker.example" }},
		{name: "wrong audience", mutate: func(claims map[string]any) { claims["aud"] = "somewhere-else" }},
		{name: "expired", mutate: func(claims map[string]any) { claims["exp"] = now.Add(-time.Minute).Unix() }},
		{name: "excessive lifetime", mutate: func(claims map[string]any) { claims["exp"] = now.Add(time.Hour).Unix() }},
		{name: "missing jti", mutate: func(claims map[string]any) { delete(claims, "jti") }},
		{name: "missing nonce", mutate: func(claims map[string]any) { delete(claims, "nonce") }},
		{name: "missing source session expiry", mutate: func(claims map[string]any) {
			delete(claims, routerSourceExpiryClaim)
		}},
		{name: "expired source session", mutate: func(claims map[string]any) {
			claims[routerSourceExpiryClaim] = now.Add(-time.Second).Unix()
		}},
		{name: "source session expires before assertion", mutate: func(claims map[string]any) {
			claims[routerSourceExpiryClaim] = now.Add(4 * time.Minute).Unix()
		}},
		{name: "source session exceeds maximum lifetime", mutate: func(claims map[string]any) {
			claims[routerSourceExpiryClaim] = now.Add(maximumSourceLifetime + time.Second).Unix()
		}},
		{name: "unverified email", mutate: func(claims map[string]any) {
			claims["email"], claims["email_verified"] = "person@example.com", false
		}},
		{name: "unknown assurance", mutate: func(claims map[string]any) { claims["aal"] = "superuser" }},
	} {
		t.Run(test.name, func(t *testing.T) {
			claims := make(map[string]any, len(valid))
			for key, value := range valid {
				claims[key] = value
			}
			test.mutate(claims)
			verifier := testVerifier(t, public)
			if _, err := verifier.Verify(context.Background(), testIssuerID,
				managementauth.SubjectTokenRouterAssertion, signedAssertion(t, private, claims), now); err == nil {
				t.Fatal("Verify() accepted an invalid assertion")
			}
		})
	}
}

func TestVerifierRejectsDuplicateClaimsAndAlgorithmConfusion(t *testing.T) {
	now := time.Date(2026, 8, 23, 1, 2, 3, 0, time.UTC)
	public, private, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	verifier := testVerifier(t, public)
	header, _ := json.Marshal(map[string]any{"alg": "EdDSA", "kid": "signing-1", "typ": "JWT"})
	claims := `{"iss":"https://issuer.example","iss":"https://attacker.example"}`
	unsigned := base64.RawURLEncoding.EncodeToString(header) + "." +
		base64.RawURLEncoding.EncodeToString([]byte(claims))
	token := unsigned + "." + base64.RawURLEncoding.EncodeToString(ed25519.Sign(private, []byte(unsigned)))
	if _, err := verifier.Verify(context.Background(), testIssuerID,
		managementauth.SubjectTokenRouterAssertion, token, now); err == nil {
		t.Fatal("Verify() accepted duplicate JWT claims")
	}

	validClaims := map[string]any{
		"iss": "https://issuer.example", "aud": ManagementAudience, "sub": "user-42",
		"iat": now.Unix(), "exp": now.Add(5 * time.Minute).Unix(), "jti": "assertion-1",
		routerSourceExpiryClaim: now.Add(12 * time.Hour).Unix(),
		"nonce":                 "challenge-nonce", "auth_time": now.Unix(), "aal": "aal2", "amr": []string{"pwd"},
	}
	confused := signedAssertionWithHeader(t, private,
		map[string]any{"alg": "RS256", "kid": "signing-1", "typ": "JWT"}, validClaims)
	if _, err := verifier.Verify(context.Background(), testIssuerID,
		managementauth.SubjectTokenRouterAssertion, confused, now); err == nil {
		t.Fatal("Verify() accepted a key/algorithm mismatch")
	}
}

func testVerifier(t *testing.T, public ed25519.PublicKey) *Verifier {
	t.Helper()
	issuer := TrustedIssuer{
		ID: testIssuerID, Issuer: "https://issuer.example", Kind: IssuerOIDC,
		DiscoveryURL: "https://issuer.example/.well-known/openid-configuration",
		Audiences:    []string{ManagementAudience}, ClaimMapping: map[string]string{},
		AssuranceMapping: map[string]string{}, Revision: 1,
	}
	verifier, err := New(Options{
		Repository: repositoryStub{issuer: issuer},
		Keys: keySourceStub{keys: KeySet{Keys: map[string]VerificationKey{
			"signing-1": {Algorithm: "EdDSA", PublicKey: public},
		}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	return verifier
}

func signedAssertion(t *testing.T, private ed25519.PrivateKey, claims map[string]any) string {
	t.Helper()
	return signedAssertionWithHeader(t, private,
		map[string]any{"alg": "EdDSA", "kid": "signing-1", "typ": "JWT"}, claims)
}

func signedAssertionWithHeader(
	t *testing.T,
	private ed25519.PrivateKey,
	header map[string]any,
	claims map[string]any,
) string {
	t.Helper()
	headerBytes, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	claimBytes, err := json.Marshal(claims)
	if err != nil {
		t.Fatal(err)
	}
	unsigned := base64.RawURLEncoding.EncodeToString(headerBytes) + "." +
		base64.RawURLEncoding.EncodeToString(claimBytes)
	return unsigned + "." + base64.RawURLEncoding.EncodeToString(ed25519.Sign(private, []byte(unsigned)))
}
