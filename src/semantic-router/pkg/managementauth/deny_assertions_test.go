package managementauth

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestDenyAllSubjectAssertionVerifierFailsClosed(t *testing.T) {
	if err := (DenyAllSubjectAssertionVerifier{}).ValidateIssuer(context.Background(),
		"11111111-1111-4111-8111-111111111111"); !errors.Is(err, ErrAuthenticationDenied) {
		t.Fatalf("ValidateIssuer() error = %v", err)
	}
	verified, err := (DenyAllSubjectAssertionVerifier{}).Verify(
		context.Background(), "issuer", SubjectTokenOIDCIDToken, "assertion", time.Now(),
	)
	if !errors.Is(err, ErrAuthenticationDenied) {
		t.Fatalf("Verify() error = %v, want ErrAuthenticationDenied", err)
	}
	if verified.IssuerID != "" || verified.Issuer != "" || verified.Subject != "" ||
		verified.Nonce != "" || verified.IssuerSessionID != nil || verified.AAL != "" ||
		len(verified.AMR) != 0 || !verified.AuthenticatedAt.IsZero() || !verified.EvidenceExpiresAt.IsZero() {
		t.Fatalf("Verify() identity = %#v, want zero value", verified)
	}
}
