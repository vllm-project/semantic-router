package managementauth

import (
	"context"
	"time"
)

// DenyAllSubjectAssertionVerifier is the explicit fail-closed verifier used
// when a deployment has not installed an OIDC or Router-local human identity
// issuer. It leaves bootstrap and service-account authentication available,
// while every human token exchange remains denied.
type DenyAllSubjectAssertionVerifier struct{}

func (DenyAllSubjectAssertionVerifier) ValidateIssuer(context.Context, string) error {
	return ErrAuthenticationDenied
}

func (DenyAllSubjectAssertionVerifier) Verify(
	context.Context,
	string,
	SubjectTokenType,
	string,
	time.Time,
) (VerifiedExternalIdentity, error) {
	return VerifiedExternalIdentity{}, ErrAuthenticationDenied
}

var _ SubjectAssertionVerifier = DenyAllSubjectAssertionVerifier{}
