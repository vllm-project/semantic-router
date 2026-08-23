package issuerverifier

import (
	"context"
	"encoding/json"
	"slices"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
)

const (
	defaultClockSkew        = 30 * time.Second
	defaultMaximumLifetime  = 10 * time.Minute
	maximumSourceLifetime   = 30 * 24 * time.Hour
	routerSourceExpiryClaim = "source_session_exp"
)

type Options struct {
	Repository      Repository
	Keys            KeySource
	ClockSkew       time.Duration
	MaximumLifetime time.Duration
}

type Verifier struct {
	repository Repository
	keys       KeySource
	skew       time.Duration
	lifetime   time.Duration
}

func New(options Options) (*Verifier, error) {
	if options.Repository == nil || options.Keys == nil {
		return nil, ErrUnavailable
	}
	skew := options.ClockSkew
	if skew == 0 {
		skew = defaultClockSkew
	}
	lifetime := options.MaximumLifetime
	if lifetime == 0 {
		lifetime = defaultMaximumLifetime
	}
	if skew < 0 || skew > 2*time.Minute || lifetime < time.Minute || lifetime > time.Hour {
		return nil, ErrUnavailable
	}
	return &Verifier{repository: options.Repository, keys: options.Keys, skew: skew, lifetime: lifetime}, nil
}

func (verifier *Verifier) ValidateIssuer(ctx context.Context, issuerID string) error {
	if verifier == nil || verifier.repository == nil {
		return managementauth.ErrAuthenticationDenied
	}
	issuer, err := verifier.repository.LoadActive(ctx, issuerID)
	if err != nil || issuer.Validate() != nil {
		return managementauth.ErrAuthenticationDenied
	}
	return nil
}

func (verifier *Verifier) Verify(
	ctx context.Context,
	issuerID string,
	tokenType managementauth.SubjectTokenType,
	subjectToken string,
	now time.Time,
) (managementauth.VerifiedExternalIdentity, error) {
	if verifier == nil || verifier.repository == nil || verifier.keys == nil || now.IsZero() ||
		(tokenType != managementauth.SubjectTokenOIDCIDToken && tokenType != managementauth.SubjectTokenRouterAssertion) {
		return managementauth.VerifiedExternalIdentity{}, managementauth.ErrAuthenticationDenied
	}
	issuer, verifyErr := verifier.repository.LoadActive(ctx, issuerID)
	if verifyErr != nil || issuer.Validate() != nil {
		return managementauth.VerifiedExternalIdentity{}, managementauth.ErrAuthenticationDenied
	}
	if tokenType == managementauth.SubjectTokenOIDCIDToken && issuer.Kind != IssuerOIDC {
		return managementauth.VerifiedExternalIdentity{}, managementauth.ErrAuthenticationDenied
	}
	assertion, verifyErr := parseAssertion(subjectToken)
	if verifyErr != nil {
		return managementauth.VerifiedExternalIdentity{}, managementauth.ErrAuthenticationDenied
	}
	algorithm, verifyErr := headerString(assertion.header, "alg")
	if verifyErr != nil || algorithm == "none" {
		return managementauth.VerifiedExternalIdentity{}, managementauth.ErrAuthenticationDenied
	}
	keyID, verifyErr := headerString(assertion.header, "kid")
	if verifyErr != nil {
		return managementauth.VerifiedExternalIdentity{}, managementauth.ErrAuthenticationDenied
	}
	if rawType, exists := assertion.header["typ"]; exists {
		typ, err := stringClaim(map[string]json.RawMessage{"typ": rawType}, "typ", true, 16)
		if err != nil || typ != "JWT" {
			return managementauth.VerifiedExternalIdentity{}, managementauth.ErrAuthenticationDenied
		}
	}
	set, verifyErr := verifier.keys.Keys(ctx, issuer)
	if verifyErr != nil {
		return managementauth.VerifiedExternalIdentity{}, managementauth.ErrAuthenticationDenied
	}
	key, exists := set.Keys[keyID]
	if !exists || key.Algorithm != algorithm || verifySignature(assertion, key) != nil {
		return managementauth.VerifiedExternalIdentity{}, managementauth.ErrAuthenticationDenied
	}
	identity, verifyErr := verifier.identity(issuer, tokenType, assertion.claims, now.UTC())
	if verifyErr != nil {
		return managementauth.VerifiedExternalIdentity{}, managementauth.ErrAuthenticationDenied
	}
	return identity, nil
}

func (verifier *Verifier) identity(
	issuer TrustedIssuer,
	tokenType managementauth.SubjectTokenType,
	claims map[string]json.RawMessage,
	now time.Time,
) (managementauth.VerifiedExternalIdentity, error) {
	issuerClaim, identityErr := stringClaim(claims, "iss", true, 2048)
	if identityErr != nil || issuerClaim != issuer.Issuer {
		return managementauth.VerifiedExternalIdentity{}, ErrDenied
	}
	audiences, identityErr := audienceClaim(claims)
	if identityErr != nil || !allowedAudience(audiences, issuer.Audiences) {
		return managementauth.VerifiedExternalIdentity{}, ErrDenied
	}
	_, assertionExpiresAt, identityErr := validateAssertionTimes(claims, now, verifier.skew, verifier.lifetime)
	if identityErr != nil {
		return managementauth.VerifiedExternalIdentity{}, identityErr
	}
	evidenceExpiresAt, identityErr := verifiedEvidenceExpiry(
		tokenType, claims, now, assertionExpiresAt,
	)
	if identityErr != nil {
		return managementauth.VerifiedExternalIdentity{}, identityErr
	}
	if _, err := stringClaim(claims, "jti", true, 512); err != nil {
		return managementauth.VerifiedExternalIdentity{}, err
	}
	subject, identityErr := stringClaim(claims, claimName(issuer, "subject", "sub"), true, 512)
	if identityErr != nil {
		return managementauth.VerifiedExternalIdentity{}, identityErr
	}
	nonce, identityErr := stringClaim(claims, claimName(issuer, "nonce", "nonce"), true, 512)
	if identityErr != nil {
		return managementauth.VerifiedExternalIdentity{}, identityErr
	}
	authenticationSeconds, identityErr := integerClaim(claims,
		claimName(issuer, "authentication_time", "auth_time"), true)
	if identityErr != nil {
		return managementauth.VerifiedExternalIdentity{}, identityErr
	}
	authenticatedAt := time.Unix(authenticationSeconds, 0).UTC()
	if authenticatedAt.After(now.Add(verifier.skew)) || authenticatedAt.After(assertionExpiresAt) {
		return managementauth.VerifiedExternalIdentity{}, ErrDenied
	}
	aalSource, identityErr := stringClaim(claims, claimName(issuer, "aal", "aal"), true, 256)
	if identityErr != nil {
		return managementauth.VerifiedExternalIdentity{}, identityErr
	}
	aal := aalSource
	if mapped, exists := issuer.AssuranceMapping[aalSource]; exists {
		aal = mapped
	}
	if !validAAL(aal) {
		return managementauth.VerifiedExternalIdentity{}, ErrDenied
	}
	amr, identityErr := stringArrayClaim(claims, claimName(issuer, "amr", "amr"))
	if identityErr != nil {
		return managementauth.VerifiedExternalIdentity{}, identityErr
	}
	slices.Sort(amr)
	email, identityErr := stringClaim(claims, claimName(issuer, "email", "email"), false, 320)
	if identityErr != nil {
		return managementauth.VerifiedExternalIdentity{}, identityErr
	}
	if email != "" {
		verified, present, err := boolClaim(claims, claimName(issuer, "email_verified", "email_verified"))
		if err != nil || !present || !verified {
			return managementauth.VerifiedExternalIdentity{}, ErrDenied
		}
	}
	displayName, identityErr := stringClaim(claims, claimName(issuer, "display_name", "name"), false, 256)
	if identityErr != nil {
		return managementauth.VerifiedExternalIdentity{}, identityErr
	}
	sessionID, identityErr := stringClaim(claims, claimName(issuer, "session_id", "sid"), false, 512)
	if identityErr != nil {
		return managementauth.VerifiedExternalIdentity{}, identityErr
	}
	var sessionIDPointer *string
	if sessionID != "" {
		sessionIDPointer = &sessionID
	}
	return managementauth.VerifiedExternalIdentity{
		IssuerID: issuer.ID, Issuer: issuer.Issuer, Subject: subject,
		VerifiedEmail: email, DisplayName: displayName, Nonce: nonce,
		IssuerSessionID: sessionIDPointer, AAL: aal, AMR: amr,
		AuthenticatedAt: authenticatedAt, EvidenceExpiresAt: evidenceExpiresAt,
	}, nil
}

func verifiedEvidenceExpiry(
	tokenType managementauth.SubjectTokenType,
	claims map[string]json.RawMessage,
	now time.Time,
	assertionExpiresAt time.Time,
) (time.Time, error) {
	if tokenType == managementauth.SubjectTokenOIDCIDToken {
		return assertionExpiresAt, nil
	}
	sourceExpirySeconds, err := integerClaim(claims, routerSourceExpiryClaim, true)
	if err != nil {
		return time.Time{}, ErrDenied
	}
	sourceExpiresAt := time.Unix(sourceExpirySeconds, 0).UTC()
	if !now.Before(sourceExpiresAt) || sourceExpiresAt.Before(assertionExpiresAt) ||
		sourceExpiresAt.After(now.Add(maximumSourceLifetime)) {
		return time.Time{}, ErrDenied
	}
	return sourceExpiresAt, nil
}

var _ managementauth.SubjectAssertionVerifier = (*Verifier)(nil)
