package issuerverifier

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
)

const backchannelLogoutEvent = "http://schemas.openid.net/event/backchannel-logout"

func (verifier *Verifier) VerifyBackchannelLogout(
	ctx context.Context,
	issuerID string,
	logoutToken string,
	now time.Time,
) (managementauth.BackchannelLogoutIdentity, error) {
	if verifier == nil || verifier.repository == nil || verifier.keys == nil || now.IsZero() {
		return managementauth.BackchannelLogoutIdentity{}, managementauth.ErrAuthenticationDenied
	}
	issuer, verifyBackchannelLogoutErr := verifier.repository.LoadActive(ctx, issuerID)
	if verifyBackchannelLogoutErr != nil || issuer.ID != issuerID || issuer.Validate() != nil {
		return managementauth.BackchannelLogoutIdentity{}, managementauth.ErrAuthenticationDenied
	}
	assertion, verifyBackchannelLogoutErr := parseAssertion(logoutToken)
	if verifyBackchannelLogoutErr != nil {
		return managementauth.BackchannelLogoutIdentity{}, managementauth.ErrAuthenticationDenied
	}
	algorithm, verifyBackchannelLogoutErr := headerString(assertion.header, "alg")
	if verifyBackchannelLogoutErr != nil || algorithm == "none" {
		return managementauth.BackchannelLogoutIdentity{}, managementauth.ErrAuthenticationDenied
	}
	keyID, verifyBackchannelLogoutErr := headerString(assertion.header, "kid")
	if verifyBackchannelLogoutErr != nil {
		return managementauth.BackchannelLogoutIdentity{}, managementauth.ErrAuthenticationDenied
	}
	if rawType, found := assertion.header["typ"]; found {
		typ, err := stringClaim(map[string]json.RawMessage{"typ": rawType}, "typ", true, 16)
		if err != nil || (typ != "JWT" && typ != "logout+jwt") {
			return managementauth.BackchannelLogoutIdentity{}, managementauth.ErrAuthenticationDenied
		}
	}
	set, verifyBackchannelLogoutErr := verifier.keys.Keys(ctx, issuer)
	if verifyBackchannelLogoutErr != nil {
		return managementauth.BackchannelLogoutIdentity{}, managementauth.ErrAuthenticationDenied
	}
	key, found := set.Keys[keyID]
	if !found || key.Algorithm != algorithm || verifySignature(assertion, key) != nil {
		return managementauth.BackchannelLogoutIdentity{}, managementauth.ErrAuthenticationDenied
	}

	issuerClaim, verifyBackchannelLogoutErr := stringClaim(assertion.claims, "iss", true, 2048)
	if verifyBackchannelLogoutErr != nil || issuerClaim != issuer.Issuer {
		return managementauth.BackchannelLogoutIdentity{}, managementauth.ErrAuthenticationDenied
	}
	audiences, verifyBackchannelLogoutErr := audienceClaim(assertion.claims)
	if verifyBackchannelLogoutErr != nil || !allowedAudience(audiences, issuer.Audiences) {
		return managementauth.BackchannelLogoutIdentity{}, managementauth.ErrAuthenticationDenied
	}
	issuedAt, expiresAt, verifyBackchannelLogoutErr := validateAssertionTimes(assertion.claims, now.UTC(), verifier.skew, verifier.lifetime)
	if verifyBackchannelLogoutErr != nil {
		return managementauth.BackchannelLogoutIdentity{}, managementauth.ErrAuthenticationDenied
	}
	tokenID, verifyBackchannelLogoutErr := stringClaim(assertion.claims, "jti", true, 512)
	if verifyBackchannelLogoutErr != nil {
		return managementauth.BackchannelLogoutIdentity{}, managementauth.ErrAuthenticationDenied
	}
	if _, forbidden := assertion.claims["nonce"]; forbidden {
		return managementauth.BackchannelLogoutIdentity{}, managementauth.ErrAuthenticationDenied
	}
	subject, verifyBackchannelLogoutErr := stringClaim(assertion.claims, "sub", false, 512)
	if verifyBackchannelLogoutErr != nil {
		return managementauth.BackchannelLogoutIdentity{}, managementauth.ErrAuthenticationDenied
	}
	issuerSessionID, verifyBackchannelLogoutErr := stringClaim(assertion.claims, "sid", false, 512)
	if verifyBackchannelLogoutErr != nil || (subject == "" && issuerSessionID == "") {
		return managementauth.BackchannelLogoutIdentity{}, managementauth.ErrAuthenticationDenied
	}
	if !validBackchannelEvents(assertion.claims) {
		return managementauth.BackchannelLogoutIdentity{}, managementauth.ErrAuthenticationDenied
	}
	canonical, verifyBackchannelLogoutErr := json.Marshal(assertion.claims)
	if verifyBackchannelLogoutErr != nil {
		return managementauth.BackchannelLogoutIdentity{}, managementauth.ErrAuthenticationUnavailable
	}
	return managementauth.BackchannelLogoutIdentity{
		IssuerID: issuerID, TokenID: tokenID, Subject: subject,
		IssuerSessionID: issuerSessionID, IssuedAt: issuedAt,
		ExpiresAt: expiresAt, ClaimsDigest: sha256.Sum256(canonical),
	}, nil
}

func validBackchannelEvents(claims map[string]json.RawMessage) bool {
	raw, found := claims["events"]
	if !found {
		return false
	}
	events, err := decodeRawObject(raw)
	if err != nil || len(events) != 1 {
		return false
	}
	payload, found := events[backchannelLogoutEvent]
	if !found {
		return false
	}
	details, err := decodeRawObject(payload)
	return err == nil && len(details) == 0
}

var _ managementauth.BackchannelLogoutVerifier = (*Verifier)(nil)
