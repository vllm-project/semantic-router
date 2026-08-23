// Package issuerverifier verifies human bootstrap assertions against the
// Router-owned trusted-issuer registry. Issuer metadata is desired state;
// verification keys are bounded process-local cache entries, never authority.
package issuerverifier

import (
	"context"
	"crypto"
	"errors"
	"net/url"
	"strings"

	"github.com/google/uuid"
)

const ManagementAudience = "vllm-sr-management"

var (
	ErrDenied      = errors.New("trusted issuer assertion denied")
	ErrUnavailable = errors.New("trusted issuer verification unavailable")
)

type IssuerKind string

const (
	IssuerOIDC IssuerKind = "oidc"
	IssuerJWT  IssuerKind = "jwt"
)

type TrustedIssuer struct {
	ID               string
	Issuer           string
	Kind             IssuerKind
	DiscoveryURL     string
	JWKSURL          string
	Audiences        []string
	ClaimMapping     map[string]string
	AssuranceMapping map[string]string
	Revision         uint64
}

func (issuer TrustedIssuer) Validate() error {
	parsedID, err := uuid.Parse(issuer.ID)
	if err != nil || parsedID.String() != issuer.ID || issuer.Revision == 0 {
		return ErrUnavailable
	}
	if !canonicalHTTPSURL(issuer.Issuer) || len(issuer.Audiences) == 0 || len(issuer.Audiences) > 16 {
		return ErrUnavailable
	}
	seen := make(map[string]struct{}, len(issuer.Audiences))
	for _, audience := range issuer.Audiences {
		if audience == "" || audience != strings.TrimSpace(audience) || len(audience) > 512 {
			return ErrUnavailable
		}
		if _, duplicate := seen[audience]; duplicate {
			return ErrUnavailable
		}
		seen[audience] = struct{}{}
	}
	if issuer.Kind != IssuerOIDC && issuer.Kind != IssuerJWT {
		return ErrUnavailable
	}
	if (issuer.DiscoveryURL == "") == (issuer.JWKSURL == "") {
		return ErrUnavailable
	}
	if issuer.DiscoveryURL != "" && !canonicalHTTPSURL(issuer.DiscoveryURL) {
		return ErrUnavailable
	}
	if issuer.JWKSURL != "" && !canonicalHTTPSURL(issuer.JWKSURL) {
		return ErrUnavailable
	}
	for key, value := range issuer.ClaimMapping {
		if !validMappingName(key) || !validClaimName(value) {
			return ErrUnavailable
		}
	}
	for source, target := range issuer.AssuranceMapping {
		if !validClaimValue(source, 256) || !validAAL(target) {
			return ErrUnavailable
		}
	}
	return nil
}

type Repository interface {
	LoadActive(context.Context, string) (TrustedIssuer, error)
}

type VerificationKey struct {
	Algorithm string
	PublicKey crypto.PublicKey
}

type KeySet struct {
	Keys map[string]VerificationKey
}

type KeySource interface {
	Keys(context.Context, TrustedIssuer) (KeySet, error)
}

func canonicalHTTPSURL(value string) bool {
	parsed, err := url.Parse(value)
	return err == nil && parsed.Scheme == "https" && parsed.Host != "" && parsed.User == nil &&
		parsed.RawQuery == "" && parsed.Fragment == "" && parsed.String() == value
}

func validMappingName(value string) bool {
	switch value {
	case "subject", "email", "email_verified", "display_name", "nonce", "session_id",
		"authentication_time", "aal", "amr":
		return true
	default:
		return false
	}
}

func validClaimName(value string) bool {
	if value == "" || len(value) > 128 || value != strings.TrimSpace(value) {
		return false
	}
	for _, character := range value {
		if character <= 0x20 || character == 0x7f {
			return false
		}
	}
	return true
}

func validClaimValue(value string, limit int) bool {
	if value == "" || len(value) > limit || value != strings.TrimSpace(value) {
		return false
	}
	for _, character := range value {
		if character < 0x20 || character == 0x7f {
			return false
		}
	}
	return true
}

func validAAL(value string) bool {
	return value == "aal1" || value == "aal2" || value == "aal3"
}
