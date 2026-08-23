package managementapi

import "time"

// ManagementSession is the non-secret durable session view shared by self and
// cluster administrators. Token IDs and assurance payloads are never exposed.
type ManagementSession struct {
	SessionID       string     `json:"sessionId"`
	PrincipalID     string     `json:"principalId"`
	AuthSourceKind  string     `json:"authSourceKind"`
	EvidenceKind    string     `json:"evidenceKind"`
	AuthenticatedAt time.Time  `json:"authenticatedAt"`
	ExpiresAt       time.Time  `json:"expiresAt"`
	Status          string     `json:"status"`
	RevokedAt       *time.Time `json:"revokedAt,omitempty"`
	CreatedAt       time.Time  `json:"createdAt"`
}

type ManagementSessionPage struct {
	Data []ManagementSession `json:"data"`
	Page PageInfo            `json:"page"`
}

type ManagementSessionRevokeRequest struct {
	Reason string `json:"reason"`
}

type ManagementSessionRevocation struct {
	SessionID string    `json:"sessionId"`
	Status    string    `json:"status"`
	RevokedAt time.Time `json:"revokedAt"`
	Changed   bool      `json:"changed"`
}

type PrincipalManagementSessionsRevocation struct {
	PrincipalID    string `json:"principalId"`
	RevokedCount   int    `json:"revokedCount"`
	AlreadyRevoked int    `json:"alreadyRevoked"`
}

// TrustedIdentityIssuer is Router-owned desired state. Verification keys are
// deliberately absent because they are bounded runtime cache entries.
type TrustedIdentityIssuer struct {
	IssuerID         string            `json:"issuerId"`
	Issuer           string            `json:"issuer"`
	Kind             string            `json:"kind"`
	DiscoveryURL     string            `json:"discoveryUrl,omitempty"`
	JWKSURL          string            `json:"jwksUrl,omitempty"`
	Audiences        []string          `json:"audiences"`
	ClaimMapping     map[string]string `json:"claimMapping"`
	AssuranceMapping map[string]string `json:"assuranceMapping"`
	Status           string            `json:"status"`
	Revision         uint64            `json:"revision"`
	CreatedAt        time.Time         `json:"createdAt"`
	UpdatedAt        time.Time         `json:"updatedAt"`
}

type TrustedIdentityIssuerPage struct {
	Data []TrustedIdentityIssuer `json:"data"`
	Page PageInfo                `json:"page"`
}

type TrustedIdentityIssuerCreateRequest struct {
	Issuer           string            `json:"issuer"`
	Kind             string            `json:"kind"`
	DiscoveryURL     string            `json:"discoveryUrl,omitempty"`
	JWKSURL          string            `json:"jwksUrl,omitempty"`
	Audiences        []string          `json:"audiences"`
	ClaimMapping     map[string]string `json:"claimMapping,omitempty"`
	AssuranceMapping map[string]string `json:"assuranceMapping,omitempty"`
}

// Issuer identity and kind are immutable. A different issuer is a distinct
// resource so existing principals can never be silently rebound.
type TrustedIdentityIssuerPatchRequest struct {
	DiscoveryURL     *string            `json:"discoveryUrl,omitempty"`
	JWKSURL          *string            `json:"jwksUrl,omitempty"`
	Audiences        *[]string          `json:"audiences,omitempty"`
	ClaimMapping     *map[string]string `json:"claimMapping,omitempty"`
	AssuranceMapping *map[string]string `json:"assuranceMapping,omitempty"`
	Status           *string            `json:"status,omitempty"`
	Reason           string             `json:"reason"`
}

type TrustedIdentityIssuerRefreshRequest struct {
	Reason string `json:"reason"`
}

type BackchannelLogoutRequest struct {
	IssuerID    string `json:"issuerId"`
	LogoutToken string `json:"logoutToken"`
}

type BackchannelLogoutResponse struct {
	Applied  bool `json:"applied"`
	Replayed bool `json:"replayed"`
}
