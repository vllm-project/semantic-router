package managementidentity

import (
	"context"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth/issuerverifier"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
)

type SelfView struct {
	Principal          Principal
	Session            ManagementSession
	ClusterPermissions []string
	Namespaces         []SelfNamespace
}

type SelfNamespace struct {
	ID                string
	Name              string
	Status            string
	DesiredRevision   uint64
	AppliedRevision   uint64
	Permissions       []string
	RoleBindings      []RoleBinding
	User              *SelfUser
	Teams             []SelfTeamMembership
	SelfServicePolicy SelfServicePolicy
}

type SelfUser struct {
	ID          string
	Email       string
	DisplayName string
	Status      string
}

type SelfTeamMembership struct {
	TeamID string
	Name   string
	Role   string
	Status string
}

type SelfServicePolicy struct {
	MaxKeysPerUser             int
	MaxDelegatedSessions       int
	DelegatedSessionTTLSeconds int64
	AllowTeamKeyDelegation     bool
	AutomaticFirstKey          bool
	Revision                   uint64
}

type ManagementSession struct {
	ID              string
	PrincipalID     string
	AuthSourceKind  managementauth.AuthSourceKind
	EvidenceKind    managementauth.EvidenceKind
	AuthenticatedAt time.Time
	ExpiresAt       time.Time
	Status          managementauth.SessionStatus
	RevokedAt       *time.Time
	CreatedAt       time.Time
}

type ManagementSessionPage struct {
	Items      []ManagementSession
	NextCursor string
}

type TrustedIdentityIssuer struct {
	ID               string
	Issuer           string
	Kind             issuerverifier.IssuerKind
	DiscoveryURL     string
	JWKSURL          string
	Audiences        []string
	ClaimMapping     map[string]string
	AssuranceMapping map[string]string
	Status           managementauth.ResourceStatus
	Revision         uint64
	CreatedAt        time.Time
	UpdatedAt        time.Time
}

func (issuer TrustedIdentityIssuer) VerificationValue() issuerverifier.TrustedIssuer {
	return issuerverifier.TrustedIssuer{
		ID: issuer.ID, Issuer: issuer.Issuer, Kind: issuer.Kind,
		DiscoveryURL: issuer.DiscoveryURL, JWKSURL: issuer.JWKSURL,
		Audiences:        append([]string(nil), issuer.Audiences...),
		ClaimMapping:     cloneStringMap(issuer.ClaimMapping),
		AssuranceMapping: cloneStringMap(issuer.AssuranceMapping),
		Revision:         issuer.Revision,
	}
}

type TrustedIdentityIssuerPage struct {
	Items      []TrustedIdentityIssuer
	NextCursor string
}

type CreateTrustedIdentityIssuer struct {
	Issuer  TrustedIdentityIssuer
	Command managementcommand.Command
	Actor   MutationActor
}

type UpdateTrustedIdentityIssuer struct {
	ID               string
	ExpectedRevision uint64
	DiscoveryURL     *string
	JWKSURL          *string
	Audiences        *[]string
	ClaimMapping     *map[string]string
	AssuranceMapping *map[string]string
	Status           *managementauth.ResourceStatus
	Actor            MutationActor
}

type RefreshTrustedIdentityIssuer struct {
	ID      string
	Command managementcommand.Command
	Actor   MutationActor
}

type IssuerMutation struct {
	Result   MutationResult
	Issuer   TrustedIdentityIssuer
	Sessions []string
}

type SessionRevocationCommand struct {
	SessionID string
	Command   managementcommand.Command
	Actor     MutationActor
}

type PrincipalSessionRevocationCommand struct {
	PrincipalID string
	Command     managementcommand.Command
	Actor       MutationActor
}

type PrincipalSessionRevocation struct {
	Result         MutationResult
	SessionIDs     []string
	RevokedCount   int
	AlreadyRevoked int
}

type BackchannelLogout struct {
	Identity  managementauth.BackchannelLogoutIdentity
	RequestID string
}

type BackchannelLogoutResult struct {
	SessionIDs []string
	Replayed   bool
}

type IssuerKeyCache interface {
	Invalidate(string)
	Refresh(context.Context, issuerverifier.TrustedIssuer) error
}

func cloneStringMap(source map[string]string) map[string]string {
	result := make(map[string]string, len(source))
	for key, value := range source {
		result[key] = value
	}
	return result
}
