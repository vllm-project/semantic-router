package managementapi

import "time"

// Me is the complete navigation and self-service authorization view for the
// current Management session. Clients may render from this response, but must
// never treat it as an authorization decision for a later mutation.
type Me struct {
	Principal          MePrincipal        `json:"principal"`
	Session            MeSession          `json:"session"`
	ClusterPermissions []string           `json:"clusterPermissions"`
	Namespaces         []MeNamespaceScope `json:"namespaces"`
}

type MePrincipal struct {
	PrincipalID string `json:"principalId"`
	DisplayName string `json:"displayName"`
	Kind        string `json:"kind"`
	Status      string `json:"status"`
}

type MeSession struct {
	SessionID       string    `json:"sessionId"`
	AuthenticatedAt time.Time `json:"authenticatedAt"`
	ExpiresAt       time.Time `json:"expiresAt"`
	EvidenceKind    string    `json:"evidenceKind"`
}

type MeNamespaceScope struct {
	Namespace         MeNamespace             `json:"namespace"`
	Permissions       []string                `json:"permissions"`
	RoleBindings      []ManagementRoleBinding `json:"roleBindings"`
	User              *MeUser                 `json:"user,omitempty"`
	Teams             []MeTeamMembership      `json:"teams"`
	SelfServicePolicy MeSelfServicePolicy     `json:"selfServicePolicy"`
}

type MeNamespace struct {
	NamespaceID     string `json:"namespaceId"`
	Name            string `json:"name"`
	Status          string `json:"status"`
	DesiredRevision uint64 `json:"desiredRevision"`
	AppliedRevision uint64 `json:"appliedRevision"`
}

type MeUser struct {
	UserID      string `json:"userId"`
	Email       string `json:"email"`
	DisplayName string `json:"displayName"`
	Status      string `json:"status"`
}

type MeTeamMembership struct {
	TeamID string `json:"teamId"`
	Name   string `json:"name"`
	Role   string `json:"role"`
	Status string `json:"status"`
}

type MeSelfServicePolicy struct {
	MaxKeysPerUser             int    `json:"maxKeysPerUser"`
	MaxDelegatedSessions       int    `json:"maxDelegatedSessions"`
	DelegatedSessionTTLSeconds int64  `json:"delegatedSessionTtlSeconds"`
	AllowTeamKeyDelegation     bool   `json:"allowTeamKeyDelegation"`
	AutomaticFirstKey          bool   `json:"automaticFirstKey"`
	Revision                   uint64 `json:"revision"`
}

// EligibleInferenceKey is the deliberately narrow logical-key view returned
// to a linked User. It does not confer general key-read permission and never
// includes credential metadata or reveal state.
type EligibleInferenceKey struct {
	KeyID         string      `json:"keyId"`
	Name          string      `json:"name"`
	Owner         APIKeyOwner `json:"owner"`
	ContextTeamID string      `json:"contextTeamId,omitempty"`
	ExpiresAt     *time.Time  `json:"expiresAt,omitempty"`
}

type EligibleInferenceKeyPage struct {
	Data []EligibleInferenceKey `json:"data"`
	Page PageInfo               `json:"page"`
}

type DelegatedInferenceSessionCreateRequest struct {
	KeyID string `json:"keyId"`
}

type DelegatedInferenceSession struct {
	SessionID string    `json:"sessionId"`
	PublicID  string    `json:"publicId"`
	KeyID     string    `json:"keyId"`
	UserID    string    `json:"userId"`
	TeamID    string    `json:"teamId,omitempty"`
	Audience  string    `json:"audience"`
	Status    string    `json:"status"`
	NotBefore time.Time `json:"notBefore"`
	ExpiresAt time.Time `json:"expiresAt"`
	CreatedAt time.Time `json:"createdAt"`
}

type DelegatedInferenceSessionPage struct {
	Data []DelegatedInferenceSession `json:"data"`
	Page PageInfo                    `json:"page"`
}
