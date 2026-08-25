package managementapi

import "time"

type InvitationExpectedIdentity struct {
	Issuer  string `json:"issuer"`
	Subject string `json:"subject,omitempty"`
	Email   string `json:"email,omitempty"`
}

type InvitationRoleGrantRequest struct {
	RoleID            string   `json:"roleId"`
	ScopeKind         string   `json:"scopeKind"`
	DelegationCeiling []string `json:"delegationCeiling,omitempty"`
}

type InvitationTeamAssignment struct {
	TeamID string `json:"teamId"`
	Role   string `json:"role"`
}

type InvitationCreateRequest struct {
	ExpectedIdentity InvitationExpectedIdentity   `json:"expectedIdentity"`
	DisplayName      string                       `json:"displayName"`
	RoleGrants       []InvitationRoleGrantRequest `json:"roleGrants"`
	Team             *InvitationTeamAssignment    `json:"team,omitempty"`
	ExpiresAt        time.Time                    `json:"expiresAt"`
}

type InvitationRotateTokenRequest struct {
	ExpiresAt *time.Time `json:"expiresAt,omitempty"`
}

type OnboardingCreateRequest struct {
	PrincipalID    string                       `json:"principalId"`
	Email          string                       `json:"email"`
	DisplayName    string                       `json:"displayName"`
	RoleGrants     []InvitationRoleGrantRequest `json:"roleGrants"`
	Team           *InvitationTeamAssignment    `json:"team,omitempty"`
	CreateFirstKey bool                         `json:"createFirstKey"`
}

type InvitationRoleGrant struct {
	RoleID                  string   `json:"roleId"`
	RoleRevision            uint64   `json:"roleRevision"`
	RolePermissionsDigest   string   `json:"rolePermissionsDigest"`
	ScopeKind               string   `json:"scopeKind"`
	DelegationCeiling       []string `json:"delegationCeiling"`
	SourceBindingID         string   `json:"sourceBindingId"`
	SourceBindingRevision   uint64   `json:"sourceBindingRevision"`
	SourceRoleID            string   `json:"sourceRoleId"`
	SourcePermissionsDigest string   `json:"sourcePermissionsDigest"`
}

type InvitationOnboardingSnapshot struct {
	RoleGrants                []InvitationRoleGrant     `json:"roleGrants"`
	Team                      *InvitationTeamAssignment `json:"team,omitempty"`
	SelfServicePolicyRevision uint64                    `json:"selfServicePolicyRevision"`
	AccessPolicyID            string                    `json:"accessPolicyId,omitempty"`
	AccessPolicyRevision      uint64                    `json:"accessPolicyRevision,omitempty"`
	RateLimitPolicyID         string                    `json:"rateLimitPolicyId,omitempty"`
	RateLimitPolicyRevision   uint64                    `json:"rateLimitPolicyRevision,omitempty"`
	AutomaticFirstKey         bool                      `json:"automaticFirstKey"`
}

type Invitation struct {
	InvitationID                string                       `json:"invitationId"`
	NamespaceID                 string                       `json:"namespaceId"`
	CreatedByPrincipalID        string                       `json:"createdByPrincipalId"`
	ExpectedIdentity            InvitationExpectedIdentity   `json:"expectedIdentity"`
	DisplayName                 string                       `json:"displayName"`
	Onboarding                  InvitationOnboardingSnapshot `json:"onboarding"`
	ExpiresAt                   time.Time                    `json:"expiresAt"`
	Status                      string                       `json:"status"`
	AcceptedPrincipalID         string                       `json:"acceptedPrincipalId,omitempty"`
	AcceptedUserID              string                       `json:"acceptedUserId,omitempty"`
	AcceptedManagementSessionID string                       `json:"acceptedManagementSessionId,omitempty"`
	AcceptedAt                  *time.Time                   `json:"acceptedAt,omitempty"`
	Revision                    uint64                       `json:"revision"`
	CreatedAt                   time.Time                    `json:"createdAt"`
	UpdatedAt                   time.Time                    `json:"updatedAt"`
}

type InvitationPage struct {
	Data []Invitation `json:"data"`
	Page PageInfo     `json:"page"`
}

type InvitationDetail struct {
	Data Invitation `json:"data"`
}

type InvitationIssuedSecret struct {
	Data              Invitation `json:"data"`
	Token             string     `json:"token"`
	DeliveryExpiresAt time.Time  `json:"deliveryExpiresAt"`
}

type OnboardingResult struct {
	InvitationID      string    `json:"invitationId,omitempty"`
	PrincipalID       string    `json:"principalId"`
	UserID            string    `json:"userId"`
	TeamID            string    `json:"teamId,omitempty"`
	APIKeyID          string    `json:"apiKeyId,omitempty"`
	APIKey            string    `json:"apiKey,omitempty"`
	DeliveryExpiresAt time.Time `json:"deliveryExpiresAt"`
}
