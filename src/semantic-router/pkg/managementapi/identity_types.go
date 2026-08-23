package managementapi

import "time"

type ExchangeChallengeRequest struct {
	IssuerID string `json:"issuerId"`
}

type ExchangeChallengeResponse struct {
	ExchangeChallengeID string    `json:"exchangeChallengeId"`
	Nonce               string    `json:"nonce"`
	ExpiresAt           time.Time `json:"expiresAt"`
}

type TokenExchangeRequest struct {
	IssuerID            string  `json:"issuerId"`
	ExchangeChallengeID string  `json:"exchangeChallengeId"`
	SubjectToken        string  `json:"subjectToken"`
	SubjectTokenType    string  `json:"subjectTokenType"`
	InvitationToken     *string `json:"invitationToken,omitempty"`
}

type TokenExchangeResponse struct {
	ManagementTokenEnvelope
	Onboarding *OnboardingResult `json:"onboarding,omitempty"`
}

type BootstrapRequest struct {
	Kind        string                     `json:"kind"`
	DisplayName string                     `json:"displayName"`
	External    *BootstrapExternalIdentity `json:"external,omitempty"`
}

type BootstrapExternalIdentity struct {
	IssuerID     string `json:"issuerId"`
	Issuer       string `json:"issuer"`
	Subject      string `json:"subject"`
	DiscoveryURL string `json:"discoveryUrl"`
	Audience     string `json:"audience"`
}

type BootstrapResponse struct {
	PrincipalID          string          `json:"principalId"`
	RoleBindingID        string          `json:"roleBindingId"`
	ServiceAccountID     string          `json:"serviceAccountId,omitempty"`
	ServiceCredential    *SecretEnvelope `json:"serviceCredential,omitempty"`
	FinalizationRequired bool            `json:"finalizationRequired"`
}

type RecoveryRequest struct {
	PrincipalID string `json:"principalId"`
	Reason      string `json:"reason"`
}

type RecoveryResponse struct {
	PrincipalID             string `json:"principalId"`
	RoleBindingID           string `json:"roleBindingId"`
	RecoveryDisableRequired bool   `json:"recoveryDisableRequired"`
}

type ManagementPrincipal struct {
	PrincipalID   string            `json:"principalId"`
	Issuer        string            `json:"issuer"`
	Subject       string            `json:"subject"`
	DisplayName   string            `json:"displayName"`
	VerifiedEmail string            `json:"verifiedEmail,omitempty"`
	Attributes    map[string]string `json:"attributes"`
	Status        string            `json:"status"`
	Revision      uint64            `json:"revision"`
	CreatedAt     time.Time         `json:"createdAt"`
	UpdatedAt     time.Time         `json:"updatedAt"`
}

type ManagementPrincipalCreateRequest struct {
	Issuer        string            `json:"issuer"`
	Subject       string            `json:"subject"`
	DisplayName   string            `json:"displayName"`
	VerifiedEmail string            `json:"verifiedEmail,omitempty"`
	Attributes    map[string]string `json:"attributes,omitempty"`
}

type ManagementPrincipalPatchRequest struct {
	DisplayName   *string `json:"displayName,omitempty"`
	VerifiedEmail *string `json:"verifiedEmail,omitempty"`
	Status        *string `json:"status,omitempty"`
	Reason        string  `json:"reason"`
}

type ManagementRole struct {
	RoleID      string    `json:"roleId"`
	NamespaceID string    `json:"namespaceId,omitempty"`
	Name        string    `json:"name"`
	DisplayName string    `json:"displayName"`
	Description string    `json:"description"`
	Permissions []string  `json:"permissions"`
	BuiltIn     bool      `json:"builtIn"`
	Status      string    `json:"status"`
	Revision    uint64    `json:"revision"`
	CreatedAt   time.Time `json:"createdAt"`
	UpdatedAt   time.Time `json:"updatedAt"`
}

type ManagementRoleCreateRequest struct {
	NamespaceID string   `json:"namespaceId"`
	Name        string   `json:"name"`
	DisplayName string   `json:"displayName"`
	Description string   `json:"description,omitempty"`
	Permissions []string `json:"permissions"`
}

type ManagementRolePatchRequest struct {
	DisplayName *string `json:"displayName,omitempty"`
	Description *string `json:"description,omitempty"`
	Reason      string  `json:"reason"`
}

type ManagementScope struct {
	Kind         string `json:"kind"`
	NamespaceID  string `json:"namespaceId,omitempty"`
	TeamID       string `json:"teamId,omitempty"`
	UserID       string `json:"userId,omitempty"`
	ResourceType string `json:"resourceType,omitempty"`
	ResourceID   string `json:"resourceId,omitempty"`
}

type ManagementRoleBinding struct {
	BindingID         string          `json:"bindingId"`
	PrincipalID       string          `json:"principalId"`
	RoleID            string          `json:"roleId"`
	Scope             ManagementScope `json:"scope"`
	DelegationCeiling []string        `json:"delegationCeiling"`
	Status            string          `json:"status"`
	Revision          uint64          `json:"revision"`
	CreatedAt         time.Time       `json:"createdAt"`
	UpdatedAt         time.Time       `json:"updatedAt"`
}

type ManagementRoleBindingCreateRequest struct {
	PrincipalID       string          `json:"principalId"`
	RoleID            string          `json:"roleId"`
	Scope             ManagementScope `json:"scope"`
	DelegationCeiling []string        `json:"delegationCeiling"`
}

type ManagementRoleBindingPatchRequest struct {
	Status string `json:"status"`
	Reason string `json:"reason"`
}

type AuthenticationRequirement struct {
	Kind     string               `json:"kind"`
	Human    *HumanRequirement    `json:"human,omitempty"`
	Workload *WorkloadRequirement `json:"workload,omitempty"`
}

type HumanRequirement struct {
	MinimumAAL                  string   `json:"minimumAal"`
	AcceptedAMR                 []string `json:"acceptedAmr"`
	MaxAuthenticationAgeSeconds int64    `json:"maxAuthenticationAgeSeconds"`
}

type WorkloadRequirement struct {
	MinimumWorkloadClass string `json:"minimumWorkloadClass"`
	MaxSourceAgeSeconds  int64  `json:"maxSourceAgeSeconds"`
}

type ManagementSessionPolicy struct {
	AccessTokenTTLSeconds int64                                  `json:"accessTokenTtlSeconds"`
	SessionTTLSeconds     int64                                  `json:"sessionTtlSeconds"`
	MaxActiveSessions     int                                    `json:"maxActiveSessions"`
	ActionRequirements    map[string][]AuthenticationRequirement `json:"actionRequirements"`
	SeedVersion           uint64                                 `json:"seedVersion"`
	Revision              uint64                                 `json:"revision"`
	UpdatedAt             time.Time                              `json:"updatedAt"`
}

type ManagementSessionPolicyPatchRequest struct {
	AccessTokenTTLSeconds int64                                  `json:"accessTokenTtlSeconds"`
	SessionTTLSeconds     int64                                  `json:"sessionTtlSeconds"`
	MaxActiveSessions     int                                    `json:"maxActiveSessions"`
	ActionRequirements    map[string][]AuthenticationRequirement `json:"actionRequirements"`
	Reason                string                                 `json:"reason"`
}
