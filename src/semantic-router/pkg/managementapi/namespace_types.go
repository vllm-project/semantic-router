package managementapi

import "time"

type Namespace struct {
	NamespaceID      string    `json:"namespaceId"`
	Name             string    `json:"name"`
	QuotaPartitionID string    `json:"quotaPartitionId"`
	BillingCurrency  string    `json:"billingCurrency"`
	Status           string    `json:"status"`
	Revision         uint64    `json:"revision"`
	RuntimeEpoch     uint64    `json:"runtimeEpoch"`
	CreatedAt        time.Time `json:"createdAt"`
	UpdatedAt        time.Time `json:"updatedAt"`
}

type NamespacePage = Page[Namespace]

type NamespaceDetail struct {
	Data Namespace `json:"data"`
}

type NamespaceCreateRequest struct {
	Name            string `json:"name"`
	BillingCurrency string `json:"billingCurrency"`
	Reason          string `json:"reason"`
}

type NamespacePatchRequest struct {
	Status string `json:"status"`
	Reason string `json:"reason"`
}

type SelfServicePolicy struct {
	NamespaceID                string    `json:"namespaceId"`
	MaxKeysPerUser             int       `json:"maxKeysPerUser"`
	MaxDelegatedSessions       int       `json:"maxDelegatedSessions"`
	DelegatedSessionTTLSeconds int64     `json:"delegatedSessionTtlSeconds"`
	AllowTeamKeyDelegation     bool      `json:"allowTeamKeyDelegation"`
	AutomaticFirstKey          bool      `json:"automaticFirstKey"`
	TeamAdminCapabilities      []string  `json:"teamAdminCapabilities"`
	DefaultAccessPolicyID      string    `json:"defaultAccessPolicyId,omitempty"`
	DefaultRateLimitPolicyID   string    `json:"defaultRateLimitPolicyId,omitempty"`
	Revision                   uint64    `json:"revision"`
	SeedVersion                uint64    `json:"seedVersion"`
	UpdatedAt                  time.Time `json:"updatedAt"`
}

type SelfServicePolicyDetail struct {
	Data SelfServicePolicy `json:"data"`
}

type SelfServicePolicyPatchRequest struct {
	MaxKeysPerUser             *int      `json:"maxKeysPerUser,omitempty"`
	MaxDelegatedSessions       *int      `json:"maxDelegatedSessions,omitempty"`
	DelegatedSessionTTLSeconds *int64    `json:"delegatedSessionTtlSeconds,omitempty"`
	AllowTeamKeyDelegation     *bool     `json:"allowTeamKeyDelegation,omitempty"`
	AutomaticFirstKey          *bool     `json:"automaticFirstKey,omitempty"`
	TeamAdminCapabilities      *[]string `json:"teamAdminCapabilities,omitempty"`
	DefaultAccessPolicyID      *string   `json:"defaultAccessPolicyId,omitempty"`
	DefaultRateLimitPolicyID   *string   `json:"defaultRateLimitPolicyId,omitempty"`
	Reason                     string    `json:"reason"`
}

type NamespaceManagementSecurityPolicy struct {
	NamespaceID        string                                 `json:"namespaceId"`
	ActionRequirements map[string][]AuthenticationRequirement `json:"actionRequirements"`
	SeedVersion        uint64                                 `json:"seedVersion"`
	Revision           uint64                                 `json:"revision"`
	UpdatedAt          time.Time                              `json:"updatedAt"`
}

type NamespaceManagementSecurityPolicyDetail struct {
	Data NamespaceManagementSecurityPolicy `json:"data"`
}

type NamespaceManagementSecurityPolicyPatchRequest struct {
	ActionRequirements map[string][]AuthenticationRequirement `json:"actionRequirements"`
	Reason             string                                 `json:"reason"`
}

type RoutingClaimDefinition struct {
	Kind      string `json:"kind"`
	Minimum   *int64 `json:"minimum,omitempty"`
	Maximum   *int64 `json:"maximum,omitempty"`
	MaxLength *int64 `json:"maxLength,omitempty"`
}

type RoutingClaimSchema struct {
	NamespaceID string                            `json:"namespaceId"`
	Definitions map[string]RoutingClaimDefinition `json:"definitions"`
	Revision    uint64                            `json:"revision"`
	UpdatedAt   time.Time                         `json:"updatedAt"`
}

type RoutingClaimSchemaDetail struct {
	Data RoutingClaimSchema `json:"data"`
}

type RoutingClaimSchemaPatchRequest struct {
	Definitions map[string]RoutingClaimDefinition `json:"definitions"`
	Reason      string                            `json:"reason"`
}
