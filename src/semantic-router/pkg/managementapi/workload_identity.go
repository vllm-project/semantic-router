package managementapi

import "time"

type ServiceAccount struct {
	ServiceAccountID string    `json:"serviceAccountId"`
	PrincipalID      string    `json:"principalId"`
	DisplayName      string    `json:"displayName"`
	OwnerScope       string    `json:"ownerScope"`
	NamespaceID      string    `json:"namespaceId,omitempty"`
	Status           string    `json:"status"`
	Revision         uint64    `json:"revision"`
	CreatedAt        time.Time `json:"createdAt"`
	UpdatedAt        time.Time `json:"updatedAt"`
}

type ServiceCredential struct {
	CredentialID     string     `json:"credentialId"`
	ServiceAccountID string     `json:"serviceAccountId"`
	PublicID         string     `json:"publicId"`
	WorkloadClass    string     `json:"workloadClass"`
	SourceAssuredAt  time.Time  `json:"sourceAssuredAt"`
	Status           string     `json:"status"`
	NotBefore        time.Time  `json:"notBefore"`
	ExpiresAt        time.Time  `json:"expiresAt"`
	RevokedAt        *time.Time `json:"revokedAt,omitempty"`
	CreatedAt        time.Time  `json:"createdAt"`
}

type ServiceAccountCreateRequest struct {
	DisplayName         string    `json:"displayName"`
	OwnerScope          string    `json:"ownerScope"`
	NamespaceID         string    `json:"namespaceId,omitempty"`
	CredentialExpiresAt time.Time `json:"credentialExpiresAt"`
	CredentialClass     string    `json:"credentialClass"`
	Reason              string    `json:"reason"`
}

type ServiceAccountPatchRequest struct {
	DisplayName *string `json:"displayName,omitempty"`
	Status      *string `json:"status,omitempty"`
	Reason      string  `json:"reason"`
}

type ServiceCredentialRotateRequest struct {
	ExpiresAt      time.Time `json:"expiresAt"`
	WorkloadClass  string    `json:"workloadClass"`
	OverlapSeconds int64     `json:"overlapSeconds"`
	Reason         string    `json:"reason"`
}

type ServiceCredentialRevokeRequest struct {
	Reason string `json:"reason"`
}

type ServiceCredentialIssue struct {
	ServiceAccount    ServiceAccount    `json:"serviceAccount"`
	Credential        ServiceCredential `json:"credential"`
	Secret            string            `json:"secret"`
	DeliveryExpiresAt time.Time         `json:"deliveryExpiresAt"`
}

type MTLSIdentityMapping struct {
	MappingID       string    `json:"mappingId"`
	MatcherKind     string    `json:"matcherKind"`
	MatcherValue    string    `json:"matcherValue"`
	PrincipalID     string    `json:"principalId"`
	WorkloadClass   string    `json:"workloadClass"`
	SourceAssuredAt time.Time `json:"sourceAssuredAt"`
	Status          string    `json:"status"`
	Revision        uint64    `json:"revision"`
	CreatedAt       time.Time `json:"createdAt"`
	UpdatedAt       time.Time `json:"updatedAt"`
}

type MTLSIdentityMappingCreateRequest struct {
	MatcherKind   string `json:"matcherKind"`
	MatcherValue  string `json:"matcherValue"`
	PrincipalID   string `json:"principalId"`
	WorkloadClass string `json:"workloadClass"`
	Reason        string `json:"reason"`
}

type MTLSIdentityMappingPatchRequest struct {
	Status        *string `json:"status,omitempty"`
	WorkloadClass *string `json:"workloadClass,omitempty"`
	Reason        string  `json:"reason"`
}

type MTLSIdentityMappingDeleteRequest struct {
	Reason string `json:"reason"`
}
