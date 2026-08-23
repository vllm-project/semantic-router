package managementapi

import "time"

// APIKeyOwner is a discriminated owner reference. Exactly one User or Team is
// selected by Type; accepting two nullable identifiers would make ownership
// and authorization ambiguous.
type APIKeyOwner struct {
	Type string `json:"type"`
	ID   string `json:"id"`
}

type APIKeyCreateRequest struct {
	Name              string                   `json:"name"`
	Owner             APIKeyOwner              `json:"owner"`
	ContextTeamID     string                   `json:"contextTeamId,omitempty"`
	ExpiresAt         *time.Time               `json:"expiresAt,omitempty"`
	Revealable        *bool                    `json:"revealable,omitempty"`
	AccessPolicyIDs   []string                 `json:"accessPolicyIds,omitempty"`
	RateLimitOverride *APIKeyRateLimitOverride `json:"rateLimitOverride,omitempty"`
}

type APIKeyInlineRateLimitPolicy struct {
	Name        string               `json:"name"`
	Description string               `json:"description,omitempty"`
	Rules       []RateLimitRuleInput `json:"rules"`
}

type APIKeyRateLimitOverride struct {
	PolicyID     string                       `json:"policyId,omitempty"`
	InlinePolicy *APIKeyInlineRateLimitPolicy `json:"inlinePolicy,omitempty"`
}

type APIKeyPatchRequest struct {
	Name string `json:"name"`
}

type APIKeyLifecycleRequest struct{}

type APIKeyRenewRequest struct {
	ExpiresAt *time.Time `json:"expiresAt"`
}

type APIKeyReassignRequest struct {
	Owner         APIKeyOwner `json:"owner"`
	ContextTeamID string      `json:"contextTeamId,omitempty"`
}

type APIKeyRotateRequest struct {
	OverlapSeconds int64 `json:"overlapSeconds"`
	Revealable     *bool `json:"revealable,omitempty"`
}

type APIKey struct {
	KeyID         string      `json:"keyId"`
	Name          string      `json:"name"`
	Owner         APIKeyOwner `json:"owner"`
	ContextTeamID string      `json:"contextTeamId,omitempty"`
	Status        string      `json:"status"`
	ExpiresAt     *time.Time  `json:"expiresAt,omitempty"`
	LastUsedAt    *time.Time  `json:"lastUsedAt,omitempty"`
	Revision      uint64      `json:"revision"`
	CreatedAt     time.Time   `json:"createdAt"`
	UpdatedAt     time.Time   `json:"updatedAt"`
	DeletedAt     *time.Time  `json:"deletedAt,omitempty"`
}

type APIKeyPage struct {
	Data []APIKey `json:"data"`
	Page PageInfo `json:"page"`
}

type APIKeyDetail struct {
	Data APIKey `json:"data"`
}

type APIKeyCredential struct {
	CredentialID string     `json:"credentialId"`
	KeyID        string     `json:"keyId"`
	KID          string     `json:"kid"`
	Status       string     `json:"status"`
	Revealable   bool       `json:"revealable"`
	NotBefore    time.Time  `json:"notBefore"`
	ExpiresAt    *time.Time `json:"expiresAt,omitempty"`
	RevokedAt    *time.Time `json:"revokedAt,omitempty"`
	CreatedAt    time.Time  `json:"createdAt"`
}

type APIKeyCredentialPage struct {
	Data []APIKeyCredential `json:"data"`
	Page PageInfo           `json:"page"`
}

type APIKeyRevealResponse struct {
	KeyID        string `json:"keyId"`
	CredentialID string `json:"credentialId"`
	Secret       string `json:"secret"`
}
