package managementapi

import "time"

// ProviderCredentialCreateRequest is write-only with respect to Secret. The
// response types below have no field capable of carrying submitted material.
type ProviderCredentialCreateRequest struct {
	Name       string `json:"name"`
	ProviderID string `json:"providerId"`
	BaseURL    string `json:"baseUrl,omitempty"`
	Secret     string `json:"secret"`
}

// ProviderCredentialPatchRequest accepts exactly one of rename, disable, or
// reactivate. Reactivation requires fresh secret material because disablement
// cryptographically erases every prior version.
type ProviderCredentialPatchRequest struct {
	Name   *string `json:"name,omitempty"`
	Status *string `json:"status,omitempty"`
	Secret *string `json:"secret,omitempty"`
}

type ProviderCredentialRotateRequest struct {
	Secret string `json:"secret"`
}

type ProviderCredential struct {
	CredentialID     string     `json:"credentialId"`
	Name             string     `json:"name"`
	ProviderID       string     `json:"providerId"`
	CatalogRevision  string     `json:"catalogRevision"`
	NormalizedOrigin string     `json:"normalizedOrigin"`
	Status           string     `json:"status"`
	Revision         uint64     `json:"revision"`
	CreatedAt        time.Time  `json:"createdAt"`
	UpdatedAt        time.Time  `json:"updatedAt"`
	DeletedAt        *time.Time `json:"deletedAt,omitempty"`
}

type ProviderCredentialPage struct {
	Data []ProviderCredential `json:"data"`
	Page PageInfo             `json:"page"`
}

type ProviderCredentialDetail struct {
	Data ProviderCredential `json:"data"`
}
