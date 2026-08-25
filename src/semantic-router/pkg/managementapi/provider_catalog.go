package managementapi

import (
	"encoding/json"
	"time"
)

type ProviderCatalogDisplay struct {
	Name        string              `json:"name"`
	Description string              `json:"description"`
	Category    string              `json:"category"`
	Icon        ProviderCatalogIcon `json:"icon"`
	Monogram    string              `json:"monogram,omitempty"`
	Accent      string              `json:"accent,omitempty"`
}

type ProviderCatalogIcon struct {
	Source string `json:"source"`
	Value  string `json:"value"`
	Color  bool   `json:"color"`
}

type ProviderCredentialPrompt struct {
	Mode  string `json:"mode"`
	Label string `json:"label,omitempty"`
	Hint  string `json:"hint,omitempty"`
}

type ProviderOriginPrompt struct {
	Mode            string `json:"mode"`
	DefaultURL      string `json:"defaultUrl,omitempty"`
	BaseURLRequired bool   `json:"baseUrlRequired"`
	Label           string `json:"label,omitempty"`
	Hint            string `json:"hint,omitempty"`
}

type ProviderConnectionField struct {
	Name        string                `json:"name"`
	Label       string                `json:"label"`
	Kind        string                `json:"kind"`
	Required    bool                  `json:"required"`
	Advanced    bool                  `json:"advanced"`
	Default     string                `json:"default,omitempty"`
	Hint        string                `json:"hint,omitempty"`
	Placeholder string                `json:"placeholder,omitempty"`
	Options     []ProviderFieldOption `json:"options,omitempty"`
}

type ProviderFieldOption struct {
	Value string `json:"value"`
	Label string `json:"label"`
}

type ProviderInterface struct {
	ID      string `json:"id"`
	Label   string `json:"label"`
	Default bool   `json:"default"`
	// Capabilities are features carried by this transport interface, not
	// model-specific capability claims.
	Capabilities []string `json:"capabilities"`
}

// ProviderCatalogItem is the safe Management view. Integration-only invocation
// headers/paths and internal credential/discovery adapter IDs are deliberately
// absent; clients receive the schema they need, not executor internals.
type ProviderCatalogItem struct {
	ProviderID         string                   `json:"providerId"`
	Revision           string                   `json:"revision"`
	Display            ProviderCatalogDisplay   `json:"display"`
	Credential         ProviderCredentialPrompt `json:"credential"`
	Origin             ProviderOriginPrompt     `json:"origin"`
	DiscoverySupported bool                     `json:"discoverySupported"`
	// Capabilities are Provider transport features used for catalog filtering.
	// They are not inherited by discovered Models.
	Capabilities     []string                  `json:"capabilities"`
	ConnectionFields []ProviderConnectionField `json:"connectionFields"`
	Interfaces       []ProviderInterface       `json:"interfaces"`
}

type ProviderCatalogPage struct {
	Data            []ProviderCatalogItem `json:"data"`
	Page            PageInfo              `json:"page"`
	CatalogRevision string                `json:"catalogRevision"`
	Categories      []string              `json:"categories"`
}

type ProviderCatalogDetail struct {
	Data            ProviderCatalogItem `json:"data"`
	CatalogRevision string              `json:"catalogRevision"`
}

type DiscoverModelsRequest struct {
	CredentialID     string                     `json:"credentialId,omitempty"`
	BaseURL          string                     `json:"baseUrl,omitempty"`
	ConnectionFields map[string]json.RawMessage `json:"connectionFields,omitempty"`
	Search           string                     `json:"search,omitempty"`
	PageSize         int                        `json:"pageSize,omitempty"`
	Cursor           string                     `json:"cursor,omitempty"`
}

type DiscoveredModel struct {
	CatalogItemID   string `json:"catalogItemId"`
	ProviderModelID string `json:"providerModelId"`
	DisplayName     string `json:"displayName"`
	// Capabilities is present only when discovery has model-specific evidence.
	Capabilities []string `json:"capabilities,omitempty"`
}

type DiscoverModelsPage struct {
	Data              []DiscoveredModel `json:"data"`
	Page              PageInfo          `json:"page"`
	CatalogRevision   string            `json:"catalogRevision"`
	DiscoveryRevision string            `json:"discoveryRevision"`
	ExpiresAt         time.Time         `json:"expiresAt"`
}
