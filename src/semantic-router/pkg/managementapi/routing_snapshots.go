package managementapi

import (
	"encoding/json"
	"time"
)

type RoutingSnapshotMetadata struct {
	NamespaceID     string     `json:"namespaceId"`
	RoutingRevision int64      `json:"routingRevision"`
	ContentDigest   string     `json:"contentDigest"`
	Status          string     `json:"status"`
	FailureReason   string     `json:"failureReason,omitempty"`
	MemberCount     int        `json:"memberCount"`
	CreatedAt       time.Time  `json:"createdAt"`
	ActivatedAt     *time.Time `json:"activatedAt,omitempty"`
}

type RoutingSnapshotMember struct {
	ResourceType     string `json:"resourceType"`
	ResourceID       string `json:"resourceId"`
	ResourceRevision int64  `json:"resourceRevision"`
}

// RoutingSnapshotBackendConnection is the immutable, non-secret connection
// shape captured by a published routing revision.
type RoutingSnapshotBackendConnection struct {
	Path    string            `json:"path"`
	Headers map[string]string `json:"headers,omitempty"`
}

type RoutingSnapshotBackend struct {
	ID                   string                           `json:"id"`
	ProviderID           string                           `json:"providerId"`
	WireFormat           string                           `json:"wireFormat"`
	Origin               string                           `json:"origin"`
	ProviderModelID      string                           `json:"providerModelId"`
	ProviderCredentialID string                           `json:"providerCredentialId,omitempty"`
	Connection           RoutingSnapshotBackendConnection `json:"connection"`
	Weight               string                           `json:"weight"`
}

// RoutingSnapshotModel is the public, immutable Model export. The Router may
// store a compiled execution value internally, but Management API consumers
// always see the same nested control shape used to author a Model.
type RoutingSnapshotModel struct {
	ID                string                   `json:"id"`
	Revision          int64                    `json:"revision"`
	CatalogRevision   string                   `json:"catalogRevision"`
	Name              string                   `json:"name"`
	Aliases           []string                 `json:"aliases,omitempty"`
	ParamSize         string                   `json:"paramSize,omitempty"`
	ContextWindowSize int                      `json:"contextWindowSize,omitempty"`
	Description       string                   `json:"description,omitempty"`
	Capabilities      []string                 `json:"capabilities,omitempty"`
	Reasoning         RoutingReasoningFamily   `json:"reasoning,omitempty"`
	LoRAs             []string                 `json:"loras,omitempty"`
	QualityScore      float64                  `json:"qualityScore,omitempty"`
	Modality          string                   `json:"modality,omitempty"`
	Tags              []string                 `json:"tags,omitempty"`
	Control           RoutingModelControl      `json:"control"`
	Pricing           RoutingPricing           `json:"pricing"`
	Backends          []RoutingSnapshotBackend `json:"backends"`
}

type RoutingSnapshotRecipe struct {
	ID          string            `json:"id"`
	Revision    int64             `json:"revision"`
	Name        string            `json:"name"`
	Description string            `json:"description,omitempty"`
	Decisions   []RoutingDecision `json:"decisions"`
	Document    json.RawMessage   `json:"document"`
}

type RoutingSnapshotEntrypoint struct {
	ID       string                  `json:"id"`
	Revision int64                   `json:"revision"`
	Name     string                  `json:"name"`
	Aliases  []string                `json:"aliases"`
	Rules    []RoutingEntrypointRule `json:"rules"`
}

// RoutingSnapshotExport is the versioned public projection of a compiled
// routing snapshot. Internal lookup indexes and compiled storage fields never
// cross the Management API boundary.
type RoutingSnapshotExport struct {
	NamespaceID string                      `json:"namespaceId"`
	Revision    int64                       `json:"revision"`
	Currency    string                      `json:"currency,omitempty"`
	Models      []RoutingSnapshotModel      `json:"models"`
	Recipes     []RoutingSnapshotRecipe     `json:"recipes"`
	Entrypoints []RoutingSnapshotEntrypoint `json:"entrypoints"`
	Digest      string                      `json:"digest"`
}

type RoutingSnapshotRecord struct {
	Metadata RoutingSnapshotMetadata `json:"metadata"`
	Members  []RoutingSnapshotMember `json:"members"`
	Export   RoutingSnapshotExport   `json:"export"`
}

type RoutingSnapshotPage = Page[RoutingSnapshotMetadata]

type RoutingSnapshotDetail struct {
	Data RoutingSnapshotRecord `json:"data"`
}
