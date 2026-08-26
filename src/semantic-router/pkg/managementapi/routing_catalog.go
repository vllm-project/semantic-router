package managementapi

// RoutingCatalog is the key-scoped, credential-free routing projection used
// by read-only clients. It deliberately cannot represent Provider backends,
// credentials, or Recipe source documents.
type RoutingCatalog struct {
	KeyID           string                     `json:"keyId"`
	PolicyRevision  uint64                     `json:"policyRevision"`
	PolicyDigest    string                     `json:"policyDigest"`
	RoutingRevision int64                      `json:"routingRevision"`
	RoutingDigest   string                     `json:"routingDigest"`
	Models          []RoutingCatalogModel      `json:"models"`
	Recipes         []RoutingCatalogRecipe     `json:"recipes"`
	Entrypoints     []RoutingCatalogEntrypoint `json:"entrypoints"`
}

type RoutingCatalogModel struct {
	ID                string                 `json:"id"`
	Revision          int64                  `json:"revision"`
	Name              string                 `json:"name"`
	Aliases           []string               `json:"aliases"`
	ParamSize         string                 `json:"paramSize,omitempty"`
	ContextWindowSize int                    `json:"contextWindowSize,omitempty"`
	Description       string                 `json:"description,omitempty"`
	Capabilities      []string               `json:"capabilities"`
	Reasoning         RoutingReasoningFamily `json:"reasoning,omitempty"`
	LoRAs             []string               `json:"loras"`
	QualityScore      float64                `json:"qualityScore,omitempty"`
	Modality          string                 `json:"modality,omitempty"`
	Tags              []string               `json:"tags"`
	Pricing           RoutingPricing         `json:"pricing"`
}

type RoutingCatalogRecipe struct {
	ID          string                     `json:"id"`
	Revision    int64                      `json:"revision"`
	Name        string                     `json:"name"`
	Description string                     `json:"description,omitempty"`
	Decisions   []RoutingDecision          `json:"decisions"`
	Signals     []RoutingCatalogSignal     `json:"signals"`
	Projections []RoutingCatalogProjection `json:"projections"`
}

type RoutingCatalogSignal struct {
	Type string `json:"type"`
	Name string `json:"name"`
}

type RoutingCatalogProjectionReference struct {
	Type   string `json:"type"`
	Name   string `json:"name,omitempty"`
	KB     string `json:"kb,omitempty"`
	Metric string `json:"metric,omitempty"`
}

type RoutingCatalogProjection struct {
	Type    string                              `json:"type"`
	Name    string                              `json:"name"`
	Members []string                            `json:"members"`
	Inputs  []RoutingCatalogProjectionReference `json:"inputs"`
	Source  string                              `json:"source,omitempty"`
	Outputs []string                            `json:"outputs"`
}

type RoutingCatalogEntrypoint struct {
	ID       string                  `json:"id"`
	Revision int64                   `json:"revision"`
	Name     string                  `json:"name"`
	Aliases  []string                `json:"aliases"`
	Rules    []RoutingEntrypointRule `json:"rules"`
}
