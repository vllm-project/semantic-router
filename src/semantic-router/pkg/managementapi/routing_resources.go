package managementapi

import (
	"encoding/json"
	"time"
)

type RoutingReasoningFamily struct {
	Type    string   `json:"type,omitempty"`
	Efforts []string `json:"efforts,omitempty"`
}

type RoutingDecision struct {
	ID                  string `json:"id"`
	Name                string `json:"name"`
	DispatchCardinality string `json:"dispatchCardinality"`
}

type RoutingAssignmentReasoning struct {
	Enabled     bool   `json:"enabled"`
	Effort      string `json:"effort,omitempty"`
	Description string `json:"description,omitempty"`
}

type RoutingClaimMatcher struct {
	Name  string            `json:"name"`
	Value RoutingClaimValue `json:"value"`
}

type RoutingMatcher struct {
	Claim      *RoutingClaimMatcher `json:"claim,omitempty"`
	ExactPath  string               `json:"exactPath,omitempty"`
	PathPrefix string               `json:"pathPrefix,omitempty"`
}

type RoutingAssignment struct {
	ModelID       string                      `json:"modelId"`
	ModelRevision int64                       `json:"modelRevision"`
	Priority      int                         `json:"priority"`
	Weight        string                      `json:"weight"`
	LoRAName      string                      `json:"loraName,omitempty"`
	Reasoning     *RoutingAssignmentReasoning `json:"reasoning,omitempty"`
}

type RoutingFallbackPolicy struct {
	Strategy string   `json:"strategy"`
	On       []string `json:"on"`
}

type RoutingAssignmentSet struct {
	Models   []RoutingAssignment    `json:"models"`
	Fallback *RoutingFallbackPolicy `json:"fallback,omitempty"`
}

type RoutingEntrypointRule struct {
	ID             string                          `json:"id"`
	Name           string                          `json:"name"`
	Matchers       []RoutingMatcher                `json:"matchers,omitempty"`
	RecipeID       string                          `json:"recipeId"`
	RecipeRevision int64                           `json:"recipeRevision"`
	Assignments    map[string]RoutingAssignmentSet `json:"assignments"`
}

type RoutingResolvedRecipe struct {
	ID        string            `json:"id"`
	Revision  int64             `json:"revision"`
	Name      string            `json:"name"`
	Decisions []RoutingDecision `json:"decisions"`
	Document  json.RawMessage   `json:"document"`
}

type RoutingModelRetryControl struct {
	Count int      `json:"count"`
	On    []string `json:"on,omitempty"`
}

type RoutingModelTimeoutControl struct {
	Request string `json:"request"`
	Stream  string `json:"stream"`
}

// RoutingModelControl is the Management API form of providers.models[].control.
// The immutable routing snapshot keeps a flattened execution value internally;
// that transport detail is not a second public authoring contract.
type RoutingModelControl struct {
	Retry   RoutingModelRetryControl   `json:"retry"`
	Timeout RoutingModelTimeoutControl `json:"timeout"`
}

type RoutingPricing struct {
	InputCostPerMillionTokens      *string `json:"inputCostPerMillionTokens"`
	OutputCostPerMillionTokens     *string `json:"outputCostPerMillionTokens"`
	CacheReadCostPerMillionTokens  *string `json:"cacheReadCostPerMillionTokens"`
	CacheWriteCostPerMillionTokens *string `json:"cacheWriteCostPerMillionTokens"`
}

type RoutingModelBackendInput struct {
	ProviderID       string                     `json:"providerId"`
	InterfaceID      string                     `json:"interfaceId,omitempty"`
	ProviderModelID  string                     `json:"providerModelId"`
	CredentialID     string                     `json:"credentialId,omitempty"`
	BaseURL          string                     `json:"baseUrl,omitempty"`
	ConnectionFields map[string]json.RawMessage `json:"connectionFields,omitempty"`
	Weight           string                     `json:"weight,omitempty"`
}

type RoutingModelWrite struct {
	ID                string                     `json:"id,omitempty"`
	Name              string                     `json:"name"`
	Aliases           []string                   `json:"aliases,omitempty"`
	ParamSize         string                     `json:"paramSize,omitempty"`
	ContextWindowSize int                        `json:"contextWindowSize,omitempty"`
	Description       string                     `json:"description,omitempty"`
	Capabilities      []string                   `json:"capabilities,omitempty"`
	Reasoning         RoutingReasoningFamily     `json:"reasoning,omitempty"`
	LoRAs             []string                   `json:"loras,omitempty"`
	QualityScore      float64                    `json:"qualityScore,omitempty"`
	Modality          string                     `json:"modality,omitempty"`
	Tags              []string                   `json:"tags,omitempty"`
	Control           RoutingModelControl        `json:"control"`
	Pricing           RoutingPricing             `json:"pricing"`
	Backends          []RoutingModelBackendInput `json:"backends"`
}

// RoutingModelPatch changes only the fields present in the request. In
// particular, callers can tune control or pricing without reading or
// resubmitting credential-bearing backend configuration.
type RoutingModelPatch struct {
	Name              *string                     `json:"name,omitempty"`
	Aliases           *[]string                   `json:"aliases,omitempty"`
	ParamSize         *string                     `json:"paramSize,omitempty"`
	ContextWindowSize *int                        `json:"contextWindowSize,omitempty"`
	Description       *string                     `json:"description,omitempty"`
	Capabilities      *[]string                   `json:"capabilities,omitempty"`
	Reasoning         *RoutingReasoningFamily     `json:"reasoning,omitempty"`
	LoRAs             *[]string                   `json:"loras,omitempty"`
	QualityScore      *float64                    `json:"qualityScore,omitempty"`
	Modality          *string                     `json:"modality,omitempty"`
	Tags              *[]string                   `json:"tags,omitempty"`
	Control           *RoutingModelControl        `json:"control,omitempty"`
	Pricing           *RoutingPricing             `json:"pricing,omitempty"`
	Backends          *[]RoutingModelBackendInput `json:"backends,omitempty"`
}

type RoutingModelBackendView struct {
	ProviderID           string `json:"providerId"`
	ProviderModelID      string `json:"providerModelId"`
	CredentialConfigured bool   `json:"credentialConfigured"`
	Weight               string `json:"weight"`
}

type RoutingModelView struct {
	ID                string                    `json:"id"`
	Name              string                    `json:"name"`
	Status            string                    `json:"status"`
	Revision          int64                     `json:"revision"`
	ModelRevision     int64                     `json:"modelRevision"`
	CatalogRevision   string                    `json:"catalogRevision"`
	Aliases           []string                  `json:"aliases"`
	ParamSize         string                    `json:"paramSize,omitempty"`
	ContextWindowSize int                       `json:"contextWindowSize,omitempty"`
	Description       string                    `json:"description,omitempty"`
	Capabilities      []string                  `json:"capabilities"`
	Reasoning         RoutingReasoningFamily    `json:"reasoning,omitempty"`
	LoRAs             []string                  `json:"loras"`
	QualityScore      float64                   `json:"qualityScore,omitempty"`
	Modality          string                    `json:"modality,omitempty"`
	Tags              []string                  `json:"tags"`
	Control           RoutingModelControl       `json:"control"`
	Pricing           RoutingPricing            `json:"pricing"`
	Backends          []RoutingModelBackendView `json:"backends"`
	CreatedAt         time.Time                 `json:"createdAt"`
	UpdatedAt         time.Time                 `json:"updatedAt"`
}

type RoutingModelPage = Page[RoutingModelView]

// RoutingModelCard is the semantic Model metadata used while authoring Recipes
// and assigning Entrypoints. Runtime policy, pricing, provider bindings, and
// connection state deliberately live outside this view.
type RoutingModelCard struct {
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
}

type RoutingModelCardView struct {
	ID   string           `json:"id"`
	Name string           `json:"name"`
	Card RoutingModelCard `json:"card"`
}

type RoutingModelCardPage = Page[RoutingModelCardView]

type RoutingModelDetail struct {
	Data RoutingModelView `json:"data"`
}

type RoutingBulkModelSelection struct {
	CatalogItemID     string                 `json:"catalogItemId"`
	ID                string                 `json:"id,omitempty"`
	Name              string                 `json:"name"`
	Aliases           []string               `json:"aliases,omitempty"`
	ParamSize         string                 `json:"paramSize,omitempty"`
	ContextWindowSize int                    `json:"contextWindowSize,omitempty"`
	Description       string                 `json:"description,omitempty"`
	Capabilities      []string               `json:"capabilities,omitempty"`
	Reasoning         RoutingReasoningFamily `json:"reasoning,omitempty"`
	LoRAs             []string               `json:"loras,omitempty"`
	QualityScore      float64                `json:"qualityScore,omitempty"`
	Modality          string                 `json:"modality,omitempty"`
	Tags              []string               `json:"tags,omitempty"`
	Control           RoutingModelControl    `json:"control"`
	Pricing           RoutingPricing         `json:"pricing"`
}

type RoutingBulkImportRequest struct {
	ProviderID       string                      `json:"providerId"`
	InterfaceID      string                      `json:"interfaceId,omitempty"`
	CatalogRevision  string                      `json:"catalogRevision"`
	DiscoveryClaim   string                      `json:"discoveryClaim"`
	CredentialID     string                      `json:"credentialId,omitempty"`
	BaseURL          string                      `json:"baseUrl,omitempty"`
	ConnectionFields map[string]json.RawMessage  `json:"connectionFields,omitempty"`
	Weight           string                      `json:"weight,omitempty"`
	Selections       []RoutingBulkModelSelection `json:"selections"`
}

type RoutingRecipeWrite struct {
	ID          string          `json:"id,omitempty"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Document    json.RawMessage `json:"document"`
}

type RoutingRecipeView struct {
	ID             string                       `json:"id"`
	Name           string                       `json:"name"`
	Description    string                       `json:"description,omitempty"`
	Status         string                       `json:"status"`
	Revision       int64                        `json:"revision"`
	RecipeRevision int64                        `json:"recipeRevision"`
	Origin         string                       `json:"origin"`
	Immutable      bool                         `json:"immutable"`
	Provenance     *RoutingRecipeProvenanceView `json:"provenance,omitempty"`
	Decisions      []RoutingDecision            `json:"decisions"`
	Document       json.RawMessage              `json:"document"`
	CreatedAt      time.Time                    `json:"createdAt"`
	UpdatedAt      time.Time                    `json:"updatedAt"`
}

type RoutingRecipeProvenanceView struct {
	DistributionID      string    `json:"distributionId"`
	DistributionVersion string    `json:"distributionVersion"`
	AssetDigest         string    `json:"assetDigest"`
	SourceRecipeID      string    `json:"sourceRecipeId"`
	SourceRevision      int64     `json:"sourceRevision"`
	RecipeDigest        string    `json:"recipeDigest"`
	InstalledAt         time.Time `json:"installedAt"`
}

type RoutingRecipePage = Page[RoutingRecipeView]

type RoutingRecipeDetail struct {
	Data RoutingRecipeView `json:"data"`
}

type RoutingAssignmentWrite struct {
	ModelID   string                      `json:"modelId"`
	Priority  int                         `json:"priority,omitempty"`
	Weight    string                      `json:"weight,omitempty"`
	LoRAName  string                      `json:"loraName,omitempty"`
	Reasoning *RoutingAssignmentReasoning `json:"reasoning,omitempty"`
}

type RoutingAssignmentSetWrite struct {
	Models   []RoutingAssignmentWrite `json:"models"`
	Fallback *RoutingFallbackPolicy   `json:"fallback,omitempty"`
}

type RoutingEntrypointRuleWrite struct {
	ID          string                               `json:"id,omitempty"`
	Name        string                               `json:"name"`
	Matchers    []RoutingMatcher                     `json:"matchers,omitempty"`
	RecipeID    string                               `json:"recipeId"`
	Assignments map[string]RoutingAssignmentSetWrite `json:"assignments"`
}

type RoutingEntrypointWrite struct {
	ID      string                       `json:"id,omitempty"`
	Name    string                       `json:"name"`
	Aliases []string                     `json:"aliases"`
	Rules   []RoutingEntrypointRuleWrite `json:"rules"`
}

type RoutingEntrypointView struct {
	ID                 string                  `json:"id"`
	Name               string                  `json:"name"`
	Status             string                  `json:"status"`
	Revision           int64                   `json:"revision"`
	EntrypointRevision int64                   `json:"entrypointRevision"`
	Aliases            []string                `json:"aliases"`
	RuleCount          int                     `json:"ruleCount"`
	AssignedModelCount int                     `json:"assignedModelCount"`
	Rules              []RoutingEntrypointRule `json:"rules,omitempty"`
	CreatedAt          time.Time               `json:"createdAt"`
	UpdatedAt          time.Time               `json:"updatedAt"`
}

type RoutingEntrypointPage = Page[RoutingEntrypointView]

type RoutingEntrypointDetail struct {
	Data RoutingEntrypointView `json:"data"`
}

type RoutingResolveRequest struct {
	Path   string                       `json:"path,omitempty"`
	Claims map[string]RoutingClaimValue `json:"claims,omitempty"`
}

type RoutingResolveResponse struct {
	Outcome    string                     `json:"outcome"`
	Entrypoint *RoutingResolvedEntrypoint `json:"entrypoint,omitempty"`
	Rule       *RoutingEntrypointRule     `json:"rule,omitempty"`
	Recipe     *RoutingResolvedRecipe     `json:"recipe,omitempty"`
}

type RoutingResolvedEntrypoint struct {
	ID       string   `json:"id"`
	Revision int64    `json:"revision"`
	Name     string   `json:"name"`
	Aliases  []string `json:"aliases"`
}

type RoutingProbeResponse struct {
	Reachable           bool      `json:"reachable"`
	LatencyMilliseconds int64     `json:"latencyMilliseconds"`
	CheckedAt           time.Time `json:"checkedAt"`
}

type RoutingManifestImportRequest struct {
	Manifest string `json:"manifest"`
	DryRun   bool   `json:"dryRun,omitempty"`
}

type RoutingManifestResourceDiff struct {
	Create  []string `json:"create"`
	Update  []string `json:"update"`
	Disable []string `json:"disable"`
}

type RoutingManifestDiff struct {
	Models      RoutingManifestResourceDiff `json:"models"`
	Recipes     RoutingManifestResourceDiff `json:"recipes"`
	Entrypoints RoutingManifestResourceDiff `json:"entrypoints"`
}

type RoutingManifestImportResult struct {
	Diff            RoutingManifestDiff `json:"diff"`
	OperationID     string              `json:"operationId,omitempty"`
	DesiredRevision *uint64             `json:"desiredRevision,omitempty"`
	Replayed        bool                `json:"replayed"`
}
