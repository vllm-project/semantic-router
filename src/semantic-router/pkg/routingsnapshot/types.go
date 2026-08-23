// Package routingsnapshot defines the immutable routing value shared by
// standalone manifests and the managed publication pipeline.
package routingsnapshot

import (
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type Lifecycle string

const (
	LifecycleActive Lifecycle = "active"
)

// Bundle is a complete publication candidate. References always use stable IDs.
type Bundle struct {
	NamespaceID string       `json:"namespaceId"`
	Revision    int64        `json:"revision"`
	Currency    string       `json:"currency,omitempty"`
	Models      []Model      `json:"models"`
	Recipes     []Recipe     `json:"recipes"`
	Entrypoints []Entrypoint `json:"entrypoints"`
}

type Model struct {
	ID                string          `json:"id"`
	Revision          int64           `json:"revision"`
	CatalogRevision   string          `json:"catalogRevision"`
	Name              string          `json:"name"`
	Aliases           []string        `json:"aliases,omitempty"`
	ParamSize         string          `json:"paramSize,omitempty"`
	ContextWindowSize int             `json:"contextWindowSize,omitempty"`
	Description       string          `json:"description,omitempty"`
	Capabilities      []string        `json:"capabilities,omitempty"`
	Reasoning         ReasoningFamily `json:"reasoning,omitempty"`
	LoRAs             []string        `json:"loras,omitempty"`
	QualityScore      float64         `json:"qualityScore,omitempty"`
	Modality          string          `json:"modality,omitempty"`
	Tags              []string        `json:"tags,omitempty"`
	Execution         ModelExecution  `json:"execution"`
	Pricing           ModelPricing    `json:"pricing"`
	Backends          []Backend       `json:"backends"`
}

type ReasoningFamily struct {
	Type    string   `json:"type,omitempty"`
	Efforts []string `json:"efforts,omitempty"`
}

type ModelExecution struct {
	MaxRetries     int    `json:"maxRetries"`
	RequestTimeout string `json:"requestTimeout"`
	StreamTimeout  string `json:"streamTimeout"`
}

type ModelPricing struct {
	InputCostPerMillionTokens      *string `json:"inputCostPerMillionTokens"`
	OutputCostPerMillionTokens     *string `json:"outputCostPerMillionTokens"`
	CacheReadCostPerMillionTokens  *string `json:"cacheReadCostPerMillionTokens"`
	CacheWriteCostPerMillionTokens *string `json:"cacheWriteCostPerMillionTokens"`
}

type Backend struct {
	ID                   string                 `json:"id"`
	ProviderID           string                 `json:"providerId"`
	WireFormat           llmprotocol.WireFormat `json:"wireFormat"`
	Origin               string                 `json:"origin"`
	ProviderModelID      string                 `json:"providerModelId"`
	ProviderCredentialID string                 `json:"providerCredentialId,omitempty"`
	Connection           BackendConnection      `json:"connection"`
	Weight               string                 `json:"weight"`
}

// BackendConnection is the non-secret wire configuration compiled from a
// provider catalog revision. Credentials are referenced separately and never
// enter this value.
type BackendConnection struct {
	Path    string            `json:"path"`
	Headers map[string]string `json:"headers,omitempty"`
}

type Recipe struct {
	ID          string          `json:"id"`
	Revision    int64           `json:"revision"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Decisions   []Decision      `json:"decisions"`
	Document    json.RawMessage `json:"document"`
}

type Decision struct {
	ID                  string              `json:"id"`
	Name                string              `json:"name"`
	DispatchCardinality DispatchCardinality `json:"dispatchCardinality"`
}

type DispatchCardinality string

const (
	DispatchCardinalitySingle DispatchCardinality = "single"
	DispatchCardinalityMulti  DispatchCardinality = "multi"
)

type Entrypoint struct {
	ID       string           `json:"id"`
	Revision int64            `json:"revision"`
	Name     string           `json:"name"`
	Aliases  []string         `json:"aliases"`
	Rules    []EntrypointRule `json:"rules"`
}

type EntrypointRule struct {
	ID             string                   `json:"id"`
	Name           string                   `json:"name"`
	Matchers       []Matcher                `json:"matchers,omitempty"`
	RecipeID       string                   `json:"recipeId"`
	RecipeRevision int64                    `json:"recipeRevision"`
	Assignments    map[string]AssignmentSet `json:"assignments"`
}

type Matcher struct {
	Claim      *ClaimMatcher `json:"claim,omitempty"`
	ExactPath  string        `json:"exactPath,omitempty"`
	PathPrefix string        `json:"pathPrefix,omitempty"`
}

type ClaimMatcher struct {
	Name  string     `json:"name"`
	Value ClaimValue `json:"value"`
}

type ClaimValue struct {
	Kind    string `json:"kind"`
	String  string `json:"string,omitempty"`
	Boolean bool   `json:"boolean,omitempty"`
	Integer int64  `json:"integer,omitempty"`
}

type Assignment struct {
	ModelID       string               `json:"modelId"`
	ModelRevision int64                `json:"modelRevision"`
	Priority      int                  `json:"priority"`
	Weight        string               `json:"weight"`
	LoRAName      string               `json:"loraName,omitempty"`
	Reasoning     *AssignmentReasoning `json:"reasoning,omitempty"`
}

type AssignmentSet struct {
	Models   []Assignment    `json:"models"`
	Fallback *FallbackPolicy `json:"fallback,omitempty"`
}

type FallbackPolicy struct {
	Strategy string   `json:"strategy"`
	On       []string `json:"on"`
}

type AssignmentReasoning struct {
	Enabled     bool   `json:"enabled"`
	Effort      string `json:"effort,omitempty"`
	Description string `json:"description,omitempty"`
}

// Snapshot is a validated, content-addressed runtime value.
type Snapshot struct {
	Bundle
	Digest string `json:"digest"`

	modelsByID      map[string]Model
	recipesByID     map[string]Recipe
	entrypointsByID map[string]Entrypoint
	aliases         map[string]string
}

type ResolveOutcome string

const (
	ResolveMatched        ResolveOutcome = "matched"
	ResolveClaimedNoMatch ResolveOutcome = "claimed_no_match"
	ResolveUnclaimed      ResolveOutcome = "unclaimed"
)

type ResolveInput struct {
	Alias  string
	Path   string
	Claims map[string]ClaimValue
}

type Resolution struct {
	Outcome    ResolveOutcome
	Entrypoint *Entrypoint
	Rule       *EntrypointRule
	Recipe     *Recipe
}
