// Package routingmanagement owns managed Model, Recipe, and Entrypoint
// authoring. PostgreSQL is desired-state authority; only published, complete
// Entrypoint dependency closures are compiled for the Router data plane.
package routingmanagement

import (
	"context"
	"encoding/json"
	"errors"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

var (
	ErrInvalid          = errors.New("routing management request is invalid")
	ErrNotFound         = errors.New("routing resource not found")
	ErrConflict         = errors.New("routing resource revision conflict")
	ErrImmutable        = errors.New("routing resource is immutable")
	ErrReferenced       = errors.New("routing resource is referenced")
	ErrPublication      = errors.New("routing publication failed")
	ErrClaim            = errors.New("model discovery claim is invalid")
	ErrProbeUnavailable = errors.New("model probe is unavailable")
)

type Status string

const (
	StatusDraft    Status = "draft"
	StatusActive   Status = "active"
	StatusDisabled Status = "disabled"
	StatusDeleted  Status = "deleted"
)

type MutationContext struct {
	PrincipalID string
	ActorChain  []string
	RequestID   string
	Reason      string
	Command     *managementcommand.Command
}

type RevisionReceipt struct {
	ResourceRevision int64
	DesiredRevision  int64
	OperationID      string
	Replayed         bool
}

type PageRequest struct {
	PageSize int
	Cursor   string
	Search   string
	Status   Status
	// Scope is compiled by Management authorization. Repositories consume it
	// before pagination; callers must never derive visibility from page rows.
	Scope accesscontrol.ResultScope
}

// ListCursor and ListQuery are the repository-side pagination contract. The
// service is solely responsible for validating and signing public cursors;
// repositories only receive a normalized, authorization-scoped seek.
type ListCursor struct {
	CreatedAt time.Time
	ID        string
}

type ListQuery struct {
	Limit  int
	Search string
	Status Status
	Scope  accesscontrol.ResultScope
	After  *ListCursor
}

type ListResult[T any] struct {
	Items   []T
	HasMore bool
}

type Page[T any] struct {
	Items      []T
	NextCursor string
	HasMore    bool
}

type ResourceIdentity struct {
	NamespaceID string
	ID          string
	Name        string
	Status      Status
	Revision    int64
	CreatedAt   time.Time
	UpdatedAt   time.Time
}

type Model struct {
	ResourceIdentity
	Current routingsnapshot.Model
}

type ModelBackendInput struct {
	ID               string
	ProviderID       string
	InterfaceID      string
	ProviderModelID  string
	CredentialID     string
	Origin           string
	ConnectionFields map[string]any
	Weight           string
}

type ModelInput struct {
	ID                string
	Name              string
	Aliases           []string
	ParamSize         string
	ContextWindowSize int
	Description       string
	Capabilities      []string
	Reasoning         routingsnapshot.ReasoningFamily
	LoRAs             []string
	QualityScore      float64
	Modality          string
	Tags              []string
	Execution         routingsnapshot.ModelExecution
	Pricing           routingsnapshot.ModelPricing
	Backends          []ModelBackendInput
}

// ModelPatch preserves omitted state. Backends remain server-owned unless a
// caller explicitly replaces them, so ordinary metadata and policy changes do
// not require secrets or provider connection details to make a round trip.
type ModelPatch struct {
	Name              *string
	Aliases           *[]string
	ParamSize         *string
	ContextWindowSize *int
	Description       *string
	Capabilities      *[]string
	Reasoning         *routingsnapshot.ReasoningFamily
	LoRAs             *[]string
	QualityScore      *float64
	Modality          *string
	Tags              *[]string
	Execution         *routingsnapshot.ModelExecution
	Pricing           *routingsnapshot.ModelPricing
	Backends          *[]ModelBackendInput
}

func (patch ModelPatch) Empty() bool {
	return patch.Name == nil && patch.Aliases == nil && patch.ParamSize == nil &&
		patch.ContextWindowSize == nil && patch.Description == nil && patch.Capabilities == nil &&
		patch.Reasoning == nil && patch.LoRAs == nil && patch.QualityScore == nil &&
		patch.Modality == nil && patch.Tags == nil && patch.Execution == nil &&
		patch.Pricing == nil && patch.Backends == nil
}

type BulkModelSelection struct {
	CatalogItemID     string
	ID                string
	Name              string
	Aliases           []string
	ParamSize         string
	ContextWindowSize int
	Description       string
	Capabilities      []string
	Reasoning         routingsnapshot.ReasoningFamily
	LoRAs             []string
	QualityScore      float64
	Modality          string
	Tags              []string
	Execution         routingsnapshot.ModelExecution
	Pricing           routingsnapshot.ModelPricing
}

type BulkImportRequest struct {
	NamespaceID      string
	AuthorityDigest  string
	CatalogRevision  string
	ProviderID       string
	InterfaceID      string
	DiscoveryClaim   string
	CredentialID     string
	Origin           string
	ConnectionFields map[string]any
	Weight           string
	Selections       []BulkModelSelection
}

type Recipe struct {
	ResourceIdentity
	Description string
	Current     routingsnapshot.Recipe
	Origin      RecipeOrigin
	Provenance  *RecipeProvenance
}

type RecipeOrigin string

const (
	RecipeOriginCustom       RecipeOrigin = "custom"
	RecipeOriginDistribution RecipeOrigin = "distribution"
)

// RecipeProvenance identifies the immutable Router distribution asset from
// which a built-in Recipe was installed. Custom Recipes have no provenance.
// Distribution upgrades create sibling resources instead of changing this
// record or any Entrypoint that already references it.
type RecipeProvenance struct {
	DistributionID      string
	DistributionVersion string
	AssetDigest         string
	SourceRecipeID      string
	SourceRevision      int64
	RecipeDigest        string
	InstalledAt         time.Time
}

type RecipeInput struct {
	ID          string
	Name        string
	Description string
	Document    json.RawMessage
}

type Entrypoint struct {
	ResourceIdentity
	Current            routingsnapshot.Entrypoint
	RuleCount          int
	AssignedModelCount int
}

type AssignmentInput struct {
	ModelID   string
	Priority  int
	Weight    string
	LoRAName  string
	Reasoning *routingsnapshot.AssignmentReasoning
}

type AssignmentSetInput struct {
	Models   []AssignmentInput
	Fallback *routingsnapshot.FallbackPolicy
}

type EntrypointRuleInput struct {
	ID          string
	Name        string
	Matchers    []routingsnapshot.Matcher
	RecipeID    string
	Assignments map[string]AssignmentSetInput
}

type EntrypointInput struct {
	ID      string
	Name    string
	Aliases []string
	Rules   []EntrypointRuleInput
}

type ProbeRequest struct {
	NamespaceID string
	Model       routingsnapshot.Model
	Timeout     time.Duration
}

type ProbeResult struct {
	Available bool
	Latency   time.Duration
	CheckedAt time.Time
}

type Prober interface {
	Probe(context.Context, ProbeRequest) (ProbeResult, error)
}

type CredentialVersionReader interface {
	Pin(context.Context, string, string, string) (string, error)
}
