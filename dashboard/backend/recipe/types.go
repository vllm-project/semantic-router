package recipe

import (
	"context"
	"encoding/json"
	"time"
)

const (
	MetadataSchemaVersion = "vllm-sr/recipe-metadata/v1"
	ProbeSchemaVersion    = "v1"
)

// Metadata is the stable, distribution-facing identity of one managed Recipe.
// Runtime routing, provider endpoints, credentials, and benchmark results do
// not belong in this document.
type Metadata struct {
	SchemaVersion string        `json:"schema_version" yaml:"schema_version"`
	ID            string        `json:"id" yaml:"id"`
	Name          string        `json:"name" yaml:"name"`
	Version       string        `json:"version" yaml:"version"`
	Description   string        `json:"description" yaml:"description"`
	Authors       []Author      `json:"authors" yaml:"authors"`
	License       string        `json:"license" yaml:"license"`
	Tags          []string      `json:"tags" yaml:"tags"`
	Links         MetadataLinks `json:"links" yaml:"links"`
}

type Author struct {
	Name  string  `json:"name" yaml:"name"`
	Email *string `json:"email,omitempty" yaml:"email,omitempty"`
	URL   *string `json:"url,omitempty" yaml:"url,omitempty"`
}

type MetadataLinks struct {
	Homepage      *string `json:"homepage,omitempty" yaml:"homepage,omitempty"`
	Source        string  `json:"source" yaml:"source"`
	Documentation *string `json:"documentation,omitempty" yaml:"documentation,omitempty"`
}

type FileHealth struct {
	Present   bool   `json:"present"`
	Digest    string `json:"digest,omitempty"`
	SizeBytes int64  `json:"size_bytes,omitempty"`
}

type SourceHealth struct {
	Status string                `json:"status"`
	Files  map[string]FileHealth `json:"files"`
	Issues []string              `json:"issues"`
}

type Digests struct {
	Recipe   string `json:"recipe,omitempty"`
	Metadata string `json:"metadata,omitempty"`
	Config   string `json:"config,omitempty"`
	Probes   string `json:"probes,omitempty"`
	DSL      string `json:"dsl,omitempty"`
	README   string `json:"readme,omitempty"`
}

type Counts struct {
	UnifiedModels int `json:"unified_models"`
	Recipes       int `json:"recipes"`
	Decisions     int `json:"decisions"`
	Probes        int `json:"probes"`
}

type Descriptor struct {
	Managed            bool         `json:"managed"`
	Metadata           *Metadata    `json:"metadata,omitempty"`
	README             string       `json:"readme,omitempty"`
	SourceHealth       SourceHealth `json:"source_health"`
	Digests            Digests      `json:"digests"`
	Counts             Counts       `json:"counts"`
	ActivationState    string       `json:"activation_state"`
	ActiveRecipeDigest string       `json:"active_recipe_digest,omitempty"`
}

const (
	ActivationNone         = "none"
	ActivationActive       = "active"
	ActivationRecovering   = "recovering"
	ActivationInconsistent = "inconsistent"
)

type PackageWarning struct {
	Code    string `json:"code"`
	Path    string `json:"path"`
	Message string `json:"message"`
}

type PackageSummary struct {
	ID              string           `json:"id"`
	Name            string           `json:"name"`
	Version         string           `json:"version"`
	Description     string           `json:"description"`
	RecipeDigest    string           `json:"recipe_digest"`
	ConfigDigest    string           `json:"config_digest"`
	ArchiveSHA256   string           `json:"archive_sha256"`
	ArchiveVerified bool             `json:"archive_verified"`
	InstalledAt     time.Time        `json:"installed_at"`
	Active          bool             `json:"active"`
	Counts          Counts           `json:"counts"`
	Warnings        []PackageWarning `json:"warnings"`
}

type PackageList struct {
	Items           []PackageSummary `json:"items"`
	Page            int              `json:"page"`
	PageSize        int              `json:"page_size"`
	Total           int              `json:"total"`
	TotalPages      int              `json:"total_pages"`
	ActiveDigest    string           `json:"active_digest,omitempty"`
	ActivationState string           `json:"activation_state"`
}

type PackageListOptions struct {
	Page     int
	PageSize int
	Query    string
}

type ImportRequest struct {
	URL                   string `json:"url"`
	ExpectedArchiveSHA256 string `json:"expected_archive_sha256,omitempty"`
}

type ActivateRequest struct {
	RecipeDigest           string `json:"recipe_digest"`
	AcknowledgeWarnings    bool   `json:"acknowledge_warnings,omitempty"`
	ExpectedPlanDigest     string `json:"expected_plan_digest,omitempty"`
	ConfirmStackRecreation bool   `json:"confirm_stack_recreation,omitempty"`
}

type ActivateResult struct {
	Status               string `json:"status"`
	RecipeDigest         string `json:"recipe_digest"`
	PreviousRecipeDigest string `json:"previous_recipe_digest,omitempty"`
	PlanDigest           string `json:"plan_digest"`
	Mode                 string `json:"mode"`
}

type DeactivateRequest struct {
	ExpectedPlanDigest     string `json:"expected_plan_digest,omitempty"`
	ConfirmStackRecreation bool   `json:"confirm_stack_recreation,omitempty"`
}

type DeactivateResult struct {
	Status               string `json:"status"`
	PreviousRecipeDigest string `json:"previous_recipe_digest,omitempty"`
	PlanDigest           string `json:"plan_digest,omitempty"`
	Mode                 string `json:"mode,omitempty"`
}

type ExpectedAssertions struct {
	Decision         string              `json:"decision"`
	Recipe           string              `json:"recipe,omitempty"`
	Algorithm        string              `json:"algorithm,omitempty"`
	Alias            string              `json:"alias,omitempty"`
	Plugins          []string            `json:"plugins"`
	ForbiddenPlugins []string            `json:"forbidden_plugins"`
	PluginMatch      string              `json:"plugin_match"`
	Signals          map[string][]string `json:"signals"`
	ForbiddenSignals map[string][]string `json:"forbidden_signals"`
	SignalMatch      string              `json:"signal_match"`
}

type Padding struct {
	Text      string `json:"text" yaml:"text"`
	Repeat    int    `json:"repeat" yaml:"repeat"`
	Placement string `json:"placement" yaml:"placement"`
}

type GeneratedText struct {
	MessageIndex    int    `json:"message_index"`
	ContentIndex    int    `json:"content_index"`
	TargetTextBytes int    `json:"target_text_bytes"`
	Character       string `json:"character"`
}

type ImageFixtureMetadata struct {
	Description string `json:"description"`
	MediaType   string `json:"media_type"`
	Bytes       int    `json:"bytes"`
	SHA256      string `json:"sha256"`
}

type ProbeSummary struct {
	ID            string             `json:"id"`
	DecisionID    string             `json:"decision_id"`
	VariantID     string             `json:"variant_id"`
	Model         string             `json:"model,omitempty"`
	DisplayPrompt string             `json:"display_prompt,omitempty"`
	QueryPreview  string             `json:"query_preview"`
	RequestShapes []string           `json:"request_shapes"`
	Tags          []string           `json:"tags"`
	Expected      ExpectedAssertions `json:"expected"`
	Playground    PlaygroundPolicy   `json:"playground"`
	Editable      bool               `json:"editable"`
}

type PlaygroundPolicy struct {
	Enabled bool   `json:"enabled"`
	Reason  string `json:"reason,omitempty"`
}

type ProbeDetail struct {
	ProbeSummary
	Query            string                          `json:"query,omitempty"`
	Messages         []map[string]any                `json:"messages,omitempty"`
	Tools            []map[string]any                `json:"tools,omitempty"`
	Repeat           int                             `json:"repeat"`
	Padding          *Padding                        `json:"padding,omitempty"`
	GeneratedText    *GeneratedText                  `json:"generated_text,omitempty"`
	ImageFixtures    map[string]ImageFixtureMetadata `json:"image_fixtures,omitempty"`
	Notes            string                          `json:"notes,omitempty"`
	RawPayloadHidden bool                            `json:"raw_payload_hidden,omitempty"`
	imageFixtures    map[string]probeImageFixture
}

type Facets struct {
	Recipes   map[string]int `json:"recipes"`
	Decisions map[string]int `json:"decisions"`
	Tags      map[string]int `json:"tags"`
	Models    map[string]int `json:"models"`
	Shapes    map[string]int `json:"shapes"`
}

type ProbeList struct {
	Items        []ProbeSummary `json:"items"`
	Page         int            `json:"page"`
	PageSize     int            `json:"page_size"`
	Total        int            `json:"total"`
	TotalPages   int            `json:"total_pages"`
	Facets       Facets         `json:"facets"`
	RecipeDigest string         `json:"recipe_digest"`
}

type ListOptions struct {
	Page     int
	PageSize int
	Query    string
	Recipe   string
	Decision string
	Tag      string
	Model    string
	Shape    string
}

type ChatRequest struct {
	Model               string           `json:"model,omitempty"`
	Messages            []map[string]any `json:"messages"`
	Tools               []map[string]any `json:"tools,omitempty"`
	Stream              bool             `json:"stream"`
	MaxCompletionTokens int              `json:"max_completion_tokens"`
}

type RunPlan struct {
	ProbeID      string           `json:"probe_id"`
	RecipeDigest string           `json:"recipe_digest"`
	Recipe       string           `json:"recipe"`
	Model        string           `json:"model,omitempty"`
	Messages     []map[string]any `json:"messages"`
	Tools        []map[string]any `json:"tools,omitempty"`
	Request      ChatRequest      `json:"request"`
	Editable     bool             `json:"editable"`
}

type EvalRequest struct {
	Text     string           `json:"text,omitempty"`
	Messages []map[string]any `json:"messages,omitempty"`
	Tools    []map[string]any `json:"tools,omitempty"`
	Model    string           `json:"model,omitempty"`
}

type Evaluator interface {
	Evaluate(context.Context, EvalRequest) (json.RawMessage, error)
}

// RequestModelResolver resolves a Recipe to one request-facing model exposed
// by the active Router. Model-free Recipe packages use this seam instead of
// embedding environment-specific Entrypoint names in their probe catalog.
type RequestModelResolver interface {
	ResolveRequestModel(context.Context, string) (string, error)
}

type ActualOutcome struct {
	Decision          string              `json:"decision"`
	Model             string              `json:"model,omitempty"`
	RequestedModel    string              `json:"requested_model,omitempty"`
	SelectionStatus   string              `json:"selection_status,omitempty"`
	SelectionMethod   string              `json:"selection_method,omitempty"`
	SelectionReason   string              `json:"selection_reason,omitempty"`
	Recipe            string              `json:"recipe,omitempty"`
	Algorithm         string              `json:"algorithm,omitempty"`
	Plugins           []string            `json:"plugins"`
	RecommendedModels []string            `json:"recommended_models"`
	MatchedSignals    map[string][]string `json:"matched_signals"`
	TraceDecisions    []string            `json:"trace_decisions"`
}

type ValidationChecks struct {
	Decision  bool `json:"decision"`
	Model     bool `json:"model"`
	Recipe    bool `json:"recipe"`
	Algorithm bool `json:"algorithm"`
	Selection bool `json:"selection"`
	Plugins   bool `json:"plugins"`
	Signals   bool `json:"signals"`
	Alias     bool `json:"alias"`
	Trace     bool `json:"trace"`
}

type ValidationResult struct {
	ProbeID      string               `json:"probe_id"`
	RecipeDigest string               `json:"recipe_digest"`
	Passed       bool                 `json:"passed"`
	Expected     ExpectedAssertions   `json:"expected"`
	Actual       ActualOutcome        `json:"actual"`
	Checks       ValidationChecks     `json:"checks"`
	Failures     []string             `json:"failures"`
	Provenance   ValidationProvenance `json:"provenance"`
	LatencyMS    int64                `json:"latency_ms"`
	Error        string               `json:"error,omitempty"`
}
