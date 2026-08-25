// Package agentmanagement owns the Router-native Agent control-plane domain.
// It deliberately contains no HTTP, Dashboard, provider, or wire-protocol code.
package agentmanagement

import (
	"encoding/json"
	"errors"
	"net/netip"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

var (
	ErrInvalid         = errors.New("agent management request is invalid")
	ErrNotFound        = errors.New("agent resource not found")
	ErrConflict        = errors.New("agent resource revision conflict")
	ErrDenied          = errors.New("agent operation is denied")
	ErrLeaseLost       = errors.New("agent turn lease is no longer owned")
	ErrTerminal        = errors.New("agent turn is terminal")
	ErrCancelled       = errors.New("agent turn is cancelled")
	ErrApproval        = errors.New("agent publication approval is invalid")
	ErrUnsupported     = errors.New("agent content is unsupported")
	ErrToolUnavailable = errors.New("agent tool is unavailable")
	ErrHistoryExpired  = errors.New("agent event history has expired")
)

type Status string

const (
	StatusActive   Status = "active"
	StatusDisabled Status = "disabled"
	StatusDeleted  Status = "deleted"
)

type TargetKind string

const (
	TargetModel      TargetKind = "model"
	TargetEntrypoint TargetKind = "entrypoint"
)

type Target struct {
	Kind TargetKind `json:"kind"`
	ID   string     `json:"id"`
}

type SkillReference struct {
	ID       string `json:"id"`
	Revision int64  `json:"revision"`
}

type ResourceIdentity struct {
	ID          string    `json:"id"`
	NamespaceID string    `json:"namespaceId,omitempty"`
	Name        string    `json:"name"`
	Description string    `json:"description,omitempty"`
	Status      Status    `json:"status"`
	Revision    int64     `json:"revision"`
	CreatedAt   time.Time `json:"createdAt"`
	UpdatedAt   time.Time `json:"updatedAt"`
}

type Profile struct {
	ResourceIdentity
	ContentRevision           int64            `json:"contentRevision"`
	DefaultTarget             *Target          `json:"defaultTarget,omitempty"`
	MinimumTargetCapabilities []string         `json:"minimumTargetCapabilities"`
	SupportedModes            []SessionMode    `json:"supportedModes"`
	DefaultForModes           []SessionMode    `json:"defaultForModes"`
	Skills                    []SkillReference `json:"skills"`
	ToolPolicy                ToolPolicy       `json:"toolPolicy"`
	ApprovalPolicy            string           `json:"approvalPolicy"`
	MaximumTurnSeconds        int64            `json:"maximumTurnSeconds"`
	MaximumToolSteps          int              `json:"maximumToolSteps"`
	ContextTokenBudget        int64            `json:"contextTokenBudget"`
}

type ProfileInput struct {
	Name                      string           `json:"name"`
	Description               string           `json:"description,omitempty"`
	DefaultTarget             *Target          `json:"defaultTarget,omitempty"`
	MinimumTargetCapabilities []string         `json:"minimumTargetCapabilities,omitempty"`
	SupportedModes            []SessionMode    `json:"supportedModes,omitempty"`
	DefaultForModes           []SessionMode    `json:"defaultForModes,omitempty"`
	Skills                    []SkillReference `json:"skills,omitempty"`
	ToolPolicy                ToolPolicy       `json:"toolPolicy"`
	ApprovalPolicy            string           `json:"approvalPolicy,omitempty"`
	MaximumTurnSeconds        int64            `json:"maximumTurnSeconds,omitempty"`
	MaximumToolSteps          int              `json:"maximumToolSteps,omitempty"`
	ContextTokenBudget        int64            `json:"contextTokenBudget,omitempty"`
}

type ProfilePatch struct {
	Name                      *string           `json:"name,omitempty"`
	Description               *string           `json:"description,omitempty"`
	DefaultTarget             OptionalTarget    `json:"-"`
	MinimumTargetCapabilities *[]string         `json:"minimumTargetCapabilities,omitempty"`
	SupportedModes            *[]SessionMode    `json:"supportedModes,omitempty"`
	DefaultForModes           *[]SessionMode    `json:"defaultForModes,omitempty"`
	Skills                    *[]SkillReference `json:"skills,omitempty"`
	ToolPolicy                *ToolPolicy       `json:"toolPolicy,omitempty"`
	ApprovalPolicy            *string           `json:"approvalPolicy,omitempty"`
	MaximumTurnSeconds        *int64            `json:"maximumTurnSeconds,omitempty"`
	MaximumToolSteps          *int              `json:"maximumToolSteps,omitempty"`
	ContextTokenBudget        *int64            `json:"contextTokenBudget,omitempty"`
}

// OptionalTarget preserves the three JSON Patch states: omitted, a concrete
// target, and explicit null (clear the convenience default). HTTP DTOs decode
// into this type instead of relying on pointer ambiguity.
type OptionalTarget struct {
	Present bool
	Value   *Target
}

type ToolPolicy struct {
	Allow []string `json:"allow"`
	Deny  []string `json:"deny,omitempty"`
}

type Skill struct {
	ResourceIdentity
	ContentRevision     int64    `json:"contentRevision"`
	Builtin             bool     `json:"builtin"`
	Instructions        string   `json:"instructions,omitempty"`
	RequiredTools       []string `json:"requiredTools"`
	MinimumCapabilities []string `json:"minimumCapabilities"`
	ContentDigest       string   `json:"contentDigest"`
}

type SkillInput struct {
	Name                string   `json:"name"`
	Description         string   `json:"description,omitempty"`
	Instructions        string   `json:"instructions"`
	RequiredTools       []string `json:"requiredTools,omitempty"`
	MinimumCapabilities []string `json:"minimumCapabilities,omitempty"`
}

type SkillPatch struct {
	Name                *string   `json:"name,omitempty"`
	Description         *string   `json:"description,omitempty"`
	Instructions        *string   `json:"instructions,omitempty"`
	RequiredTools       *[]string `json:"requiredTools,omitempty"`
	MinimumCapabilities *[]string `json:"minimumCapabilities,omitempty"`
}

type ToolSource struct {
	ResourceIdentity
	ContentRevision         int64                  `json:"contentRevision"`
	Kind                    string                 `json:"kind"`
	Transport               string                 `json:"transport"`
	Endpoint                string                 `json:"endpoint"`
	CredentialID            string                 `json:"credentialId,omitempty"`
	EgressPolicy            EgressPolicy           `json:"egressPolicy"`
	DiscoveredTools         []ToolDefinition       `json:"discoveredTools"`
	DiscoveryDigest         string                 `json:"discoveryDigest,omitempty"`
	ApprovedDiscoveryDigest string                 `json:"approvedDiscoveryDigest,omitempty"`
	Availability            ToolSourceAvailability `json:"availability"`
}

type ToolSourceAvailability string

const (
	ToolSourceUndiscovered    ToolSourceAvailability = "undiscovered"
	ToolSourcePendingApproval ToolSourceAvailability = "pending_approval"
	ToolSourceReady           ToolSourceAvailability = "ready"
	ToolSourceDrifted         ToolSourceAvailability = "drifted"
	ToolSourceDisabled        ToolSourceAvailability = "disabled"
)

type ToolSourceInput struct {
	Name         string       `json:"name"`
	Description  string       `json:"description,omitempty"`
	Kind         string       `json:"kind"`
	Transport    string       `json:"transport"`
	Endpoint     string       `json:"endpoint"`
	CredentialID string       `json:"credentialId,omitempty"`
	EgressPolicy EgressPolicy `json:"egressPolicy"`
}

type ToolSourcePatch struct {
	Name         *string        `json:"name,omitempty"`
	Description  *string        `json:"description,omitempty"`
	Transport    *string        `json:"transport,omitempty"`
	Endpoint     *string        `json:"endpoint,omitempty"`
	CredentialID OptionalString `json:"-"`
	EgressPolicy *EgressPolicy  `json:"egressPolicy,omitempty"`
	Status       *Status        `json:"status,omitempty"`
}

// OptionalString preserves the three PATCH states: omitted, a value, and
// explicit null. HTTP adapters set Present when the property was supplied and
// keep Value nil when the caller explicitly clears the reference.
type OptionalString struct {
	Present bool
	Value   *string
}

type EgressPolicy struct {
	AllowedHosts        []string `json:"allowedHosts"`
	AllowedPorts        []int    `json:"allowedPorts,omitempty"`
	AllowedPrivateCIDRs []string `json:"allowedPrivateCidrs,omitempty"`
}

type SessionMode string

const (
	SessionChat    SessionMode = "chat"
	SessionBuilder SessionMode = "builder"
)

type SessionStatus string

const (
	SessionActive  SessionStatus = "active"
	SessionClosed  SessionStatus = "closed"
	SessionDeleted SessionStatus = "deleted"
)

type Session struct {
	ID                          string        `json:"id"`
	NamespaceID                 string        `json:"namespaceId"`
	OwnerPrincipalID            string        `json:"ownerPrincipalId"`
	EffectiveUserID             string        `json:"effectiveUserId,omitempty"`
	EffectiveTeamID             string        `json:"effectiveTeamId,omitempty"`
	KeyID                       string        `json:"keyId"`
	DelegatedInferenceSessionID string        `json:"-"`
	ProfileID                   string        `json:"profileId"`
	ProfileRevision             int64         `json:"profileRevision"`
	Target                      Target        `json:"target"`
	TargetResourceID            string        `json:"-"`
	AuthorityDigest             string        `json:"-"`
	Mode                        SessionMode   `json:"mode"`
	Title                       string        `json:"title"`
	Status                      SessionStatus `json:"status"`
	Revision                    int64         `json:"revision"`
	CreatedAt                   time.Time     `json:"createdAt"`
	UpdatedAt                   time.Time     `json:"updatedAt"`
}

type SessionInput struct {
	ProfileID       string      `json:"profileId,omitempty"`
	KeyID           string      `json:"keyId"`
	EffectiveTeamID string      `json:"effectiveTeamId,omitempty"`
	Target          Target      `json:"target"`
	Mode            SessionMode `json:"mode"`
	Title           string      `json:"title,omitempty"`
}

// SessionAccess is an internal Management authorization projection. It is
// intentionally not part of the public Session response contract.
type SessionAccess struct {
	ID               string
	OwnerPrincipalID string
	EffectiveUserID  string
	EffectiveTeamID  string
}

type SessionPatch struct {
	Title  *string        `json:"title,omitempty"`
	Status *SessionStatus `json:"status,omitempty"`
}

type TurnStatus string

const (
	TurnQueued          TurnStatus = "queued"
	TurnRunning         TurnStatus = "running"
	TurnWaitingApproval TurnStatus = "waiting_approval"
	TurnCompleted       TurnStatus = "completed"
	TurnFailed          TurnStatus = "failed"
	TurnCancelled       TurnStatus = "cancelled"
)

type ContentBlock struct {
	Type   string `json:"type"`
	Text   string `json:"text,omitempty"`
	URL    string `json:"url,omitempty"`
	Detail string `json:"detail,omitempty"`
	FileID string `json:"fileId,omitempty"`
}

type TurnInput struct {
	Content []ContentBlock `json:"content"`
}

type Turn struct {
	ID                string     `json:"id"`
	SessionID         string     `json:"sessionId"`
	Ordinal           int64      `json:"ordinal"`
	Status            TurnStatus `json:"status"`
	RegistryRevision  string     `json:"registryRevision,omitempty"`
	Fence             int64      `json:"-"`
	Input             TurnInput  `json:"input"`
	Revision          int64      `json:"revision"`
	CancelRequestedAt *time.Time `json:"cancelRequestedAt,omitempty"`
	Failure           *Failure   `json:"failure,omitempty"`
	CreatedAt         time.Time  `json:"createdAt"`
	UpdatedAt         time.Time  `json:"updatedAt"`
}

type Failure struct {
	Code      string `json:"code"`
	Message   string `json:"message"`
	Retryable bool   `json:"retryable"`
}

type EventType string

const (
	EventUserInput         EventType = "user_input"
	EventAssistantDelta    EventType = "assistant_delta"
	EventModelStepSummary  EventType = "model_step_summary"
	EventToolRequest       EventType = "tool_request"
	EventToolResult        EventType = "tool_result"
	EventProgress          EventType = "progress"
	EventContextCheckpoint EventType = "context_checkpoint"
	EventApprovalRequest   EventType = "approval_request"
	EventApprovalResult    EventType = "approval_result"
	EventCancellation      EventType = "cancellation"
	EventTerminal          EventType = "terminal"
)

type Event struct {
	SessionID string          `json:"sessionId"`
	TurnID    string          `json:"turnId,omitempty"`
	Sequence  int64           `json:"sequence"`
	Type      EventType       `json:"type"`
	Payload   json.RawMessage `json:"payload"`
	CreatedAt time.Time       `json:"createdAt"`
}

// Event payloads are closed, transport-safe summaries. Hidden reasoning,
// credentials, and unrestricted upstream failures are never represented by
// these types.
type UserInputEvent struct {
	Content []ContentBlock `json:"content"`
}

type AssistantDeltaKind string

const (
	AssistantTextDelta AssistantDeltaKind = "text"
)

type AssistantDelta struct {
	Kind AssistantDeltaKind `json:"kind"`
	Text string             `json:"text"`
}

type AssistantDeltaEvent struct {
	// ModelStepID and ChunkIndex make a committed delta the authoritative
	// replacement for any best-effort live preview from the same model step.
	// They are Router-assigned; an inference provider never controls either
	// value.
	ModelStepID string         `json:"modelStepId"`
	ChunkIndex  int            `json:"chunkIndex"`
	Delta       AssistantDelta `json:"delta"`
}

// ModelStepUsage is the authoritative token accounting returned by the
// Router's public inference path for one model step. Optional breakdowns are
// omitted when the upstream protocol did not provide authoritative evidence;
// the Agent runtime never estimates them.
type ModelStepUsage struct {
	InputTokens           int64  `json:"inputTokens"`
	OutputTokens          int64  `json:"outputTokens"`
	TotalTokens           int64  `json:"totalTokens"`
	InputUncachedTokens   *int64 `json:"inputUncachedTokens,omitempty"`
	InputCacheReadTokens  *int64 `json:"inputCacheReadTokens,omitempty"`
	InputCacheWriteTokens *int64 `json:"inputCacheWriteTokens,omitempty"`
	OutputReasoningTokens *int64 `json:"outputReasoningTokens,omitempty"`
	OutputOtherTokens     *int64 `json:"outputOtherTokens,omitempty"`
}

// ModelStepSummaryEvent is the closed, durable Router metadata surface for an
// assistant response. It intentionally excludes arbitrary provider payloads,
// hidden reasoning, credentials, request bodies, and inferred cost.
type ModelStepSummaryEvent struct {
	ModelStepID         string          `json:"modelStepId"`
	RequestID           string          `json:"requestId"`
	SelectedRecipe      string          `json:"selectedRecipe,omitempty"`
	SelectedDecision    string          `json:"selectedDecision,omitempty"`
	SelectedModel       string          `json:"selectedModel,omitempty"`
	SelectedAlgorithm   string          `json:"selectedAlgorithm,omitempty"`
	ResponsePath        string          `json:"responsePath,omitempty"`
	LatencyMilliseconds int64           `json:"latencyMilliseconds"`
	TTFTMilliseconds    *int64          `json:"ttftMilliseconds,omitempty"`
	Usage               *ModelStepUsage `json:"usage,omitempty"`
}

// LiveModelStepPhase describes transient model output. Live events are never
// inserted into agent_events and are intentionally absent from history and
// reconnect replay. A committed event tells an attached client to replace its
// preview with the durable assistant_delta events for ModelStepID; discarded
// removes a preview whose model step did not commit.
type LiveModelStepPhase string

const (
	LiveModelStepDelta     LiveModelStepPhase = "delta"
	LiveModelStepCommitted LiveModelStepPhase = "committed"
	LiveModelStepDiscarded LiveModelStepPhase = "discarded"
)

type LiveModelStepEvent struct {
	SessionID   string             `json:"sessionId"`
	TurnID      string             `json:"turnId"`
	ModelStepID string             `json:"modelStepId"`
	Phase       LiveModelStepPhase `json:"phase"`
	Ordinal     int                `json:"ordinal,omitempty"`
	Delta       *AssistantDelta    `json:"delta,omitempty"`
	CreatedAt   time.Time          `json:"createdAt"`
}

type ToolRequestEvent struct {
	InvocationID string          `json:"invocationId"`
	ToolName     string          `json:"toolName"`
	Arguments    json.RawMessage `json:"arguments"`
	Class        ToolClass       `json:"class"`
}

type ToolResultEvent struct {
	InvocationID string          `json:"invocationId"`
	ToolName     string          `json:"toolName"`
	Status       string          `json:"status"`
	Result       json.RawMessage `json:"result,omitempty"`
	ArtifactID   string          `json:"artifactId,omitempty"`
	Error        *Failure        `json:"error,omitempty"`
}

type ProgressEvent struct {
	Phase   string `json:"phase"`
	Message string `json:"message"`
}

type ContextCheckpointEvent struct {
	CheckpointID    string `json:"checkpointId"`
	ThroughSequence int64  `json:"throughSequence"`
}

type ApprovalRequestEvent struct {
	PlanID       string             `json:"planId"`
	PlanDigest   string             `json:"planDigest"`
	PlanRevision int64              `json:"planRevision"`
	PlanETag     string             `json:"planEtag"`
	ExpiresAt    time.Time          `json:"expiresAt"`
	Summary      PublicationSummary `json:"summary"`
}

type PublicationSummary struct {
	RecipeID         string          `json:"recipeId,omitempty"`
	RecipeName       string          `json:"recipeName,omitempty"`
	EntrypointID     string          `json:"entrypointId,omitempty"`
	EntrypointName   string          `json:"entrypointName,omitempty"`
	ChangedResources []string        `json:"changedResources,omitempty"`
	Warnings         []string        `json:"warnings,omitempty"`
	Topology         json.RawMessage `json:"topology,omitempty"`
	Assignments      json.RawMessage `json:"assignments,omitempty"`
	GateResults      json.RawMessage `json:"gateResults,omitempty"`
}

type ApprovalResultEvent struct {
	PlanID      string `json:"planId"`
	Status      string `json:"status"`
	OperationID string `json:"operationId,omitempty"`
}

type CancellationEvent struct {
	RequestedAt time.Time `json:"requestedAt"`
}

type TerminalEvent struct {
	Status TurnStatus `json:"status"`
	Error  *Failure   `json:"error,omitempty"`
}

type Artifact struct {
	ID          string          `json:"id"`
	SessionID   string          `json:"sessionId"`
	TurnID      string          `json:"turnId,omitempty"`
	Kind        string          `json:"kind"`
	MediaType   string          `json:"mediaType"`
	Content     []byte          `json:"content,omitempty"`
	Digest      string          `json:"digest"`
	SafePreview json.RawMessage `json:"safePreview"`
	ExpiresAt   time.Time       `json:"expiresAt"`
	CreatedAt   time.Time       `json:"createdAt"`
}

type ArtifactAccess struct {
	ID      string
	Session SessionAccess
}

type Checkpoint struct {
	ID                   string              `json:"id"`
	SessionID            string              `json:"sessionId"`
	TurnID               string              `json:"turnId"`
	ThroughSequence      int64               `json:"throughSequence"`
	Summary              string              `json:"summary"`
	UnresolvedGoals      []string            `json:"unresolvedGoals"`
	ResourceReferences   []ResourceReference `json:"resourceReferences"`
	ToolResultReferences []string            `json:"toolResultReferences"`
	Decisions            []string            `json:"decisions"`
	// State is the bounded, versioned execution projection used only by the
	// Router worker. Management responses expose the safe summary and
	// provenance fields, never this internal transcript projection.
	State     json.RawMessage `json:"-"`
	Digest    string          `json:"digest"`
	CreatedAt time.Time       `json:"createdAt"`
}

type ResourceReference struct {
	Kind     string `json:"kind"`
	ID       string `json:"id"`
	Revision string `json:"revision"`
}

type PublicationPlanStatus string

const (
	PublicationReady       PublicationPlanStatus = "ready"
	PublicationPublishing  PublicationPlanStatus = "publishing"
	PublicationCommitted   PublicationPlanStatus = "committed"
	PublicationExpired     PublicationPlanStatus = "expired"
	PublicationInvalidated PublicationPlanStatus = "invalidated"
	PublicationFailed      PublicationPlanStatus = "failed"
)

type PublicationPlan struct {
	ID                         string                `json:"id"`
	SessionID                  string                `json:"sessionId"`
	TurnID                     string                `json:"turnId"`
	RecipeID                   string                `json:"recipeId"`
	RecipeContentRevision      int64                 `json:"recipeContentRevision"`
	RecipeResourceRevision     int64                 `json:"recipeResourceRevision"`
	EntrypointID               string                `json:"entrypointId"`
	EntrypointContentRevision  int64                 `json:"entrypointContentRevision"`
	EntrypointResourceRevision int64                 `json:"entrypointResourceRevision"`
	CatalogRevision            string                `json:"catalogRevision"`
	ExactDiff                  json.RawMessage       `json:"exactDiff"`
	Diagnostics                json.RawMessage       `json:"diagnostics"`
	GateResults                json.RawMessage       `json:"gateResults"`
	Digest                     string                `json:"digest"`
	Status                     PublicationPlanStatus `json:"status"`
	ExpiresAt                  time.Time             `json:"expiresAt"`
	Revision                   int64                 `json:"revision"`
	OperationID                string                `json:"operationId,omitempty"`
	CreatedAt                  time.Time             `json:"createdAt"`
	UpdatedAt                  time.Time             `json:"updatedAt"`
}

// PublicationAccess is an internal authorization projection for the separate
// human commit endpoint. The public API exposes approval data only through the
// durable approval_request event.
type PublicationAccess struct {
	PlanID       string
	SessionID    string
	RecipeID     string
	EntrypointID string
	ModelIDs     []string
	Revision     int64
	Digest       string
	ExpiresAt    time.Time
	Session      SessionAccess
}

// PublicationCommitReservation is the durable hand-off between human
// approval and the existing Routing publication command. Publishing is an
// explicit state so cancellation cannot win between the Routing transaction
// and the Agent terminal event.
type PublicationCommitReservation struct {
	Plan            PublicationPlan
	PrincipalID     string
	OperationID     string
	DesiredRevision int64
	Replayed        bool
}

type PublicationCommitResult struct {
	Plan            PublicationPlan
	DesiredRevision int64
	ApprovalEvent   Event
	TerminalEvent   Event
	Replayed        bool
}

type PageRequest struct {
	PageSize         int
	Cursor           string
	Search           string
	Scope            accesscontrol.ResultScope
	OwnerPrincipalID string
}

// ToolPageRequest pages the immutable Tool Registry by canonical tool name.
// The opaque cursor additionally binds the registry revision and normalized
// search so a catalog refresh cannot silently change a page traversal.
type ToolPageRequest struct {
	PageSize int
	Cursor   string
	Search   string
}

type Page[T any] struct {
	Items      []T    `json:"items"`
	NextCursor string `json:"nextCursor,omitempty"`
	HasMore    bool   `json:"hasMore"`
}

type EventPageRequest struct {
	PageSize int
	Cursor   string
	Scope    accesscontrol.ResultScope
}

type ArtifactContent struct {
	ID        string `json:"id"`
	MediaType string `json:"mediaType"`
	Encoding  string `json:"encoding"`
	Content   []byte `json:"content"`
	Digest    string `json:"digest"`
}

type MutationContext struct {
	PrincipalID         string
	ManagementSessionID string
	ActorChain          []string
	RequestID           string
	Reason              string
	SourceIP            netip.Addr
}

// AccessContext is produced only by the Management authorization layer. The
// application service never reconstructs grants from a role name or trusts a
// client-supplied owner filter.
type AccessContext struct {
	PrincipalID string
	Scope       accesscontrol.ResultScope
}

type HistoryRecovery struct {
	CheckpointID    string `json:"checkpointId"`
	ThroughSequence int64  `json:"throughSequence"`
}

type HistoryExpiredError struct {
	Recovery HistoryRecovery
}

func (HistoryExpiredError) Error() string { return ErrHistoryExpired.Error() }
func (HistoryExpiredError) Unwrap() error { return ErrHistoryExpired }
