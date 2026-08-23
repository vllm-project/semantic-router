package agentmanagement

import (
	"context"
	"encoding/json"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
)

type Seek struct {
	Timestamp time.Time
	ID        string
}

type ListQuery struct {
	Limit            int
	After            *Seek
	Search           string
	OwnerPrincipalID string
	Scope            accesscontrol.ResultScope
}

type ListResult[T any] struct {
	Items   []T
	HasMore bool
}

// ResourceCommand is the exact durable mutation identity handed from the
// Agent application service to PostgreSQL. The transaction serializes it
// before changing domain state and persists its receipt before commit.
type ResourceCommand struct {
	Mutation MutationContext
	Command  managementcommand.Command
}

// ResourceMutationResult is the immutable synchronous receipt returned for a
// first execution or replay. Mutable resource bodies are deliberately absent.
type ResourceMutationResult struct {
	ResourceID       string
	ResourceRevision int64
	Replayed         bool
}

// EventHistoryQuery pages backwards from the newest retained event while
// returning every page in ascending sequence order. BeforeSequence is
// exclusive; zero means the newest retained page.
type EventHistoryQuery struct {
	BeforeSequence int64
	Limit          int
}

type CreateTurnRequest struct {
	Turn             Turn
	NamespaceID      string
	ActorPrincipalID string
	Command          managementcommand.Command
}

type TurnLease struct {
	NamespaceID      string
	SessionID        string
	TurnID           string
	WorkerID         string
	Fence            int64
	RegistryRevision string
	ExpiresAt        time.Time
}

type TurnTransition struct {
	Lease       TurnLease
	Status      TurnStatus
	Failure     *Failure
	Approval    *ApprovalRequestEvent
	CompletedAt time.Time
}

// ModelStep is the durable boundary around one delegated public-inference
// request. A replacement worker may replay a completed step, but it must never
// issue a second request for a step whose outcome is unknown.
type ModelStep struct {
	ID               string
	NamespaceID      string
	SessionID        string
	TurnID           string
	Ordinal          int64
	WorkerID         string
	Fence            int64
	RegistryRevision string
	RequestDigest    []byte
	Status           string
	StopReason       string
	OutputDigest     []byte
	StartedAt        time.Time
	CompletedAt      *time.Time
}

type ModelStepCommit struct {
	Lease      TurnLease
	Step       ModelStep
	Events     []EventAppend
	Checkpoint Checkpoint
}

type ModelStepCommitResult struct {
	Step            ModelStep
	Events          []Event
	Checkpoint      Checkpoint
	CheckpointEvent Event
}

type EventAppend struct {
	NamespaceID string
	SessionID   string
	TurnID      string
	Origin      string
	Fence       *int64
	Type        EventType
	Payload     json.RawMessage
}

type InvocationRecord struct {
	ID                  string
	NamespaceID         string
	SessionID           string
	TurnID              string
	Fence               int64
	RegistryRevision    string
	ToolName            string
	CredentialVersionID string
	InputDigest         []byte
	Input               json.RawMessage
	Idempotency         ToolIdempotency
	Class               ToolClass
	Status              string
	Result              json.RawMessage
	ArtifactID          string
	ErrorCode           string
	StartedAt           time.Time
	CompletedAt         *time.Time
}

type ToolCredential struct {
	ResourceIdentity
	ActiveVersionID string `json:"-"`
}

type ToolCredentialInput struct {
	Name   string `json:"name"`
	Secret []byte `json:"-"`
}

type ToolCredentialPatch struct {
	Name   *string `json:"name,omitempty"`
	Status *Status `json:"status,omitempty"`
}

type EncryptedSecret struct {
	Ciphertext []byte
	Nonce      []byte
	KEKVersion string
}

type ToolCredentialSecret struct {
	CredentialID string
	VersionID    string
	Secret       EncryptedSecret
	ExpiresAt    *time.Time
}

type ToolManifest struct {
	Definition ToolDefinition `json:"definition"`
	Origin     ToolOrigin     `json:"origin"`
}

type RegistryManifest struct {
	Revision  string         `json:"revision"`
	Tools     []ToolManifest `json:"tools"`
	CreatedAt time.Time      `json:"createdAt"`
	ExpiresAt time.Time      `json:"expiresAt"`
}

type Store interface {
	ListProfiles(context.Context, string, ListQuery) (ListResult[Profile], error)
	GetProfile(context.Context, string, string) (Profile, error)
	GetProfileRevision(context.Context, string, string, int64) (Profile, error)
	GetDefaultProfile(context.Context, string, SessionMode) (Profile, error)
	CreateProfile(context.Context, string, string, ProfileInput, ResourceCommand) (ResourceMutationResult, error)
	PatchProfile(context.Context, string, string, int64, ProfilePatch, MutationContext) (Profile, error)
	DeleteProfile(context.Context, string, string, int64, MutationContext) (int64, error)

	ListSkills(context.Context, string, ListQuery) (ListResult[Skill], error)
	GetSkill(context.Context, string, string) (Skill, error)
	GetSkillRevision(context.Context, string, string, int64) (Skill, error)
	CreateSkill(context.Context, string, string, SkillInput, ResourceCommand) (ResourceMutationResult, error)
	PatchSkill(context.Context, string, string, int64, SkillPatch, MutationContext) (Skill, error)
	DeleteSkill(context.Context, string, string, int64, MutationContext) (int64, error)

	ListToolCredentials(context.Context, string, ListQuery) (ListResult[ToolCredential], error)
	GetToolCredential(context.Context, string, string) (ToolCredential, error)
	CreateToolCredential(context.Context, string, string, string, EncryptedSecret, ResourceCommand) (ResourceMutationResult, error)
	PatchToolCredential(context.Context, string, string, int64, ToolCredentialPatch, MutationContext) (ToolCredential, error)
	RotateToolCredential(context.Context, string, string, int64, EncryptedSecret, time.Time, ResourceCommand) (ResourceMutationResult, error)
	DeleteToolCredential(context.Context, string, string, int64, MutationContext) (int64, error)
	ResolveToolCredentialSecret(context.Context, string, string, string) (ToolCredentialSecret, error)

	ListToolSources(context.Context, string, ListQuery) (ListResult[ToolSource], error)
	GetToolSource(context.Context, string, string) (ToolSource, error)
	GetToolSourceRevision(context.Context, string, string, int64) (ToolSource, error)
	ListRegistryToolSources(context.Context, string) ([]ToolSource, error)
	CreateToolSource(context.Context, string, string, ToolSourceInput, ResourceCommand) (ResourceMutationResult, error)
	PatchToolSource(context.Context, string, string, int64, ToolSourcePatch, MutationContext) (ToolSource, error)
	DeleteToolSource(context.Context, string, string, int64, MutationContext) (int64, error)
	UpdateToolSourceDiscovery(context.Context, string, string, int64, []ToolDefinition, ResourceCommand) (ResourceMutationResult, error)
	ApproveToolSourceDiscovery(context.Context, string, string, int64, string, ResourceCommand) (ResourceMutationResult, error)
	ReplayResourceCommand(context.Context, managementcommand.Command, string) (ResourceMutationResult, bool, error)
	PutRegistryManifest(context.Context, string, RegistryManifest) error
	GetRegistryManifest(context.Context, string, string) (RegistryManifest, error)

	ListSessions(context.Context, string, ListQuery) (ListResult[Session], error)
	GetSession(context.Context, string, string) (Session, error)
	PatchSession(context.Context, string, string, int64, SessionPatch, MutationContext) (Session, error)
	DeleteSession(context.Context, string, string, int64, MutationContext) (int64, error)

	CreateTurn(context.Context, CreateTurnRequest) (Turn, bool, error)
	ListTurns(context.Context, string, string, ListQuery) (ListResult[Turn], error)
	GetTurn(context.Context, string, string, string) (Turn, error)
	ClaimNextTurn(context.Context, string, time.Time) (TurnLease, error)
	RenewTurn(context.Context, TurnLease, time.Time) (TurnLease, error)
	TransitionTurn(context.Context, TurnTransition) (Event, error)
	RequestCancellation(context.Context, string, string, string, time.Time) (Turn, bool, error)
	CancellationRequested(context.Context, TurnLease) (bool, error)
	BeginModelStep(context.Context, ModelStep) (ModelStep, bool, error)
	CommitModelStep(context.Context, ModelStepCommit) (ModelStepCommitResult, error)

	AppendEvent(context.Context, EventAppend) (Event, error)
	ListEventsAfter(context.Context, string, string, int64, int) ([]Event, bool, error)
	ListEventHistory(context.Context, string, string, EventHistoryQuery) ([]Event, bool, error)
	OldestEventSequence(context.Context, string, string) (int64, error)

	BeginInvocation(context.Context, InvocationRecord) (InvocationRecord, bool, error)
	FinishInvocation(context.Context, InvocationRecord) (Event, error)
	GetInvocation(context.Context, string, string, string, string) (InvocationRecord, error)

	PutArtifact(context.Context, string, Artifact, json.RawMessage) (Artifact, error)
	GetArtifact(context.Context, string, string) (Artifact, error)
	PutCheckpoint(context.Context, string, Checkpoint) (Checkpoint, error)
	CommitCheckpoint(context.Context, TurnLease, Checkpoint) (Checkpoint, Event, error)
	LatestCheckpoint(context.Context, string, string) (Checkpoint, error)

	CreatePublicationPlan(context.Context, string, PublicationPlan, MutationContext) (PublicationPlan, error)
	GetPublicationPlan(context.Context, string, string) (PublicationPlan, error)
	GetPublicationModelIDs(context.Context, string, string) ([]string, error)
	ReservePublicationCommit(context.Context, string, string, string, int64, MutationContext) (PublicationCommitReservation, error)
	FinalizePublicationCommit(context.Context, string, string, string, int64, time.Time) (PublicationCommitResult, error)
	FailPublicationCommit(context.Context, string, string, string, time.Time) (PublicationCommitResult, error)

	Ready(context.Context) error
}

// TurnNotifier is acceleration only. PostgreSQL remains the durable queue and
// event authority when a notification is missed.
type TurnNotifier interface {
	Wake(context.Context, string, string) error
	NotifyEvents(context.Context, string, string, int64) error
	NotifyCancellation(context.Context, string, string) error
}

// LiveEventPublisher is an optional acceleration path for attached clients.
// PostgreSQL remains the only durable transcript. Publish failures must not
// fail or roll back a model step, and a replacement worker never replays these
// events.
type LiveEventPublisher interface {
	PublishLiveModelStep(context.Context, string, LiveModelStepEvent) error
}

// LiveEventSubscription owns one transient, best-effort delivery stream. A
// slow or disconnected consumer may miss previews and must reconcile from the
// durable session event stream.
type LiveEventSubscription interface {
	Events() <-chan LiveModelStepEvent
	Close() error
}

type LiveEventSubscriber interface {
	SubscribeLiveModelSteps(context.Context, string, string) (LiveEventSubscription, error)
}

type LiveEventBroker interface {
	LiveEventPublisher
	LiveEventSubscriber
}

type DelegatedSubject struct {
	NamespaceID string
	PrincipalID string
	UserID      string
	TeamID      string
	APIKeyID    string
	Status      string
	ExpiresAt   time.Time
}

type SessionBootstrapRequest struct {
	SessionID       string
	NamespaceID     string
	PrincipalID     string
	EffectiveTeamID string
	Profile         Profile
	Target          Target
	Mode            SessionMode
	Title           string
	SessionTTL      time.Duration
	Mutation        MutationContext
	Command         managementcommand.Command
}

// SessionAuthorizationRequest is a side-effect-free authorization preflight
// for the Management HTTP boundary. Bootstrap repeats the same resolution and
// access checks transactionally before minting the delegated credential.
type SessionAuthorizationRequest struct {
	NamespaceID     string
	PrincipalID     string
	EffectiveTeamID string
	Profile         Profile
	Target          Target
}

// SessionAuthorization contains only server-resolved scope operands. It never
// carries a delegated credential or exposes the backing API Key.
type SessionAuthorization struct {
	EffectiveUserID  string
	EffectiveTeamID  string
	TargetKind       TargetKind
	TargetResourceID string
}

// SessionAuthority is the only seam allowed to compose Management identity,
// effective User/Team attribution, public inference access, delegated
// credentials, target resolution, and the Agent session row. Bootstrap is one
// PostgreSQL transaction: a client can never observe a delegation without its
// session or bind a session to another subject's delegation.
type SessionAuthority interface {
	Prepare(context.Context, SessionAuthorizationRequest) (SessionAuthorization, error)
	Bootstrap(context.Context, SessionBootstrapRequest) (Session, bool, error)
	Reauthorize(context.Context, Session, []string) error
	RenewDelegation(context.Context, Session, time.Duration) error
	ResolveInferenceCredential(context.Context, Session) ([]byte, error)
	Close(context.Context, Session, int64, SessionPatch, MutationContext) (Session, error)
}

// TargetVisibility applies the same effective inference discovery policy used
// by /v1/models. It never distinguishes a missing target from a hidden one.
// Management response mappers use it before exposing an optional Profile
// default, while SessionAuthority repeats the check transactionally at create.
type TargetVisibility interface {
	CanDiscover(context.Context, string, string, Target) (bool, error)
}

type DefinitionValidator interface {
	ValidateProfile(context.Context, string, ProfileInput, *ToolRegistry) error
	ValidateSkill(context.Context, string, SkillInput, *ToolRegistry) error
}

// ToolSourcePolicyValidator is supplied by the remote Tool Source boundary.
// The Agent domain deliberately does not implement a second network policy;
// implementations must compile the source policy through the shared backend
// egress guard used at execution time.
type ToolSourcePolicyValidator interface {
	Normalize(ToolSourceInput) (ToolSourceInput, error)
}

type ToolSourceDiscoverer interface {
	Discover(context.Context, ToolSource) ([]ToolDefinition, string, error)
}

type SecretCodec interface {
	Encrypt(context.Context, []byte) (EncryptedSecret, error)
	Decrypt(context.Context, EncryptedSecret) ([]byte, error)
}
