// Package quotareconciliation owns the durable unknown-usage settlement saga.
// It deliberately contains no HTTP, Dashboard, or process-local state.
package quotareconciliation

import (
	"context"
	"errors"
	"net/netip"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
)

const OperationKind = "unknown_usage_fence.reconcile"

var (
	ErrInvalidRequest         = errors.New("invalid unknown-usage reconciliation request")
	ErrNotFound               = errors.New("unknown-usage fence not found")
	ErrRevisionConflict       = errors.New("unknown-usage fence revision conflict")
	ErrReconciliationConflict = errors.New("unknown-usage fence has another reconciliation plan")
	ErrResolved               = errors.New("unknown-usage fence is resolved")
	ErrEvidenceConflict       = errors.New("unknown-usage evidence conflicts with the immutable ledger")
	ErrWaiveDenied            = errors.New("unknown-usage waiver authentication requirement was not met")
	ErrLeaseLost              = errors.New("unknown-usage reconciliation lease was lost")
	ErrUnavailable            = errors.New("unknown-usage reconciliation is unavailable")
)

type Strategy string

const (
	StrategyActual            Strategy = "actual"
	StrategyConservativeDebit Strategy = "conservative_debit"
	StrategyWaive             Strategy = "waive"
)

func (strategy Strategy) Valid() bool {
	return strategy == StrategyActual || strategy == StrategyConservativeDebit || strategy == StrategyWaive
}

type FenceState string

const (
	FenceOpen        FenceState = "open"
	FenceReconciling FenceState = "reconciling"
	FenceResolved    FenceState = "resolved"
)

type Subject struct {
	Kind accesscontrol.SubjectKind `json:"kind"`
	ID   string                    `json:"id"`
}

type Binding struct {
	BindingID              string               `json:"bindingId"`
	RuleID                 string               `json:"ruleId"`
	PolicyID               string               `json:"policyId"`
	Subject                Subject              `json:"subject"`
	Metric                 quota.Metric         `json:"metric"`
	Algorithm              quota.Algorithm      `json:"algorithm"`
	Enforcement            quota.Enforcement    `json:"enforcement"`
	AdmissionLimit         string               `json:"admissionLimit,omitempty"`
	MaximumDebit           string               `json:"maximumDebit,omitempty"`
	Window                 time.Duration        `json:"window,omitempty"`
	CalendarPeriod         quota.CalendarPeriod `json:"calendarPeriod,omitempty"`
	Timezone               string               `json:"timezone,omitempty"`
	Currency               string               `json:"currency,omitempty"`
	UnknownDispatchCount   string               `json:"unknownDispatchCount"`
	CounterIncompleteCount string               `json:"counterIncompleteCount"`
}

type Cost struct {
	Currency  string `json:"currency"`
	Numerator string `json:"numerator"`
}

type Charge struct {
	InputTokens  string `json:"inputTokens"`
	OutputTokens string `json:"outputTokens"`
	TotalTokens  string `json:"totalTokens"`
	Costs        []Cost `json:"costs"`
}

type UnknownDispatch struct {
	DispatchID      string `json:"dispatchId"`
	EvidenceDigest  string `json:"evidenceDigest"`
	Reason          string `json:"reason"`
	ModelID         string `json:"modelId,omitempty"`
	BackendID       string `json:"backendId,omitempty"`
	ProviderID      string `json:"providerId,omitempty"`
	ProviderModelID string `json:"providerModelId,omitempty"`
	PricingRevision int64  `json:"pricingRevision,omitempty"`
}

type ReconciliationInfo struct {
	ID        string     `json:"reconciliationId"`
	Strategy  Strategy   `json:"strategy"`
	ActorID   string     `json:"actorPrincipalId"`
	Reason    string     `json:"reason"`
	CreatedAt time.Time  `json:"createdAt"`
	AppliedAt *time.Time `json:"appliedAt,omitempty"`
}

type Fence struct {
	ID             string              `json:"fenceId"`
	NamespaceID    string              `json:"namespaceId"`
	AdmissionID    string              `json:"admissionId"`
	State          FenceState          `json:"state"`
	Revision       uint64              `json:"revision"`
	Reason         string              `json:"reason"`
	Bindings       []Binding           `json:"bindings"`
	KnownCharge    Charge              `json:"knownCharge"`
	Unknown        []UnknownDispatch   `json:"unknownDispatches,omitempty"`
	Reconciliation *ReconciliationInfo `json:"reconciliation,omitempty"`
	CreatedAt      time.Time           `json:"createdAt"`
	UpdatedAt      time.Time           `json:"updatedAt"`
	ResolvedAt     *time.Time          `json:"resolvedAt,omitempty"`
}

type ActualDispatchUsage struct {
	DispatchID       string `json:"dispatchId"`
	EvidenceDigest   string `json:"evidenceDigest"`
	InputTokens      string `json:"inputTokens"`
	CacheReadTokens  string `json:"cacheReadTokens"`
	CacheWriteTokens string `json:"cacheWriteTokens"`
	OutputTokens     string `json:"outputTokens"`
	Cost             Cost   `json:"cost"`
}

type ActualUsage struct {
	Dispatches         []ActualDispatchUsage `json:"dispatches"`
	ServedInputTokens  string                `json:"servedInputTokens"`
	ServedOutputTokens string                `json:"servedOutputTokens"`
}

type Actor struct {
	PrincipalID string     `json:"principalId"`
	ActorChain  []string   `json:"actorChain"`
	RequestID   string     `json:"requestId"`
	SourceIP    netip.Addr `json:"sourceIp,omitempty"`
}

type ReconcileRequest struct {
	NamespaceID        string
	FenceID            string
	ExpectedRevision   uint64
	IdempotencyKey     string
	Strategy           Strategy
	Actual             *ActualUsage
	EvidenceReferences []string
	Reason             string
	Actor              Actor
	Session            managementauth.LiveSession
}

type OperationState string

const (
	OperationPending   OperationState = "pending"
	OperationRunning   OperationState = "running"
	OperationSucceeded OperationState = "succeeded"
)

type Operation struct {
	ID                string
	NamespaceID       string
	FenceID           string
	Kind              string
	OriginPrincipalID string
	ActorChain        []string
	Version           uint64
	State             OperationState
	Completed         uint64
	Total             uint64
	CreatedAt         time.Time
	UpdatedAt         time.Time
	CompletedAt       *time.Time
}

type EnqueueResult struct {
	Operation Operation
	Replayed  bool
}

type Cursor struct {
	CreatedAt time.Time
	ID        string
}

type FenceQuery struct {
	NamespaceID string
	State       FenceState
	Scope       accesscontrol.ResultScope
	After       *Cursor
	Limit       int
}

type RepositoryPage struct {
	Items   []Fence
	HasMore bool
}

type Page struct {
	Items      []Fence
	NextCursor string
	HasMore    bool
	PageSize   int
}

type ListRequest struct {
	NamespaceID string
	State       FenceState
	Scope       accesscontrol.ResultScope
	Cursor      string
	PageSize    int
}

type Phase string

const (
	PhaseRuntimePending Phase = "runtime_pending"
	PhaseRuntimeApplied Phase = "runtime_applied"
	PhaseLedgerApplied  Phase = "ledger_applied"
	PhaseCompleted      Phase = "completed"
)

type CorrectionDispatch struct {
	DispatchID         string    `json:"dispatchId"`
	CorrectsDispatchID string    `json:"correctsDispatchId"`
	Ordinal            int       `json:"ordinal"`
	DispatchType       string    `json:"dispatchType"`
	ModelID            string    `json:"modelId,omitempty"`
	ModelRevision      int64     `json:"modelRevision,omitempty"`
	BackendID          string    `json:"backendId,omitempty"`
	ProviderID         string    `json:"providerId,omitempty"`
	ProviderModelID    string    `json:"providerModelId,omitempty"`
	PricingRevision    int64     `json:"pricingRevision,omitempty"`
	InputTokens        string    `json:"inputTokens"`
	CacheReadTokens    string    `json:"cacheReadTokens"`
	CacheWriteTokens   string    `json:"cacheWriteTokens"`
	OutputTokens       string    `json:"outputTokens"`
	Cost               Cost      `json:"cost"`
	EvidenceDigest     string    `json:"evidenceDigest"`
	StartedAt          time.Time `json:"startedAt"`
	CompletedAt        time.Time `json:"completedAt"`
}

// Plan is immutable after insertion. Only its phase, lease, and runtime stream
// marker change during recovery.
type Plan struct {
	ReconciliationID     string                           `json:"reconciliationId"`
	NamespaceID          string                           `json:"namespaceId"`
	Partition            string                           `json:"partition"`
	FenceID              string                           `json:"fenceId"`
	AdmissionID          string                           `json:"admissionId"`
	OriginalEventID      string                           `json:"originalEventId"`
	CorrectionEventID    string                           `json:"correctionEventId"`
	OperationID          string                           `json:"operationId"`
	Strategy             Strategy                         `json:"strategy"`
	Reason               string                           `json:"reason"`
	Actor                Actor                            `json:"actor"`
	EvidenceReferences   []string                         `json:"evidenceReferences,omitempty"`
	Corrections          []quotaruntime.CounterCorrection `json:"corrections"`
	Dispatches           []CorrectionDispatch             `json:"dispatches"`
	UnknownDispatchCount string                           `json:"unknownDispatchCount"`
	CorrectionCharge     Charge                           `json:"correctionCharge"`
	ServedInputTokens    string                           `json:"servedInputTokens"`
	ServedOutputTokens   string                           `json:"servedOutputTokens"`
	RequestSnapshot      RequestSnapshot                  `json:"requestSnapshot"`
	CreatedAt            time.Time                        `json:"createdAt"`
}

type RequestSnapshot struct {
	ExternalRequestID string    `json:"externalRequestId,omitempty"`
	Protocol          string    `json:"protocol"`
	Path              string    `json:"path"`
	APIKeyID          string    `json:"apiKeyId,omitempty"`
	CredentialID      string    `json:"credentialId,omitempty"`
	UserID            string    `json:"userId,omitempty"`
	TeamID            string    `json:"teamId,omitempty"`
	EntrypointID      string    `json:"entrypointId,omitempty"`
	EntrypointRuleID  string    `json:"entrypointRuleId,omitempty"`
	RecipeID          string    `json:"recipeId,omitempty"`
	RoutingRevision   int64     `json:"routingRevision,omitempty"`
	StatusCode        int       `json:"statusCode"`
	ErrorCode         string    `json:"errorCode,omitempty"`
	OccurredAt        time.Time `json:"occurredAt"`
	CompletedAt       time.Time `json:"completedAt"`
}

type Claim struct {
	Plan            Plan
	PlanDigest      string
	Phase           Phase
	RuntimeStreamID string
	Attempt         int
	LeaseOwner      string
	LeaseToken      string
	LeaseExpiresAt  time.Time
}

type Repository interface {
	ReconciliationStateRepository
	ReconciliationCommandRepository
	ReconciliationWorkerRepository
}

type ReconciliationStateRepository interface {
	ReadyQuotaReconciliation(context.Context, *managementcommand.Codec) error
	Get(context.Context, string, string) (Fence, error)
	GetOperation(context.Context, string, string) (Operation, error)
	List(context.Context, FenceQuery) (RepositoryPage, error)
}

type ReconciliationCommandRepository interface {
	Prepare(context.Context, managementcommand.Command, ReconcileRequest, string, time.Time) (EnqueueResult, error)
}

type ReconciliationWorkerRepository interface {
	Claim(context.Context, string, time.Time, time.Duration) (Claim, bool, error)
	MarkRuntimeApplied(context.Context, Claim, string, time.Time) error
	SettleLedger(context.Context, Claim, time.Time) error
	Complete(context.Context, Claim, time.Time) (Operation, error)
	Release(context.Context, Claim, time.Time, error) error
}

type WaiveAuthenticator interface {
	AuthorizeWaive(context.Context, string, managementauth.LiveSession, time.Time) error
}

type Runtime interface {
	ApplyReconciliation(context.Context, quotaruntime.ReconciliationRequest) (quotaruntime.ReconciliationResult, error)
	RemoveReconciledFence(context.Context, quotaruntime.FenceRemovalRequest) (quotaruntime.MutationResult, error)
}
