package quotaruntime

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
)

var (
	ErrInvalidRequest     = errors.New("invalid quota runtime request")
	ErrConflict           = errors.New("quota runtime idempotency conflict")
	ErrEvidenceChanged    = errors.New("quota attempt evidence changed during settlement")
	ErrAdmissionNotFound  = errors.New("quota admission not found")
	ErrRuntimeCorrupt     = errors.New("quota runtime state is corrupt")
	ErrRuntimeUnavailable = errors.New("quota runtime is unavailable")
)

// Engine is the store-independent runtime seam used by an inference data
// plane. Implementations must make every method atomic within one Partition.
type Engine interface {
	AdmissionEngine
	AttemptEvidenceEngine
	SettlementEngine
}

type AdmissionEngine interface {
	CheckAccess(context.Context, AccessCheckRequest) (AccessCheckResult, error)
	Admit(context.Context, AdmissionRequest) (AdmissionResult, error)
	Heartbeat(context.Context, AdmissionHeartbeatRequest) (AdmissionHeartbeatResult, error)
	JournalDispatch(context.Context, DispatchJournalRequest) (MutationResult, error)
}

type SettlementEngine interface {
	Finalize(context.Context, FinalizationRequest) (FinalizationResult, error)
	ReleaseConcurrency(context.Context, ConcurrencyReleaseRequest) (MutationResult, error)
	ReadMeters(context.Context, MeterReadRequest) (MeterReadResult, error)
}

// AttemptEvidenceEngine is the narrow, cross-replica dispatch-attempt journal
// used by the backend invoker. Every implementation must validate the pending
// admission and immutable dispatch journal in the same atomic operation that
// mutates or reads attempt state.
type AttemptEvidenceEngine interface {
	BeginDispatch(context.Context, BeginDispatchRequest) (BeginDispatchResult, error)
	BeginAttempt(context.Context, BeginAttemptRequest) (BeginAttemptResult, error)
	FinishAttempt(context.Context, FinishAttemptRequest) (FinishAttemptResult, error)
	ReadAttemptEvidence(context.Context, ReadAttemptEvidenceRequest) (ReadAttemptEvidenceResult, error)
}

// AccessCheckRequest performs the same partition-local AuthN/AuthZ projection
// assertions as Admit without creating pending work or consuming quota. It is
// used by discovery endpoints such as /v1/models.
type AccessCheckRequest struct {
	Partition     string
	Preconditions []AdmissionPrecondition
}

type AccessCheckResult struct {
	Disposition AdmissionDisposition
	ServerTime  time.Time
	Reason      string
}

func (r AccessCheckResult) Allowed() bool {
	return r.Disposition == AdmissionAllowed
}

// RuleBinding pins one validated rule to the binding that owns its counter.
// Reusing a policy therefore never shares runtime state by accident.
type RuleBinding struct {
	BindingID        string
	Rule             quota.RateLimitRule
	Currency         string
	CalendarSchedule []CalendarInterval
}

// CalendarInterval is one contiguous, UTC boundary pair compiled by the
// control plane from an IANA timezone. Runtime scripts select an interval with
// Redis TIME and never interpret timezone rules themselves.
type CalendarInterval struct {
	Start time.Time
	End   time.Time
}

func (b RuleBinding) Validate() error {
	if err := validateOpaque("binding ID", b.BindingID); err != nil {
		return err
	}
	if err := b.Rule.Validate(); err != nil {
		return err
	}
	if b.Rule.Algorithm == quota.AlgorithmCalendarWindow {
		if err := validateCalendarSchedule(b.CalendarSchedule); err != nil {
			return err
		}
	} else if len(b.CalendarSchedule) != 0 {
		return fmt.Errorf(
			"%w: calendar schedule is valid only for calendar_window",
			ErrInvalidRequest,
		)
	}
	if b.Rule.Metric == quota.MetricCost {
		if !validCurrencyCode(b.Currency) {
			return fmt.Errorf("%w: cost rule requires a three-letter uppercase currency", ErrInvalidRequest)
		}
	} else if b.Currency != "" {
		return fmt.Errorf("%w: currency is valid only for cost", ErrInvalidRequest)
	}
	return nil
}

func (b RuleBinding) Counter() (quota.CounterIdentity, error) {
	return quota.NewCounterIdentity(b.BindingID, b.Rule.ID)
}

func (b RuleBinding) limit() quota.QuotaInteger {
	if b.Rule.Algorithm == quota.AlgorithmTokenBucket {
		return *b.Rule.BucketCapacity
	}
	if b.Rule.Algorithm == quota.AlgorithmGCRA {
		return quotaOne
	}
	if b.Rule.Metric == quota.MetricCost {
		return b.Rule.CostLimit.ScaledInteger()
	}
	return *b.Rule.WholeLimit
}

var quotaOne = func() quota.QuotaInteger {
	value, err := quota.ParseQuotaInteger("1")
	if err != nil {
		panic(err)
	}
	return value
}()

func validateCalendarSchedule(schedule []CalendarInterval) error {
	if len(schedule) == 0 {
		return fmt.Errorf("%w: calendar_window requires a compiled UTC schedule", ErrInvalidRequest)
	}
	for index, interval := range schedule {
		if interval.Start.IsZero() || interval.End.IsZero() || !interval.Start.Before(interval.End) {
			return fmt.Errorf("%w: calendar interval %d is empty or reversed", ErrInvalidRequest, index)
		}
		if interval.Start.Nanosecond()%int(time.Millisecond) != 0 ||
			interval.End.Nanosecond()%int(time.Millisecond) != 0 {
			return fmt.Errorf("%w: calendar interval %d is not millisecond-aligned", ErrInvalidRequest, index)
		}
		_, startOffset := interval.Start.Zone()
		_, endOffset := interval.End.Zone()
		if startOffset != 0 || endOffset != 0 {
			return fmt.Errorf("%w: calendar interval %d must use UTC boundaries", ErrInvalidRequest, index)
		}
		if index > 0 && !schedule[index-1].End.Equal(interval.Start) {
			return fmt.Errorf("%w: calendar schedule must be ordered and contiguous", ErrInvalidRequest)
		}
	}
	return nil
}

func (b RuleBinding) isResponseActual() bool {
	return b.Rule.Accounting == quota.AccountingResponseActual
}

func (b RuleBinding) isConcurrency() bool {
	return b.Rule.Algorithm == quota.AlgorithmConcurrency
}

// AdmissionRequest admits one bounded request against every applicable
// binding/rule counter. LeaseDuration is converted to a deadline with Redis
// TIME, never process-local time.
type AdmissionRequest struct {
	Partition     string
	AdmissionID   string
	Digest        string
	LeaseDuration time.Duration
	Preconditions []AdmissionPrecondition
	Rules         []RuleBinding
}

type AdmissionDisposition string

const (
	AdmissionAllowed         AdmissionDisposition = "allowed"
	AdmissionUnauthenticated AdmissionDisposition = "unauthenticated"
	AdmissionForbidden       AdmissionDisposition = "forbidden"
	AdmissionRateLimited     AdmissionDisposition = "rate_limited"
	AdmissionUnavailable     AdmissionDisposition = "unavailable"
)

type AdmissionResult struct {
	Disposition    AdmissionDisposition
	Idempotent     bool
	Limiting       *quota.CounterIdentity
	ServerTime     time.Time
	Deadline       time.Time
	RetryAt        *time.Time
	ResetAt        *time.Time
	BlockingReason string
	// PlanDigest binds follow-up heartbeats to the exact immutable admission
	// preconditions, lease, and quota rule plan accepted by the atomic store.
	PlanDigest string
}

func (r AdmissionResult) Allowed() bool {
	return r.Disposition == AdmissionAllowed
}

// AdmissionHeartbeatRequest renews only the liveness lease of one already
// admitted request. It cannot change quota rules, consume quota, or revive an
// expired admission. Rules are included solely to identify and atomically
// renew every concurrency counter owned by the admission.
type AdmissionHeartbeatRequest struct {
	Partition       string
	AdmissionID     string
	AdmissionDigest string
	PlanDigest      string
	LeaseDuration   time.Duration
	Rules           []RuleBinding
}

type AdmissionHeartbeatResult struct {
	ServerTime time.Time
	Deadline   time.Time
	// Stopped means the exact admission is already terminal. Callers must stop
	// their heartbeat loop and may treat this response as idempotent success.
	Stopped bool
}

// DispatchJournalRequest records one stable bounded backend dispatch before
// that dispatch starts. Digest covers route, attempt, and maximum-charge facts.
type DispatchJournalRequest struct {
	Partition       string
	AdmissionID     string
	AdmissionDigest string
	DispatchID      string
	Ordinal         uint32
	Digest          string
}

// DispatchReference identifies one immutable physical dispatch. The
// DispatchPlanDigest must equal the digest already stored by JournalDispatch;
// attempt operations never create or retarget a dispatch journal entry.
type DispatchReference struct {
	Partition          string
	AdmissionID        string
	AdmissionDigest    string
	DispatchID         string
	DispatchPlanDigest string
	ModelID            string
	ModelRevision      int64
	RequestDigest      string
}

// BeginDispatchRequest pins the bounded execution envelope before any backend
// attempt can start. Deadline is immutable and cannot extend the admission
// lease. MaxAttempts includes the initial attempt.
type BeginDispatchRequest struct {
	DispatchReference
	DispatchType string
	Ordinal      uint32
	Deadline     time.Time
	MaxAttempts  uint32
}

type BeginDispatchResult struct {
	MutationResult
	StartedAt time.Time
	Deadline  time.Time
}

// BeginAttemptRequest starts one 1-based attempt. AttemptNumber must be the
// next contiguous number and every preceding attempt must have completed as
// known_zero.
type BeginAttemptRequest struct {
	DispatchReference
	AttemptID     string
	AttemptNumber uint32
	BackendID     string
	ProviderID    string
}

type BeginAttemptResult struct {
	MutationResult
	StartedAt time.Time
}

// AttemptEvidenceState records what the Router can prove about one physical
// attempt. response_started is intentionally distinct from known actual usage;
// protocol finalization classifies authoritative usage separately.
type AttemptEvidenceState string

const (
	AttemptEvidenceKnownZero       AttemptEvidenceState = "known_zero"
	AttemptEvidenceResponseStarted AttemptEvidenceState = "response_started"
	AttemptEvidenceUnknown         AttemptEvidenceState = "unknown"
)

type FinishAttemptRequest struct {
	DispatchReference
	AttemptID     string
	AttemptNumber uint32
	BackendID     string
	ProviderID    string
	State         AttemptEvidenceState
	StatusCode    int
	ErrorCode     string
}

type FinishAttemptResult struct {
	MutationResult
	CompletedAt time.Time
}

type ReadAttemptEvidenceRequest struct {
	AttemptEvidenceReference
}

// AttemptEvidenceReference identifies an already-journaled dispatch for a
// read-only settlement observation. The exact provider request digest remains
// pinned in Redis and is returned with the evidence; unlike attempt mutations,
// a read cannot authorize or retarget backend work.
type AttemptEvidenceReference struct {
	Partition          string
	AdmissionID        string
	AdmissionDigest    string
	DispatchID         string
	Ordinal            uint32
	DispatchPlanDigest string
	ModelID            string
	ModelRevision      int64
}

type AttemptEvidence struct {
	AttemptID     string
	AttemptNumber uint32
	BackendID     string
	ProviderID    string
	State         AttemptEvidenceState
	StatusCode    int
	ErrorCode     string
	StartedAt     time.Time
	CompletedAt   time.Time
	// Finished is false only when a replica disappeared after BeginAttempt.
	// Such a record is surfaced conservatively as unknown.
	Finished bool
}

type DispatchAttemptEvidence struct {
	DispatchID         string
	DispatchType       string
	Ordinal            uint32
	DispatchPlanDigest string
	ModelID            string
	ModelRevision      int64
	RequestDigest      string
	StartedAt          time.Time
	Deadline           time.Time
	MaxAttempts        uint32
	Attempts           []AttemptEvidence
}

type ReadAttemptEvidenceResult struct {
	Present    bool
	Revision   uint64
	Evidence   DispatchAttemptEvidence
	ServerTime time.Time
}

type ActualEvidenceState string

const (
	ActualEvidenceKnown   ActualEvidenceState = "known"
	ActualEvidenceUnknown ActualEvidenceState = "unknown"
)

// ActualEvidence explicitly classifies each response-actual counter. Known
// zero is distinct from unknown: only unknown evidence contributes incomplete
// usage and can open an enforce-binding fence.
type ActualEvidence struct {
	State  ActualEvidenceState
	Amount quota.QuotaInteger
	Reason string
}

// FinalizationRequest closes one pending admission in one atomic operation.
// DispatchCount and EvidenceRevision compare-and-set the authoritative attempt
// snapshot used to construct Event; counters, the terminal marker, the event,
// and attempt-journal cleanup therefore commit against the same evidence.
type FinalizationRequest struct {
	Partition          string
	AdmissionID        string
	AdmissionDigest    string
	FinalizationDigest string
	DispatchCount      uint32
	EvidenceRevision   uint64
	Event              string
	FenceID            string
	Rules              []RuleBinding
	Evidence           map[quota.CounterIdentity]ActualEvidence
}

type FinalizationResult struct {
	MutationResult
	EvidenceState string
	StreamID      string
}

type ConcurrencyReleaseRequest struct {
	Partition       string
	AdmissionID     string
	AdmissionDigest string
	Rules           []RuleBinding
}

type MeterReadRequest struct {
	Partition string
	Rules     []RuleBinding
}

type MutationResult struct {
	Idempotent bool
	ServerTime time.Time
}

// Meter enriches the precision-safe public quantity shape with immutable rule
// semantics and the live reset instant returned by the same atomic read.
type Meter struct {
	quota.PublicMeter
	Algorithm      quota.Algorithm  `json:"algorithm"`
	Accounting     quota.Accounting `json:"accounting"`
	ResetAt        *time.Time       `json:"resetAt"`
	ActiveFenceIDs []string         `json:"activeFenceIds"`
}

type MeterReadResult struct {
	Meters []Meter   `json:"meters"`
	AsOf   time.Time `json:"asOf"`
}

func validateAdmissionRequest(request AdmissionRequest) ([]compiledRule, error) {
	return validateAdmissionRequestWithPrefix("", request)
}

func validateAdmissionRequestWithPrefix(prefix string, request AdmissionRequest) ([]compiledRule, error) {
	if err := validateEnvelope(request.Partition, request.AdmissionID, request.Digest); err != nil {
		return nil, err
	}
	if request.LeaseDuration <= 0 || request.LeaseDuration%time.Millisecond != 0 {
		return nil, fmt.Errorf("%w: lease duration must be a positive whole number of milliseconds", ErrInvalidRequest)
	}
	if len(request.Preconditions) == 0 {
		return nil, fmt.Errorf(
			"%w: final admission requires atomic access preconditions",
			ErrInvalidRequest,
		)
	}
	for index, precondition := range request.Preconditions {
		if err := precondition.Validate(); err != nil {
			return nil, fmt.Errorf("precondition %d: %w", index, err)
		}
	}
	return compileRulesWithPrefix(prefix, request.Partition, request.Rules)
}

func validateEnvelope(partition, admissionID, digest string) error {
	if err := validatePartition(partition); err != nil {
		return err
	}
	if err := validateOpaque("admission ID", admissionID); err != nil {
		return err
	}
	if err := validateDigest("digest", digest); err != nil {
		return err
	}
	return nil
}

func validateOpaque(label, value string) error {
	if value == "" {
		return fmt.Errorf("%w: %s is required", ErrInvalidRequest, label)
	}
	if strings.TrimSpace(value) != value || strings.ContainsRune(value, '\x00') {
		return fmt.Errorf("%w: %s has invalid whitespace or NUL", ErrInvalidRequest, label)
	}
	return nil
}

func validateDigest(label, value string) error {
	if err := validateOpaque(label, value); err != nil {
		return err
	}
	if len(value) > 512 {
		return fmt.Errorf("%w: %s is too long", ErrInvalidRequest, label)
	}
	return nil
}

func validCurrencyCode(value string) bool {
	if len(value) != 3 {
		return false
	}
	for index := range value {
		if value[index] < 'A' || value[index] > 'Z' {
			return false
		}
	}
	return true
}
