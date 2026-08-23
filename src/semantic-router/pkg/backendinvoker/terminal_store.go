package backendinvoker

import (
	"container/heap"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const (
	defaultTerminalCapacity = 65_536
	defaultTerminalTTL      = 10 * time.Minute
)

var (
	ErrResponseTerminalInvalid     = errors.New("response terminal is invalid")
	ErrResponseTerminalDuplicate   = errors.New("response terminal already exists")
	ErrResponseTerminalCapacity    = errors.New("response terminal store capacity is exhausted")
	ErrResponseTerminalUnavailable = errors.New("response terminal store is unavailable")
)

// ResponseTerminalReference is the immutable, credential-free identity shared
// by the signed dispatch capability, its resolved Plan, and the ExtProc request
// that is allowed to consume the resulting terminal. It deliberately excludes
// request bodies, headers, and provider details.
type ResponseTerminalReference struct {
	NamespaceID        string `json:"namespaceId"`
	QuotaPartition     string `json:"quotaPartition"`
	PublicationID      string `json:"publicationId"`
	RuntimeEpoch       uint64 `json:"runtimeEpoch"`
	RoutingRevision    int64  `json:"routingRevision"`
	RoutingDigest      string `json:"routingDigest"`
	AdmissionID        string `json:"admissionId"`
	AdmissionDigest    string `json:"admissionDigest"`
	RequestID          string `json:"requestId"`
	DispatchID         string `json:"dispatchId"`
	DispatchType       string `json:"dispatchType"`
	Ordinal            int    `json:"ordinal"`
	Priority           int    `json:"priority"`
	DispatchPlanDigest string `json:"dispatchPlanDigest"`
	ModelID            string `json:"modelId"`
	ModelRevision      int64  `json:"modelRevision"`
}

// ResponseTerminalReferenceFromPlan derives the only terminal reference a
// physical Plan may finalize.
func ResponseTerminalReferenceFromPlan(plan Plan) (ResponseTerminalReference, error) {
	reference := ResponseTerminalReference{
		NamespaceID: plan.NamespaceID, QuotaPartition: plan.QuotaPartition,
		PublicationID: plan.PublicationID, RuntimeEpoch: plan.RuntimeEpoch,
		RoutingRevision: plan.RoutingRevision, RoutingDigest: plan.RoutingDigest,
		AdmissionID: plan.AdmissionID, AdmissionDigest: plan.AdmissionDigest,
		RequestID: plan.RequestID, DispatchID: plan.DispatchID,
		DispatchType: plan.DispatchType, Ordinal: plan.Ordinal, Priority: plan.Priority,
		DispatchPlanDigest: plan.DispatchPlanDigest,
		ModelID:            plan.ModelID, ModelRevision: plan.ModelRevision,
	}
	if err := reference.Validate(); err != nil {
		return ResponseTerminalReference{}, err
	}
	return reference, nil
}

func (reference ResponseTerminalReference) Validate() error {
	if !validBoundedIdentity(reference.NamespaceID) ||
		!validBoundedIdentity(reference.QuotaPartition) ||
		!validBoundedIdentity(reference.PublicationID) ||
		reference.RuntimeEpoch == 0 || reference.RoutingRevision <= 0 ||
		!validSHA256Hex(reference.RoutingDigest) ||
		!validBoundedIdentity(reference.AdmissionID) ||
		!validSHA256Hex(reference.AdmissionDigest) ||
		!validBoundedIdentity(reference.RequestID) ||
		!validBoundedIdentity(reference.DispatchID) ||
		!validBoundedIdentity(reference.DispatchType) ||
		reference.Ordinal < 0 || uint64(reference.Ordinal) > uint64(^uint32(0)) ||
		reference.Priority < 0 || reference.Priority > 31 ||
		!validSHA256Hex(reference.DispatchPlanDigest) ||
		!validBoundedIdentity(reference.ModelID) || reference.ModelRevision <= 0 {
		return fmt.Errorf("%w: immutable terminal reference is incomplete", ErrResponseTerminalInvalid)
	}
	return nil
}

func (reference ResponseTerminalReference) digest() (string, error) {
	if err := reference.Validate(); err != nil {
		return "", err
	}
	hash := sha256.New()
	writeTerminalDigestString(hash, "vllm-sr/response-terminal-reference/v1")
	for _, value := range []string{
		reference.NamespaceID, reference.QuotaPartition, reference.PublicationID,
		reference.RoutingDigest, reference.AdmissionID, reference.AdmissionDigest,
		reference.RequestID, reference.DispatchID, reference.DispatchType,
		reference.DispatchPlanDigest, reference.ModelID,
	} {
		writeTerminalDigestString(hash, value)
	}
	for _, value := range []uint64{
		reference.RuntimeEpoch, uint64(reference.RoutingRevision), uint64(reference.Ordinal),
		uint64(reference.Priority), uint64(reference.ModelRevision),
	} {
		var encoded [8]byte
		binary.BigEndian.PutUint64(encoded[:], value)
		_, _ = hash.Write(encoded[:])
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}

type terminalDigestWriter interface {
	Write([]byte) (int, error)
}

func writeTerminalDigestString(hash terminalDigestWriter, value string) {
	var size [8]byte
	binary.BigEndian.PutUint64(size[:], uint64(len(value)))
	_, _ = hash.Write(size[:])
	_, _ = hash.Write([]byte(value))
}

// ResponseTerminalRecord binds one semantic terminal to the immutable attempt
// that produced it. It contains no wire body, credential, or source envelope.
type ResponseTerminalRecord struct {
	Reference ResponseTerminalReference
	Attempt   AttemptResult
	Terminal  ResponseTerminal
}

// ResponseTerminalReader destructively consumes a terminal using the exact
// immutable dispatch reference that created it.
type ResponseTerminalReader interface {
	Take(context.Context, ResponseTerminalReference) (ResponseTerminalRecord, bool, error)
}

// ResponseTerminalStore is the complete finalizer/reader rendezvous contract.
// Managed mode implements it with one shared Valkey store; standalone mode uses
// the bounded process-local implementation below.
type ResponseTerminalStore interface {
	ResponseFinalizer
	ResponseTerminalReader
}

// LocalResponseTerminalStore is the bounded, expiring standalone rendezvous.
type LocalResponseTerminalStore struct {
	mu       sync.Mutex
	records  map[string]*terminalEntry
	expiry   terminalExpiryHeap
	capacity int
	ttl      time.Duration
	now      func() time.Time
}

type terminalEntry struct {
	key       string
	record    ResponseTerminalRecord
	expiresAt time.Time
	index     int
}

type terminalExpiryHeap []*terminalEntry

func (values terminalExpiryHeap) Len() int { return len(values) }
func (values terminalExpiryHeap) Less(left, right int) bool {
	return values[left].expiresAt.Before(values[right].expiresAt)
}

func (values terminalExpiryHeap) Swap(left, right int) {
	values[left], values[right] = values[right], values[left]
	values[left].index, values[right].index = left, right
}

func (values *terminalExpiryHeap) Push(value any) {
	entry := value.(*terminalEntry)
	entry.index = len(*values)
	*values = append(*values, entry)
}

func (values *terminalExpiryHeap) Pop() any {
	old := *values
	last := len(old) - 1
	entry := old[last]
	old[last] = nil
	entry.index = -1
	*values = old[:last]
	return entry
}

var _ ResponseTerminalStore = (*LocalResponseTerminalStore)(nil)

func NewLocalResponseTerminalStore() *LocalResponseTerminalStore {
	return &LocalResponseTerminalStore{
		records:  make(map[string]*terminalEntry),
		expiry:   make(terminalExpiryHeap, 0),
		capacity: defaultTerminalCapacity,
		ttl:      defaultTerminalTTL,
		now:      time.Now,
	}
}

func (store *LocalResponseTerminalStore) Finalize(
	_ context.Context,
	plan Plan,
	attempt AttemptResult,
	terminal ResponseTerminal,
) error {
	if store == nil {
		return ErrResponseTerminalUnavailable
	}
	reference, err := ResponseTerminalReferenceFromPlan(plan)
	if err != nil {
		return err
	}
	key, err := reference.digest()
	if err != nil {
		return err
	}
	if err := validateResponseTerminalRecord(reference, attempt, terminal); err != nil {
		return err
	}
	now := store.now().UTC()
	record := ResponseTerminalRecord{
		Reference: reference, Attempt: attempt,
		Terminal: cloneResponseTerminal(terminal),
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	store.pruneLocked(now)
	if _, duplicate := store.records[key]; duplicate {
		return ErrResponseTerminalDuplicate
	}
	if len(store.records) >= store.capacity {
		return fmt.Errorf("%w: local store is full", ErrResponseTerminalCapacity)
	}
	entry := &terminalEntry{key: key, record: record, expiresAt: now.Add(store.ttl), index: -1}
	store.records[key] = entry
	heap.Push(&store.expiry, entry)
	return nil
}

func (store *LocalResponseTerminalStore) Take(
	_ context.Context,
	reference ResponseTerminalReference,
) (ResponseTerminalRecord, bool, error) {
	if store == nil {
		return ResponseTerminalRecord{}, false, ErrResponseTerminalUnavailable
	}
	key, err := reference.digest()
	if err != nil {
		return ResponseTerminalRecord{}, false, err
	}
	now := store.now().UTC()
	store.mu.Lock()
	defer store.mu.Unlock()
	store.pruneLocked(now)
	entry, found := store.records[key]
	if !found {
		return ResponseTerminalRecord{}, false, nil
	}
	delete(store.records, key)
	heap.Remove(&store.expiry, entry.index)
	if entry.record.Reference != reference {
		return ResponseTerminalRecord{}, false, fmt.Errorf("%w: local reference mismatch", ErrResponseTerminalUnavailable)
	}
	return entry.record, true, nil
}

func (store *LocalResponseTerminalStore) pruneLocked(now time.Time) {
	for len(store.expiry) > 0 && !store.expiry[0].expiresAt.After(now) {
		entry := heap.Pop(&store.expiry).(*terminalEntry)
		delete(store.records, entry.key)
	}
}

func validateResponseTerminalRecord(
	reference ResponseTerminalReference,
	attempt AttemptResult,
	terminal ResponseTerminal,
) error {
	if err := reference.Validate(); err != nil {
		return err
	}
	if !validBoundedIdentity(attempt.ID) || attempt.Number < 1 || attempt.Number > 6 ||
		!validBoundedIdentity(attempt.BackendID) ||
		(attempt.State != AttemptKnownZero && attempt.State != AttemptResponseStarted && attempt.State != AttemptUnknown) ||
		attempt.StatusCode < 0 || attempt.StatusCode > 599 ||
		!boundedOptionalIdentity(attempt.ErrorCode, 256) ||
		(attempt.FallbackTrigger != "" && fallbackTriggerOrder(attempt.FallbackTrigger) < 0) {
		return fmt.Errorf("%w: attempt evidence is incomplete", ErrResponseTerminalInvalid)
	}
	if (!attempt.StartedAt.IsZero() && attempt.StartedAt.UnixNano() <= 0) ||
		(!attempt.CompletedAt.IsZero() && attempt.CompletedAt.UnixNano() <= 0) ||
		(!attempt.StartedAt.IsZero() && !attempt.CompletedAt.IsZero() &&
			attempt.CompletedAt.Before(attempt.StartedAt)) {
		return fmt.Errorf("%w: attempt timestamps are invalid", ErrResponseTerminalInvalid)
	}
	return validateResponseTerminal(terminal)
}

func validateResponseTerminal(terminal ResponseTerminal) error {
	if terminal.Error != nil {
		if terminal.Usage.State != llmprotocol.UsageUnavailable || terminal.StopReason != llmprotocol.StopError {
			return fmt.Errorf("%w: failed terminal must carry unknown usage and an error stop reason", ErrResponseTerminalInvalid)
		}
		if !validTerminalErrorCategory(terminal.Error.Category) ||
			!boundedOptionalIdentity(terminal.Error.Code, 256) ||
			!boundedOptionalIdentity(terminal.Error.Message, 4096) ||
			!boundedOptionalIdentity(terminal.Error.Parameter, 256) ||
			terminal.Error.RetryAfter < 0 || terminal.Error.RetryAfter > 86_400 {
			return fmt.Errorf("%w: protocol error is invalid", ErrResponseTerminalInvalid)
		}
		return nil
	}
	if !validSuccessfulTerminalStopReason(terminal.StopReason) {
		return fmt.Errorf("%w: successful terminal has an invalid stop reason", ErrResponseTerminalInvalid)
	}
	if err := llmprotocol.ValidateUsage(terminal.Usage); err != nil {
		return fmt.Errorf("%w: %v", ErrResponseTerminalInvalid, err)
	}
	return nil
}

func validTerminalErrorCategory(category llmprotocol.ErrorCategory) bool {
	switch category {
	case llmprotocol.ErrorInvalidRequest,
		llmprotocol.ErrorAuthentication,
		llmprotocol.ErrorPermission,
		llmprotocol.ErrorNotFound,
		llmprotocol.ErrorConflict,
		llmprotocol.ErrorUnsupportedFeature,
		llmprotocol.ErrorRateLimited,
		llmprotocol.ErrorUpstreamUnavailable,
		llmprotocol.ErrorUpstreamTimeout,
		llmprotocol.ErrorInternal:
		return true
	default:
		return false
	}
}

func validSuccessfulTerminalStopReason(reason llmprotocol.StopReason) bool {
	switch reason {
	case llmprotocol.StopEndTurn,
		llmprotocol.StopMaxTokens,
		llmprotocol.StopSequence,
		llmprotocol.StopToolCall,
		llmprotocol.StopContentFilter,
		llmprotocol.StopCanceled,
		llmprotocol.StopUnknown:
		return true
	default:
		return false
	}
}

func cloneResponseTerminal(source ResponseTerminal) ResponseTerminal {
	result := source
	if source.Error != nil {
		cloned := *source.Error
		cloned.Cause = nil
		result.Error = &cloned
	}
	return result
}
