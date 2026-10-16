package store

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"sync"
	"time"
)

const (
	DefaultMaxRecords = 200
)

// MemoryStore implements an in-memory ring buffer for routing records.
// Records are stored in a circular buffer with a maximum capacity.
// When the buffer is full, the oldest record is evicted.
type MemoryStore struct {
	mu sync.RWMutex

	records    []*Record
	byID       map[string]*Record
	maxRecords int
	// ttl bounds record retention in seconds. Records older than ttl are
	// elided from List/Get and purged on the next mutation. 0 (or negative)
	// disables time-based expiry and keeps only the maxRecords ring bound.
	ttl int
}

// NewMemoryStore creates a new in-memory storage backend.
func NewMemoryStore(maxRecords int, ttlSeconds int) *MemoryStore {
	if maxRecords <= 0 {
		maxRecords = DefaultMaxRecords
	}

	return &MemoryStore{
		records:    make([]*Record, 0, maxRecords),
		byID:       make(map[string]*Record),
		maxRecords: maxRecords,
		ttl:        ttlSeconds,
	}
}

// Add inserts a new record into memory storage.
func (m *MemoryStore) Add(ctx context.Context, record Record) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.purgeExpiredLocked(time.Now())

	if record.ID == "" {
		id, err := generateID()
		if err != nil {
			return "", fmt.Errorf("failed to generate ID: %w", err)
		}
		record.ID = id
	}
	if record.LifecycleState == "" {
		record.LifecycleState = LifecycleInProgress
	}
	if record.Timestamp.IsZero() {
		record.Timestamp = time.Now().UTC()
	}

	// Evict oldest if at capacity. Clear the evicted slot so the
	// backing array does not retain a reference to the dropped Record,
	// allowing GC to reclaim its payload immediately.
	if len(m.records) >= m.maxRecords {
		oldest := m.records[0]
		delete(m.byID, oldest.ID)
		m.records[0] = nil
		m.records = m.records[1:]
	}

	// Make a copy to avoid external mutations
	copyRec := cloneRecord(record)
	m.records = append(m.records, &copyRec)
	m.byID[copyRec.ID] = &copyRec

	return copyRec.ID, nil
}

// Get retrieves a record by ID from memory. Expired records are elided.
func (m *MemoryStore) Get(ctx context.Context, id string) (Record, bool, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.purgeExpiredLocked(time.Now())

	rec, ok := m.byID[id]
	if !ok {
		return Record{}, false, nil
	}

	return cloneRecord(*rec), true, nil
}

// List returns all records ordered by timestamp descending. Expired records
// are elided so the caller never observes past-retention entries.
func (m *MemoryStore) List(ctx context.Context) ([]Record, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.purgeExpiredLocked(time.Now())

	result := make([]Record, len(m.records))
	for i := len(m.records) - 1; i >= 0; i-- {
		result[len(m.records)-1-i] = cloneRecord(*m.records[i])
	}

	return result, nil
}

// purgeExpiredLocked drops records older than the configured retention ttl.
// It is a no-op when ttl <= 0, in which case only the maxRecords ring bound
// applies. Callers must hold m.mu.
func (m *MemoryStore) purgeExpiredLocked(now time.Time) {
	if m.ttl <= 0 || len(m.records) == 0 {
		return
	}
	cutoff := now.Add(-time.Duration(m.ttl) * time.Second)

	// Re-slice in place so we preserve the original capacity, but nil out
	// every slot that we skip. Go keeps the backing array alive for the full
	// capacity even after the length shortens, so we must clear the stale
	// pointers explicitly or GC cannot reclaim the expired Record payloads.
	kept := m.records[:0]
	for _, rec := range m.records {
		ts := rec.Timestamp
		if ts.IsZero() {
			// A record with no timestamp is treated as fresh; it is only
			// bounded by the maxRecords ring eviction, never by TTL.
			kept = append(kept, rec)
			continue
		}
		if ts.Before(cutoff) {
			delete(m.byID, rec.ID)
			continue
		}
		kept = append(kept, rec)
	}
	// Nil the trailing slots in the original backing array so stale
	// *Record pointers don't keep expired payloads reachable.
	for i := len(kept); i < len(m.records); i++ {
		m.records[i] = nil
	}
	m.records = kept
}

// UpdateStatus updates the response status and flags for a record.
func (m *MemoryStore) UpdateStatus(ctx context.Context, id string, status int, fromCache bool, streaming bool) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	rec, ok := m.byID[id]
	if !ok {
		return fmt.Errorf("record with ID %s not found", id)
	}

	if status != 0 {
		rec.ResponseStatus = status
	}
	rec.FromCache = rec.FromCache || fromCache
	rec.Streaming = rec.Streaming || streaming

	return nil
}

func (m *MemoryStore) UpdateLifecycle(
	ctx context.Context,
	id string,
	state string,
	endedAt time.Time,
	durationMS int64,
	reason string,
) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	rec, ok := m.byID[id]
	if !ok {
		return fmt.Errorf("record with ID %s not found", id)
	}
	if rec.LifecycleState != "" && rec.LifecycleState != LifecycleInProgress {
		return nil
	}
	rec.LifecycleState = state
	rec.EndedAt = cloneTimePtr(&endedAt)
	rec.DurationMS = durationMS
	rec.TerminalReason = reason
	return nil
}

// AttachRequest updates the request body for a record.
func (m *MemoryStore) AttachRequest(ctx context.Context, id string, body string, truncated bool) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	rec, ok := m.byID[id]
	if !ok {
		return fmt.Errorf("record with ID %s not found", id)
	}

	rec.RequestBody = body
	rec.RequestBodyTruncated = rec.RequestBodyTruncated || truncated

	return nil
}

// AttachResponse updates the response body for a record.
func (m *MemoryStore) AttachResponse(ctx context.Context, id string, body string, truncated bool) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	rec, ok := m.byID[id]
	if !ok {
		return fmt.Errorf("record with ID %s not found", id)
	}

	rec.ResponseBody = body
	rec.ResponseBodyTruncated = rec.ResponseBodyTruncated || truncated

	return nil
}

// AppendOutcome links post-route feedback to a record.
func (m *MemoryStore) AppendOutcome(ctx context.Context, id string, outcome Outcome) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	rec, ok := m.byID[id]
	if !ok {
		return fmt.Errorf("record with ID %s not found", id)
	}

	rec.Outcomes = append(rec.Outcomes, cloneOutcome(outcome))
	return nil
}

// UpdateHallucinationStatus updates hallucination detection results for a record.
func (m *MemoryStore) UpdateHallucinationStatus(ctx context.Context, id string, detected bool, confidence float32, spans []string, spanDetails []HallucinationSpan, score ...HallucinationScore) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	rec, ok := m.byID[id]
	if !ok {
		return fmt.Errorf("record with ID %s not found", id)
	}

	rec.HallucinationDetected = detected
	rec.HallucinationConfidence = confidence
	applyHallucinationScore(rec, score)
	rec.HallucinationSpans = cloneStringSlice(spans)
	rec.HallucinationSpanDetails = cloneHallucinationSpanDetails(spanDetails)

	return nil
}

// UpdateUsageCost updates token usage and pricing-derived cost fields for a record.
func (m *MemoryStore) UpdateUsageCost(ctx context.Context, id string, usage UsageCost) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	rec, ok := m.byID[id]
	if !ok {
		return fmt.Errorf("record with ID %s not found", id)
	}

	rec.PromptTokens = cloneIntPtr(usage.PromptTokens)
	rec.CachedPromptTokens = cloneIntPtr(usage.CachedPromptTokens)
	rec.CacheWriteTokens = cloneIntPtr(usage.CacheWriteTokens)
	rec.CompletionTokens = cloneIntPtr(usage.CompletionTokens)
	rec.TotalTokens = cloneIntPtr(usage.TotalTokens)
	rec.ActualCost = cloneFloat64Ptr(usage.ActualCost)
	rec.BaselineCost = cloneFloat64Ptr(usage.BaselineCost)
	rec.CostSavings = cloneFloat64Ptr(usage.CostSavings)
	rec.Currency = cloneStringPtr(usage.Currency)
	rec.BaselineModel = cloneStringPtr(usage.BaselineModel)

	return nil
}

// UpdateToolTrace updates tool-calling trace details for a record.
func (m *MemoryStore) UpdateToolTrace(ctx context.Context, id string, trace ToolTrace) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	rec, ok := m.byID[id]
	if !ok {
		return fmt.Errorf("record with ID %s not found", id)
	}

	rec.ToolTrace = cloneToolTrace(&trace)
	return nil
}

// Close is a no-op for memory storage.
func (m *MemoryStore) Close() error {
	return nil
}

// SetMaxRecords updates the maximum number of records to keep in memory.
func (m *MemoryStore) SetMaxRecords(max int) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if max <= 0 {
		max = DefaultMaxRecords
	}
	m.maxRecords = max

	// Evict oldest records if over capacity. Clear evicted slots so the
	// backing array does not retain references to dropped Records.
	for len(m.records) > m.maxRecords {
		oldest := m.records[0]
		delete(m.byID, oldest.ID)
		m.records[0] = nil
		m.records = m.records[1:]
	}
}

// generateID creates a random hex ID for a record.
func generateID() (string, error) {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return hex.EncodeToString(b), nil
}
