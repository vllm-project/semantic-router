package store

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"slices"
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
	ttl        time.Duration
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
		ttl:        time.Duration(ttlSeconds) * time.Second,
	}
}

// Add inserts a new record into memory storage.
func (m *MemoryStore) Add(ctx context.Context, record Record) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

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
	if _, err := encodeRecord(&record, MaxRecordBytes); err != nil {
		return "", err
	}

	// Evict oldest if at capacity
	if len(m.records) >= m.maxRecords {
		oldest := m.records[0]
		delete(m.byID, oldest.ID)
		m.records = m.records[1:]
	}

	// Make a copy to avoid external mutations
	copyRec := cloneRecord(record)
	m.records = append(m.records, &copyRec)
	m.byID[copyRec.ID] = &copyRec

	return copyRec.ID, nil
}

// Get retrieves a record by ID from memory.
func (m *MemoryStore) Get(ctx context.Context, id string) (Record, bool, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	rec, ok := m.byID[id]
	if !ok || isExpired(*rec, m.ttl) {
		return Record{}, false, nil
	}

	return cloneRecord(*rec), true, nil
}

// List returns all records ordered by timestamp descending.
func (m *MemoryStore) List(ctx context.Context) ([]Record, error) {
	return listAll(m.scan)
}

func (m *MemoryStore) QueryPage(ctx context.Context, filters QueryFilters, limit, offset int, details bool) (RecordPage, error) {
	return collectPage(filters, limit, offset, m.count, m.scan)
}

func (m *MemoryStore) ScanMetadata(ctx context.Context, visit func(Record) error) error {
	return scanAll(m.scan, visit)
}

func (m *MemoryStore) count() (int, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	live := 0
	for _, rec := range m.records {
		if !isExpired(*rec, m.ttl) {
			live++
		}
	}
	return live, nil
}

// scan visits live records newest first by timestamp, then ID.
func (m *MemoryStore) scan(from int, visit func(Record) bool) error {
	m.mu.RLock()
	defer m.mu.RUnlock()
	live := make([]*Record, 0, len(m.records))
	for _, rec := range m.records {
		if !isExpired(*rec, m.ttl) {
			live = append(live, rec)
		}
	}
	slices.SortFunc(live, func(a, b *Record) int { return newestFirst(*a, *b) })
	for _, rec := range live[min(from, len(live)):] {
		if !visit(cloneRecord(*rec)) {
			break
		}
	}
	return nil
}

// UpdateStatus updates the response status and flags for a record.
func (m *MemoryStore) UpdateStatus(ctx context.Context, id string, status int, fromCache bool, streaming bool) error {
	return m.updateRecord(id, func(rec *Record) {
		if status != 0 {
			rec.ResponseStatus = status
		}
		rec.FromCache = rec.FromCache || fromCache
		rec.Streaming = rec.Streaming || streaming
	})
}

func (m *MemoryStore) UpdateLifecycle(
	ctx context.Context,
	id string,
	state string,
	endedAt time.Time,
	durationMS int64,
	reason string,
) error {
	return m.updateRecord(id, func(rec *Record) {
		if rec.LifecycleState != "" && rec.LifecycleState != LifecycleInProgress {
			return
		}
		rec.LifecycleState = state
		rec.EndedAt = cloneTimePtr(&endedAt)
		rec.DurationMS = durationMS
		rec.TerminalReason = reason
	})
}

// AttachRequest updates the request body for a record.
func (m *MemoryStore) AttachRequest(ctx context.Context, id string, body string, truncated bool) error {
	return m.updateRecord(id, func(rec *Record) {
		rec.RequestBody = body
		rec.RequestBodyTruncated = rec.RequestBodyTruncated || truncated
	})
}

// AttachResponse updates the response body for a record.
func (m *MemoryStore) AttachResponse(ctx context.Context, id string, body string, truncated bool) error {
	return m.updateRecord(id, func(rec *Record) {
		rec.ResponseBody = body
		rec.ResponseBodyTruncated = rec.ResponseBodyTruncated || truncated
	})
}

// AppendOutcome links post-route feedback to a record.
func (m *MemoryStore) AppendOutcome(ctx context.Context, id string, outcome Outcome) error {
	return m.updateRecord(id, func(rec *Record) {
		rec.Outcomes = append(rec.Outcomes, cloneOutcome(outcome))
	})
}

// UpdateHallucinationStatus updates hallucination detection results for a record.
func (m *MemoryStore) UpdateHallucinationStatus(ctx context.Context, id string, detected bool, confidence float32, spans []string, spanDetails []HallucinationSpan, score ...HallucinationScore) error {
	return m.updateRecord(id, func(rec *Record) {
		rec.HallucinationDetected = detected
		rec.HallucinationConfidence = confidence
		applyHallucinationScore(rec, score)
		rec.HallucinationSpans = cloneStringSlice(spans)
		rec.HallucinationSpanDetails = cloneHallucinationSpanDetails(spanDetails)
	})
}

// UpdateUsageCost updates token usage and pricing-derived cost fields for a record.
func (m *MemoryStore) UpdateUsageCost(ctx context.Context, id string, usage UsageCost) error {
	return m.updateRecord(id, func(rec *Record) {
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
	})
}

// UpdateToolTrace updates tool-calling trace details for a record.
func (m *MemoryStore) UpdateToolTrace(ctx context.Context, id string, trace ToolTrace) error {
	return m.updateRecord(id, func(rec *Record) {
		rec.ToolTrace = cloneToolTrace(&trace)
	})
}

// updateRecord keeps a change only if the whole record still fits.
func (m *MemoryStore) updateRecord(id string, mutate func(*Record)) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	rec, ok := m.byID[id]
	if !ok {
		return fmt.Errorf("record with ID %s not found", id)
	}
	updated := cloneRecord(*rec)
	mutate(&updated)
	if _, err := encodeRecord(&updated, MaxRecordBytes); err != nil {
		return err
	}
	*rec = updated
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

	// Evict oldest records if over capacity
	for len(m.records) > m.maxRecords {
		oldest := m.records[0]
		delete(m.byID, oldest.ID)
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
