package store

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"sync"
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
	ttl        int // Not used in memory store, but kept for interface compatibility
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

	if record.ID == "" {
		id, err := generateID()
		if err != nil {
			return "", fmt.Errorf("failed to generate ID: %w", err)
		}
		record.ID = id
	}

	// Evict oldest if at capacity
	if len(m.records) >= m.maxRecords {
		oldest := m.records[0]
		delete(m.byID, oldest.ID)
		m.records = m.records[1:]
	}

	// Make a copy to avoid external mutations
	copyRec := record
	m.records = append(m.records, &copyRec)
	m.byID[copyRec.ID] = &copyRec

	return copyRec.ID, nil
}

// Get retrieves a record by ID from memory.
func (m *MemoryStore) Get(ctx context.Context, id string) (Record, bool, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	rec, ok := m.byID[id]
	if !ok {
		return Record{}, false, nil
	}

	return *rec, true, nil
}

// List returns all records ordered by timestamp descending.
func (m *MemoryStore) List(ctx context.Context) ([]Record, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make([]Record, len(m.records))
	for i := len(m.records) - 1; i >= 0; i-- {
		result[len(m.records)-1-i] = *m.records[i]
	}

	return result, nil
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

// UpdateHallucinationStatus updates hallucination detection results for a record.
func (m *MemoryStore) UpdateHallucinationStatus(ctx context.Context, id string, detected bool, confidence float32, spans []string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	rec, ok := m.byID[id]
	if !ok {
		return fmt.Errorf("record with ID %s not found", id)
	}

	rec.HallucinationDetected = detected
	rec.HallucinationConfidence = confidence
	rec.HallucinationSpans = spans

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
