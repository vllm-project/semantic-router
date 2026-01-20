package store

import (
	"context"
	"sort"
	"sync"
	"time"
)

// MemoryStore is an in-memory implementation of ReplayStore.
// It uses a ring buffer to store records and evicts the oldest
// records when capacity is reached.
type MemoryStore struct {
	mu      sync.RWMutex
	records []*RoutingRecord
	byID    map[string]*RoutingRecord

	maxRecords int
	enabled    bool
	ttl        time.Duration

	// done channel for cleanup goroutine
	done chan struct{}
}

// NewMemoryStore creates a new in-memory replay store.
func NewMemoryStore(config MemoryStoreConfig, ttlSeconds int, enabled bool) *MemoryStore {
	maxRecords := config.MaxRecords
	if maxRecords <= 0 {
		maxRecords = 200 // Default from routerreplay.DefaultMaxRecords
	}

	ttl := time.Duration(ttlSeconds) * time.Second
	if ttlSeconds <= 0 {
		ttl = DefaultTTL
	}

	store := &MemoryStore{
		records:    make([]*RoutingRecord, 0, maxRecords),
		byID:       make(map[string]*RoutingRecord),
		maxRecords: maxRecords,
		enabled:    enabled,
		ttl:        ttl,
		done:       make(chan struct{}),
	}

	// Start TTL cleanup goroutine
	go store.cleanupExpired()

	return store
}

// IsEnabled returns whether the store is enabled.
func (m *MemoryStore) IsEnabled() bool {
	return m.enabled
}

// CheckConnection verifies the store connection is healthy.
func (m *MemoryStore) CheckConnection(_ context.Context) error {
	if !m.enabled {
		return ErrStoreDisabled
	}
	return nil
}

// Close releases resources held by the store.
func (m *MemoryStore) Close() error {
	close(m.done)

	m.mu.Lock()
	defer m.mu.Unlock()

	m.records = nil
	m.byID = nil
	return nil
}

// StoreRecord stores a new routing record.
func (m *MemoryStore) StoreRecord(_ context.Context, record *RoutingRecord) error {
	if !m.enabled {
		return ErrStoreDisabled
	}
	if record == nil || record.ID == "" {
		return ErrInvalidInput
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	// Check if record already exists
	if _, exists := m.byID[record.ID]; exists {
		return ErrAlreadyExists
	}

	// Evict oldest if at capacity (ring buffer behavior)
	if len(m.records) >= m.maxRecords {
		oldest := m.records[0]
		delete(m.byID, oldest.ID)
		m.records = m.records[1:]
	}

	// Store a copy
	stored := *record
	m.records = append(m.records, &stored)
	m.byID[stored.ID] = &stored

	return nil
}

// GetRecord retrieves a routing record by ID.
func (m *MemoryStore) GetRecord(_ context.Context, recordID string) (*RoutingRecord, error) {
	if !m.enabled {
		return nil, ErrStoreDisabled
	}
	if recordID == "" {
		return nil, ErrInvalidID
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	record, exists := m.byID[recordID]
	if !exists {
		return nil, ErrNotFound
	}

	// Return a copy
	result := *record
	return &result, nil
}

// UpdateRecord updates an existing routing record.
func (m *MemoryStore) UpdateRecord(_ context.Context, record *RoutingRecord) error {
	if !m.enabled {
		return ErrStoreDisabled
	}
	if record == nil || record.ID == "" {
		return ErrInvalidInput
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	existing, exists := m.byID[record.ID]
	if !exists {
		return ErrNotFound
	}

	// Update in place
	*existing = *record
	return nil
}

// DeleteRecord deletes a routing record by ID.
func (m *MemoryStore) DeleteRecord(_ context.Context, recordID string) error {
	if !m.enabled {
		return ErrStoreDisabled
	}
	if recordID == "" {
		return ErrInvalidID
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.byID[recordID]; !exists {
		return ErrNotFound
	}

	delete(m.byID, recordID)

	// Remove from slice
	for i, rec := range m.records {
		if rec.ID == recordID {
			m.records = append(m.records[:i], m.records[i+1:]...)
			break
		}
	}

	return nil
}

// ListRecords lists routing records with pagination and filtering.
func (m *MemoryStore) ListRecords(_ context.Context, opts ListOptions) (*ListResult, error) {
	if !m.enabled {
		return nil, ErrStoreDisabled
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	// Apply filters and collect matching records
	var filtered []*RoutingRecord
	for _, rec := range m.records {
		if m.matchesFilters(rec, opts) {
			// Make a copy
			result := *rec
			filtered = append(filtered, &result)
		}
	}

	// Sort by timestamp
	if opts.Order == "asc" {
		sort.Slice(filtered, func(i, j int) bool {
			return filtered[i].Timestamp.Before(filtered[j].Timestamp)
		})
	} else {
		// Default: descending (newest first)
		sort.Slice(filtered, func(i, j int) bool {
			return filtered[i].Timestamp.After(filtered[j].Timestamp)
		})
	}

	// Apply pagination
	return m.applyPagination(filtered, opts), nil
}

// matchesFilters checks if a record matches the filter criteria.
func (m *MemoryStore) matchesFilters(rec *RoutingRecord, opts ListOptions) bool {
	if opts.DecisionName != "" && rec.Decision != opts.DecisionName {
		return false
	}
	if opts.Category != "" && rec.Category != opts.Category {
		return false
	}
	if opts.Model != "" && rec.SelectedModel != opts.Model {
		return false
	}
	if opts.StartTime != nil && rec.Timestamp.Before(*opts.StartTime) {
		return false
	}
	if opts.EndTime != nil && rec.Timestamp.After(*opts.EndTime) {
		return false
	}
	if opts.FromCache != nil && rec.FromCache != *opts.FromCache {
		return false
	}
	if opts.RequestID != "" && rec.RequestID != opts.RequestID {
		return false
	}
	return true
}

// applyPagination applies cursor-based pagination to the records.
func (m *MemoryStore) applyPagination(records []*RoutingRecord, opts ListOptions) *ListResult {
	limit := opts.Limit
	if limit <= 0 {
		limit = DefaultListLimit
	}
	if limit > MaxListLimit {
		limit = MaxListLimit
	}

	// Find cursor position if After is set
	startIdx := 0
	if opts.After != "" {
		for i, rec := range records {
			if rec.ID == opts.After {
				startIdx = i + 1
				break
			}
		}
	}

	// Find cursor position if Before is set
	endIdx := len(records)
	if opts.Before != "" {
		for i, rec := range records {
			if rec.ID == opts.Before {
				endIdx = i
				break
			}
		}
	}

	// Apply bounds
	if startIdx >= len(records) {
		return &ListResult{Records: nil, HasMore: false}
	}
	if startIdx >= endIdx {
		return &ListResult{Records: nil, HasMore: false}
	}

	// Slice to range
	records = records[startIdx:endIdx]

	// Apply limit
	hasMore := len(records) > limit
	if hasMore {
		records = records[:limit]
	}

	result := &ListResult{
		Records: records,
		HasMore: hasMore,
	}

	if len(records) > 0 {
		result.FirstID = records[0].ID
		result.LastID = records[len(records)-1].ID
	}

	return result
}

// cleanupExpired removes records that have exceeded their TTL.
func (m *MemoryStore) cleanupExpired() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-m.done:
			return
		case <-ticker.C:
			m.mu.Lock()
			now := time.Now()
			cutoff := now.Add(-m.ttl)

			// Remove expired records
			newRecords := make([]*RoutingRecord, 0, len(m.records))
			for _, rec := range m.records {
				if rec.Timestamp.After(cutoff) {
					newRecords = append(newRecords, rec)
				} else {
					delete(m.byID, rec.ID)
				}
			}
			m.records = newRecords
			m.mu.Unlock()
		}
	}
}

// RecordCount returns the current number of records (for testing).
func (m *MemoryStore) RecordCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.records)
}
