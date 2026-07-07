package backend

import (
	"fmt"
	"strings"
	"sync"
	"time"
)

const DefaultTelemetryTTL = 5 * time.Second

// Store keeps the latest telemetry sample per model/backend/replica tuple.
type Store struct {
	mu         sync.RWMutex
	items      map[string]BackendTelemetry
	defaultTTL time.Duration
	now        func() time.Time
}

var defaultStore = NewStore(DefaultTelemetryTTL)

// NewStore creates a telemetry store with the provided default TTL.
func NewStore(defaultTTL time.Duration) *Store {
	if defaultTTL <= 0 {
		defaultTTL = DefaultTelemetryTTL
	}
	return newStoreWithClock(defaultTTL, time.Now)
}

func newStoreWithClock(defaultTTL time.Duration, now func() time.Time) *Store {
	if defaultTTL <= 0 {
		defaultTTL = DefaultTelemetryTTL
	}
	if now == nil {
		now = time.Now
	}
	return &Store{
		items:      map[string]BackendTelemetry{},
		defaultTTL: defaultTTL,
		now:        now,
	}
}

// DefaultStore returns the package-level telemetry store.
func DefaultStore() *Store {
	return defaultStore
}

// Upsert writes telemetry into the package-level store.
func Upsert(telemetry BackendTelemetry) error {
	return defaultStore.Upsert(telemetry)
}

// UpsertMany writes telemetry samples into the package-level store.
func UpsertMany(samples []BackendTelemetry) error {
	return defaultStore.UpsertMany(samples)
}

// Get returns raw telemetry from the package-level store.
func Get(identity BackendIdentity) (BackendTelemetry, bool) {
	return defaultStore.Get(identity)
}

// GetFresh returns non-stale telemetry from the package-level store.
func GetFresh(identity BackendIdentity) (BackendTelemetry, bool) {
	return defaultStore.GetFresh(identity)
}

// ListByModel returns raw model telemetry from the package-level store.
func ListByModel(modelName string) []BackendTelemetry {
	return defaultStore.ListByModel(modelName)
}

// Upsert writes telemetry into the store. CollectedAt defaults to now when
// omitted; TTL defaults are applied during freshness checks.
func (s *Store) Upsert(telemetry BackendTelemetry) error {
	if s == nil {
		return fmt.Errorf("backend telemetry store is nil")
	}
	telemetry.Identity = telemetry.Identity.Normalize()
	if telemetry.Identity.BackendID == "" {
		return fmt.Errorf("backend telemetry requires backend identity")
	}
	if telemetry.Identity.ModelName == "" {
		return fmt.Errorf("backend telemetry requires model name")
	}
	if telemetry.CollectedAt.IsZero() {
		telemetry.CollectedAt = s.now()
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.items[telemetry.Identity.Key()] = telemetry
	return nil
}

// UpsertMany writes telemetry samples into the store.
func (s *Store) UpsertMany(samples []BackendTelemetry) error {
	if s == nil {
		return fmt.Errorf("backend telemetry store is nil")
	}
	for _, sample := range samples {
		if err := s.Upsert(sample); err != nil {
			return err
		}
	}
	return nil
}

// Get returns raw telemetry without freshness filtering.
func (s *Store) Get(identity BackendIdentity) (BackendTelemetry, bool) {
	if s == nil {
		return BackendTelemetry{}, false
	}
	identity = identity.Normalize()

	s.mu.RLock()
	defer s.mu.RUnlock()
	telemetry, ok := s.items[identity.Key()]
	return telemetry, ok
}

// GetFresh returns telemetry only when it is still within its TTL window.
func (s *Store) GetFresh(identity BackendIdentity) (BackendTelemetry, bool) {
	telemetry, ok := s.Get(identity)
	if !ok || !s.IsFresh(telemetry) {
		return BackendTelemetry{}, false
	}
	return telemetry, true
}

// ListByModel returns raw telemetry for a model.
func (s *Store) ListByModel(modelName string) []BackendTelemetry {
	if s == nil {
		return nil
	}
	modelName = strings.TrimSpace(modelName)
	results := []BackendTelemetry{}

	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, telemetry := range s.items {
		if telemetry.Identity.ModelName == modelName {
			results = append(results, telemetry)
		}
	}
	return results
}

// ListFreshByModel returns non-stale telemetry for a model.
func (s *Store) ListFreshByModel(modelName string) []BackendTelemetry {
	raw := s.ListByModel(modelName)
	results := make([]BackendTelemetry, 0, len(raw))
	for _, telemetry := range raw {
		if s.IsFresh(telemetry) {
			results = append(results, telemetry)
		}
	}
	return results
}

// ListByBackend returns raw telemetry for a model/backend pair.
func (s *Store) ListByBackend(modelName string, backendID string) []BackendTelemetry {
	if s == nil {
		return nil
	}
	modelName = strings.TrimSpace(modelName)
	backendID = strings.TrimSpace(backendID)
	results := []BackendTelemetry{}

	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, telemetry := range s.items {
		if telemetry.Identity.ModelName == modelName && telemetry.Identity.BackendID == backendID {
			results = append(results, telemetry)
		}
	}
	return results
}

// IsFresh reports whether telemetry is within its own TTL or the store default.
func (s *Store) IsFresh(telemetry BackendTelemetry) bool {
	if s == nil || telemetry.CollectedAt.IsZero() {
		return false
	}
	age := s.Age(telemetry)
	return age >= 0 && age <= s.ttlFor(telemetry)
}

// Age reports telemetry age using the store clock.
func (s *Store) Age(telemetry BackendTelemetry) time.Duration {
	if s == nil || telemetry.CollectedAt.IsZero() {
		return 0
	}
	return s.now().Sub(telemetry.CollectedAt)
}

// Remove deletes telemetry for an identity.
func (s *Store) Remove(identity BackendIdentity) {
	if s == nil {
		return
	}
	identity = identity.Normalize()

	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.items, identity.Key())
}

// Reset clears the store.
func (s *Store) Reset() {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.items = map[string]BackendTelemetry{}
}

func (s *Store) ttlFor(telemetry BackendTelemetry) time.Duration {
	if telemetry.TTL > 0 {
		return telemetry.TTL
	}
	return s.defaultTTL
}
