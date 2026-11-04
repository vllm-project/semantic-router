package backendtelemetry

import (
	"cmp"
	"errors"
	"slices"
	"sync"
	"time"
)

// DefaultTTL is how long an observation without its own TTL stays fresh.
const DefaultTTL = 5 * time.Second

// Store keeps the latest observation per replica. It is safe for concurrent
// use. Observations are values, so nothing a caller does to an argument or a
// result changes what the store holds.
type Store struct {
	defaultTTL time.Duration
	now        func() time.Time

	mu   sync.RWMutex
	byID map[Identity]Observation
}

// NewStore returns an empty store. A non-positive defaultTTL uses DefaultTTL.
func NewStore(defaultTTL time.Duration) *Store {
	return newStoreWithClock(defaultTTL, time.Now)
}

func newStoreWithClock(defaultTTL time.Duration, now func() time.Time) *Store {
	if defaultTTL <= 0 {
		defaultTTL = DefaultTTL
	}
	return &Store{defaultTTL: defaultTTL, now: now, byID: make(map[Identity]Observation)}
}

// Upsert records a valid observation. An observation collected before the one
// already held for the same replica is ignored, so a slow scrape cannot
// replace newer evidence.
func (s *Store) Upsert(o Observation) error {
	if err := o.Validate(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if held, ok := s.byID[o.Identity]; ok && o.CollectedAt.Before(held.CollectedAt) {
		return nil
	}
	s.byID[o.Identity] = o
	return nil
}

// UpsertMany records every valid observation and returns the joined errors of
// the invalid ones; one bad replica does not discard the rest of a scrape.
func (s *Store) UpsertMany(observations []Observation) error {
	var errs []error
	for _, o := range observations {
		if err := s.Upsert(o); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

// Get returns the latest observation for a replica whether or not it is
// fresh, for diagnostics.
func (s *Store) Get(id Identity) (Observation, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	o, ok := s.byID[id]
	return o, ok
}

// Fresh returns the latest observation for a replica only while it is fresh.
func (s *Store) Fresh(id Identity) (Observation, bool) {
	o, ok := s.Get(id)
	if !ok || !s.IsFresh(o) {
		return Observation{}, false
	}
	return o, true
}

// FreshForModel returns the fresh observations of one model's replicas,
// ordered by backend ref and replica.
func (s *Store) FreshForModel(model string) []Observation {
	s.mu.RLock()
	var out []Observation
	for id, o := range s.byID {
		if id.Model == model && s.IsFresh(o) {
			out = append(out, o)
		}
	}
	s.mu.RUnlock()
	slices.SortFunc(out, func(a, b Observation) int {
		return cmp.Or(
			cmp.Compare(a.Identity.BackendRef, b.Identity.BackendRef),
			cmp.Compare(a.Identity.Replica, b.Identity.Replica),
			cmp.Compare(a.Identity.Engine, b.Identity.Engine),
		)
	})
	return out
}

// IsFresh reports whether an observation is within its TTL now. An
// observation stamped after the store's clock is not fresh: its age cannot be
// trusted.
func (s *Store) IsFresh(o Observation) bool {
	age := s.Age(o)
	return age >= 0 && age < s.ttl(o)
}

// Age is how long ago the observation was collected by the store's clock.
func (s *Store) Age(o Observation) time.Duration { return s.now().Sub(o.CollectedAt) }

// Remove forgets a replica, for example when it leaves the configuration.
func (s *Store) Remove(id Identity) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.byID, id)
}

func (s *Store) ttl(o Observation) time.Duration {
	if o.TTL > 0 {
		return o.TTL
	}
	return s.defaultTTL
}
