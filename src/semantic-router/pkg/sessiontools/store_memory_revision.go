package sessiontools

import (
	"math"
	"time"
)

func (s *MemoryStore) allocateRevision() (uint64, error) {
	for {
		current := s.nextRevision.Load()
		if current == math.MaxUint64 {
			return 0, ErrRevisionExhausted
		}
		if s.nextRevision.CompareAndSwap(current, current+1) {
			return current + 1, nil
		}
	}
}

// prepareWrite is shared by creation and updates. All potentially failing
// preparation occurs before deleting, evicting, or publishing state. A token
// reserved for a rejected write is never reused; gaps are not update counts.
func (s *MemoryStore) prepareWrite(next State, now time.Time, ttl time.Duration) (State, error) {
	revision, err := s.allocateRevision()
	if err != nil {
		return State{}, err
	}
	stored := next.Clone()
	stored.Revision = revision
	stored.LastSeenAt = now
	stored.ExpiresAt = now.Add(ttl)
	if err := stored.validateEncodedSize(s.maxStateBytes); err != nil {
		return State{}, err
	}
	return stored, nil
}
