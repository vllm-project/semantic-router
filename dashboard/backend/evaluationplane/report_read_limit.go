package evaluationplane

import (
	"fmt"
	"sync"
)

func (s *Service) reserveEvidenceReadCapacity() (func(), error) {
	select {
	case s.activity.evidenceReads <- struct{}{}:
		var once sync.Once
		return func() {
			once.Do(func() {
				<-s.activity.evidenceReads
			})
		}, nil
	default:
		return nil, fmt.Errorf("%w: evaluation evidence read capacity is exhausted", ErrConflict)
	}
}

func (s *Service) acquireEvidenceRead() (func(), error) {
	releaseCapacity, err := s.reserveEvidenceReadCapacity()
	if err != nil {
		return nil, err
	}
	// Reserve bounded read capacity before waiting on the durable-evidence
	// read lock. A saturated reader set must not hold a read lock that blocks
	// publication or deletion from making progress.
	s.activity.evidenceMu.RLock()
	var once sync.Once
	return func() {
		once.Do(func() {
			s.activity.evidenceMu.RUnlock()
			releaseCapacity()
		})
	}, nil
}
