package evaluationplane

import "fmt"

func (s *Service) acquireEvidenceRead() (func(), error) {
	select {
	case s.evidenceReads <- struct{}{}:
		return func() { <-s.evidenceReads }, nil
	default:
		return nil, fmt.Errorf("%w: evaluation evidence read capacity is exhausted", ErrConflict)
	}
}
