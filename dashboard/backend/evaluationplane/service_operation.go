package evaluationplane

import "fmt"

// beginOperation pins the Service's root lease for the complete externally
// callable operation. Close takes the write side, rejects new work, drains all
// readers, and only then releases the last per-root coordinator reference.
func (s *Service) beginOperation() (func(), error) {
	s.operationMu.RLock()
	s.mu.Lock()
	closed := s.closed
	s.mu.Unlock()
	if closed {
		s.operationMu.RUnlock()
		return nil, fmt.Errorf("%w: evaluation service is closed", ErrConflict)
	}
	return s.operationMu.RUnlock, nil
}
