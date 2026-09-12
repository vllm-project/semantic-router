package extproc

import (
	"errors"
	"sync"
)

type resourceScope struct {
	mu      sync.Mutex
	closers []func() error
	closed  bool
}

func newResourceScope() *resourceScope {
	return &resourceScope{}
}

func (s *resourceScope) add(closer func() error) {
	if s == nil || closer == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return
	}
	s.closers = append(s.closers, closer)
}

func (s *resourceScope) close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	closers := s.closers
	s.closers = nil
	s.mu.Unlock()

	var errs []error
	for i := len(closers) - 1; i >= 0; i-- {
		if err := closers[i](); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}
