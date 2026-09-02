package extproc

import "errors"

type resourceScope struct {
	closers []func() error
	closed  bool
}

func newResourceScope() *resourceScope {
	return &resourceScope{}
}

func (s *resourceScope) add(closer func() error) {
	if closer == nil {
		return
	}
	s.closers = append(s.closers, closer)
}

func (s *resourceScope) close() error {
	if s == nil {
		return nil
	}
	if s.closed {
		return nil
	}
	s.closed = true
	closers := s.closers
	s.closers = nil

	var errs []error
	for i := len(closers) - 1; i >= 0; i-- {
		if err := closers[i](); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}
