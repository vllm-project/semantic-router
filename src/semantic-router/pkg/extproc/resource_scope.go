package extproc

import (
	"errors"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type resourceCloser struct {
	close func() error
	done  <-chan struct{}
}

type resourceScope struct {
	closers []resourceCloser
	closed  bool
}

func newResourceScope() *resourceScope {
	return &resourceScope{}
}

func (s *resourceScope) add(closer func() error) {
	if closer == nil {
		return
	}
	s.addDraining(closer, nil)
}

// addDraining retains all earlier resources until the asynchronous owner exits.
func (s *resourceScope) addDraining(closer func() error, done <-chan struct{}) {
	s.closers = append(s.closers, resourceCloser{close: closer, done: done})
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
	return closeResources(closers)
}

func closeResources(closers []resourceCloser) error {
	var errs []error
	for i := len(closers) - 1; i >= 0; i-- {
		if err := closers[i].close(); err != nil {
			errs = append(errs, err)
		}
		if done := closers[i].done; done != nil {
			select {
			case <-done:
			default:
				go func(remaining []resourceCloser) {
					<-done
					if err := closeResources(remaining); err != nil {
						logging.Errorf("Deferred router resource cleanup failed: %v", err)
					}
				}(closers[:i])
				return errors.Join(errs...)
			}
		}
	}
	return errors.Join(errs...)
}
