package extproc

import (
	"context"
	"errors"
)

// Pin leases the serving router generation and its configuration snapshot
// for one native gateway request. A reload swaps the generation for later
// requests only, exactly as it does for ext_proc streams.
func (s *Server) Pin() (*Lease, error) {
	return s.service.Pin()
}

// StartContextWithoutExtProc runs the server for the native gateway: the
// config watcher, generation swaps and shutdown behave as in
// StartContextWithReady, but no ext_proc gRPC listener is opened. onServing
// runs once the server is ready; the call returns when ctx ends.
func (s *Server) StartContextWithoutExtProc(ctx context.Context, onServing func()) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if s.lifecycle.isStopping() {
		return errors.New("router server is shutting down")
	}
	s.servingMu.Lock()
	ready := s.servingReadyLocked()
	select {
	case <-ready:
		s.servingMu.Unlock()
		return errors.New("router server is already serving")
	default:
		close(ready)
	}
	s.servingMu.Unlock()
	if onServing != nil {
		onServing()
	}
	watchCtx, watcherDone := s.lifecycle.startWatcher(ctx)
	defer s.lifecycle.beginShutdown()
	go func() {
		defer watcherDone()
		s.watchConfigAndReload(watchCtx)
	}()
	<-ctx.Done()
	return nil
}
