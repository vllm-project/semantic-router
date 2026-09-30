package extproc

import (
	"context"
	"errors"
	"net"
	"sync"

	healthpb "google.golang.org/grpc/health/grpc_health_v1"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type servingListener struct {
	net.Listener
	once      sync.Once
	onServing func()
}

func (l *servingListener) Accept() (net.Conn, error) {
	l.once.Do(l.onServing)
	return l.Listener.Accept()
}

func (s *Server) servingReadyLocked() chan struct{} {
	if s.servingReady == nil {
		s.servingReady = make(chan struct{})
	}
	return s.servingReady
}

// WaitForServing waits for the gRPC server to begin accepting on its bound
// listener. The controller and server start concurrently, so cold-start
// candidates cannot be dropped before a watcher subscribes.
func (s *Server) WaitForServing(ctx context.Context) error {
	s.servingMu.Lock()
	ready := s.servingReadyLocked()
	s.servingMu.Unlock()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-ready:
		if s.lifecycle.isStopping() {
			return errors.New("router server is shutting down")
		}
		return ctx.Err()
	}
}

// ActivateKubernetesConfig acknowledges only a successfully published serving
// generation. A failed candidate leaves the previous generation available.
func (s *Server) ActivateKubernetesConfig(ctx context.Context, cfg *config.RouterConfig) error {
	if err := s.WaitForServing(ctx); err != nil {
		return err
	}
	finish, ok := s.lifecycle.startReload()
	if !ok {
		return errors.New("router server is shutting down")
	}
	defer finish()
	s.reloadMu.Lock()
	defer s.reloadMu.Unlock()
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := s.reloadRouterFromConfigLockedContext(ctx, "kubernetes", s.configPath, cfg); err != nil {
		return err
	}
	s.markServingReady()
	return nil
}

func (s *Server) markServingReady() {
	s.servingMu.Lock()
	defer s.servingMu.Unlock()
	if s.healthServer != nil && !s.lifecycle.isStopping() {
		s.healthServer.SetServingStatus("", healthpb.HealthCheckResponse_SERVING)
	}
}
