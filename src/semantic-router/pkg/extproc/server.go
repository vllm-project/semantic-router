package extproc

import (
	"context"
	"fmt"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"sync/atomic"
	"syscall"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/fsnotify/fsnotify"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability"
	"google.golang.org/grpc"
)

// Server represents a gRPC server for the Envoy ExtProc
type Server struct {
	configPath string
	service    *RouterService
	server     *grpc.Server
	port       int
}

// NewServer creates a new ExtProc gRPC server
func NewServer(configPath string, port int) (*Server, error) {
	router, err := NewOpenAIRouter(configPath)
	if err != nil {
		return nil, err
	}

	service := NewRouterService(router)
	return &Server{
		configPath: configPath,
		service:    service,
		port:       port,
	}, nil
}

// Start starts the gRPC server
func (s *Server) Start() error {
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", s.port))
	if err != nil {
		return fmt.Errorf("failed to listen on port %d: %w", s.port, err)
	}

	s.server = grpc.NewServer()
	ext_proc.RegisterExternalProcessorServer(s.server, s.service)

	observability.Infof("Starting LLM Router ExtProc server on port %d...", s.port)

	// Run the server in a separate goroutine
	serverErrCh := make(chan error, 1)
	go func() {
		if err := s.server.Serve(lis); err != nil && err != grpc.ErrServerStopped {
			observability.Errorf("Server error: %v", err)
			serverErrCh <- err
		} else {
			serverErrCh <- nil
		}
	}()

	// Start config file watcher in background
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go s.watchConfigAndReload(ctx)

	// Wait for interrupt signal to gracefully shut down the server
	signalChan := make(chan os.Signal, 1)
	signal.Notify(signalChan, syscall.SIGINT, syscall.SIGTERM)

	// Wait for either server error or shutdown signal
	select {
	case err := <-serverErrCh:
		if err != nil {
			observability.Errorf("Server exited with error: %v", err)
			return err
		}
	case <-signalChan:
		observability.Infof("Received shutdown signal, gracefully stopping server...")
	}

	s.Stop()
	return nil
}

// Stop stops the gRPC server
func (s *Server) Stop() {
	if s.server != nil {
		s.server.GracefulStop()
		observability.Infof("Server stopped")
	}
}

// RouterService is a delegating gRPC service that forwards to the current router implementation.
type RouterService struct {
	current atomic.Pointer[OpenAIRouter]
}

func NewRouterService(r *OpenAIRouter) *RouterService {
	rs := &RouterService{}
	rs.current.Store(r)
	return rs
}

// Swap replaces the current router implementation.
func (rs *RouterService) Swap(r *OpenAIRouter) { rs.current.Store(r) }

// Process delegates to the current router.
func (rs *RouterService) Process(stream ext_proc.ExternalProcessor_ProcessServer) error {
	r := rs.current.Load()
	return r.Process(stream)
}

// watchConfigAndReload watches the config file and reloads router on changes.
func (s *Server) watchConfigAndReload(ctx context.Context) {
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		observability.LogEvent("config_watcher_error", map[string]interface{}{
			"stage": "create_watcher",
			"error": err.Error(),
		})
		return
	}
	defer watcher.Close()

	cfgFile := s.configPath
	cfgDir := filepath.Dir(cfgFile)

	// Watch both the file and its directory to handle symlink swaps (Kubernetes ConfigMap)
	if err := watcher.Add(cfgDir); err != nil {
		observability.LogEvent("config_watcher_error", map[string]interface{}{
			"stage": "watch_dir",
			"dir":   cfgDir,
			"error": err.Error(),
		})
		return
	}
	_ = watcher.Add(cfgFile) // best-effort; may fail if file replaced by symlink later

	// Debounce events
	var (
		pending bool
		last    time.Time
	)

	reload := func() {
		// Parse and build a new router
		newRouter, err := NewOpenAIRouter(cfgFile)
		if err != nil {
			observability.LogEvent("config_reload_failed", map[string]interface{}{
				"file":  cfgFile,
				"error": err.Error(),
			})
			return
		}
		s.service.Swap(newRouter)
		observability.LogEvent("config_reloaded", map[string]interface{}{
			"file": cfgFile,
		})
	}

	for {
		select {
		case <-ctx.Done():
			return
		case ev, ok := <-watcher.Events:
			if !ok {
				return
			}
			if ev.Op&(fsnotify.Write|fsnotify.Create|fsnotify.Rename|fsnotify.Remove|fsnotify.Chmod) != 0 {
				// If the event pertains to the config file or directory, trigger debounce
				if filepath.Base(ev.Name) == filepath.Base(cfgFile) || filepath.Dir(ev.Name) == cfgDir {
					if !pending || time.Since(last) > 250*time.Millisecond {
						pending = true
						last = time.Now()
						// Slight delay to let file settle
						go func() { time.Sleep(300 * time.Millisecond); reload() }()
					}
				}
			}
		case err, ok := <-watcher.Errors:
			if !ok {
				return
			}
			observability.LogEvent("config_watcher_error", map[string]interface{}{
				"stage": "watch_loop",
				"error": err.Error(),
			})
		}
	}
}
