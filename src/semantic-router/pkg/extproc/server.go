package extproc

import (
	"context"
	"crypto/tls"
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
	tlsutil "github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/tls"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
)

// Server represents a gRPC server for the Envoy ExtProc
type Server struct {
	configPath string
	service    *RouterService
	server     *grpc.Server
	port       int
	secure     bool
	certPath   string
}

// NewServer creates a new ExtProc gRPC server
func NewServer(configPath string, port int, secure bool, certPath string) (*Server, error) {
	router, err := NewOpenAIRouter(configPath)
	if err != nil {
		return nil, err
	}

	service := NewRouterService(router)
	return &Server{
		configPath: configPath,
		service:    service,
		port:       port,
		secure:     secure,
		certPath:   certPath,
	}, nil
}

// Start starts the gRPC server
func (s *Server) Start() error {
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", s.port))
	if err != nil {
		return fmt.Errorf("failed to listen on port %d: %w", s.port, err)
	}

	// Configure server options based on secure mode
	var serverOpts []grpc.ServerOption

	if s.secure {
		var cert tls.Certificate
		var err error

		if s.certPath != "" {
			// Load certificate from provided path
			certFile := filepath.Join(s.certPath, "tls.crt")
			keyFile := filepath.Join(s.certPath, "tls.key")
			cert, err = tls.LoadX509KeyPair(certFile, keyFile)
			if err != nil {
				return fmt.Errorf("failed to load TLS certificate from %s: %w", s.certPath, err)
			}
			observability.Infof("Loaded TLS certificate from %s", s.certPath)
		} else {
			// Create self-signed certificate
			cert, err = tlsutil.CreateSelfSignedTLSCertificate()
			if err != nil {
				return fmt.Errorf("failed to create self-signed certificate: %w", err)
			}
			observability.Infof("Created self-signed TLS certificate")
		}

		creds := credentials.NewTLS(&tls.Config{
			Certificates: []tls.Certificate{cert},
		})
		serverOpts = append(serverOpts, grpc.Creds(creds))
		observability.Infof("Starting secure LLM Router ExtProc server on port %d...", s.port)
	} else {
		observability.Infof("Starting insecure LLM Router ExtProc server on port %d...", s.port)
	}

	s.server = grpc.NewServer(serverOpts...)
	ext_proc.RegisterExternalProcessorServer(s.server, s.service)

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
