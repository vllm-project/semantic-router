package extproc

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"sync/atomic"
	"syscall"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modeldownload"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
	tlsutil "github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/tls"
)

var (
	parseReloadConfig        = config.Parse
	ensureReloadConfigModels = modeldownload.EnsureModelsForConfig
	buildReloadRouter        = buildOpenAIRouterFromConfig
	replaceReloadConfig      = config.Replace
	prepareReloadRuntime     = func(cfg *config.RouterConfig) (modelruntime.EmbeddingRuntimeState, error) {
		return modelruntime.PrepareRouterRuntime(context.Background(), cfg, modelruntime.PrepareRouterRuntimeOptions{
			Component:                  "extproc",
			MaxParallelism:             modelruntime.DefaultParallelism(5),
			OnEvent:                    logReloadRuntimeLifecycleEvent,
			InitModalityClassifierFunc: InitModalityClassifier,
		})
	}
	warmupReloadRouter = func(router *OpenAIRouter, state modelruntime.EmbeddingRuntimeState) error {
		if router == nil {
			return nil
		}
		_, err := modelruntime.WarmupRouter(context.Background(), []modelruntime.RouterWarmupTask{
			{
				Name:       "tools_database",
				Ready:      state.ToolsReady,
				SkipReason: "embedding_runtime_not_ready_for_tools",
				Load:       router.LoadToolsDatabase,
			},
			{
				Name:       "knowledge_bases",
				Ready:      state.AnyReady,
				SkipReason: "embedding_runtime_not_ready_for_knowledge_bases",
				Load:       router.PreloadKnowledgeBases,
			},
		}, modelruntime.WarmupRouterOptions{
			Component:      "extproc",
			MaxParallelism: 2,
			OnEvent:        logReloadRuntimeLifecycleEvent,
		})
		return err
	}
)

// Server represents a gRPC server for the Envoy ExtProc
type Server struct {
	configPath string
	service    *RouterService
	server     *grpc.Server
	port       int
	secure     bool
	certPath   string
	runtime    *routerruntime.Registry

	// shutdownHooks run, in registration order, after the gRPC server has
	// gracefully stopped. Server is the sole SIGINT/SIGTERM owner (see
	// Start), so this is the single coordinated place process-wide cleanup
	// (e.g. vector store shutdown) runs — instead of racing it against
	// Start's graceful drain via a second, independent signal.Notify.
	shutdownHooks []func()
}

// RegisterShutdownHook registers a hook to run during Stop, after the gRPC
// server has finished its graceful drain. Hooks run in registration order.
// Call before Start(); registration is not synchronized against a
// concurrently running Stop().
func (s *Server) RegisterShutdownHook(hook func()) {
	if hook == nil {
		return
	}
	s.shutdownHooks = append(s.shutdownHooks, hook)
}

// NewServer creates a new ExtProc gRPC server
func NewServer(
	configPath string,
	port int,
	secure bool,
	certPath string,
	runtimeRegistry *routerruntime.Registry,
) (*Server, error) {
	router, err := newOpenAIRouterForServer(configPath, runtimeRegistry)
	if err != nil {
		return nil, err
	}
	attachRuntimeRegistry(router, runtimeRegistry)
	publishRouterState(router.Config, router, runtimeRegistry)

	service := NewRouterService(router)
	return &Server{
		configPath: configPath,
		service:    service,
		port:       port,
		secure:     secure,
		certPath:   certPath,
		runtime:    runtimeRegistry,
	}, nil
}

// GetRouter returns the current router instance
func (s *Server) GetRouter() *OpenAIRouter {
	return s.service.GetRouter()
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
			logging.ComponentEvent("extproc", "tls_certificate_loaded", map[string]interface{}{
				"path": s.certPath,
			})
		} else {
			// Create self-signed certificate
			cert, err = tlsutil.CreateSelfSignedTLSCertificate()
			if err != nil {
				return fmt.Errorf("failed to create self-signed certificate: %w", err)
			}
			logging.ComponentEvent("extproc", "tls_certificate_created", map[string]interface{}{
				"source": "self_signed",
			})
		}

		creds := credentials.NewTLS(&tls.Config{
			Certificates: []tls.Certificate{cert},
		})
		serverOpts = append(serverOpts, grpc.Creds(creds))
	}

	maxMsgSize := s.configuredGRPCMaxMessageSize()
	serverOpts = append(serverOpts,
		grpc.MaxRecvMsgSize(maxMsgSize),
		grpc.MaxSendMsgSize(maxMsgSize),
	)
	logging.ComponentEvent("extproc", "server_starting", map[string]interface{}{
		"port":       s.port,
		"secure":     s.secure,
		"max_msg_mb": maxMsgSize / (1024 * 1024),
	})
	s.server = grpc.NewServer(serverOpts...)
	ext_proc.RegisterExternalProcessorServer(s.server, s.service)

	// Run the server in a separate goroutine
	serverErrCh := make(chan error, 1)
	go func() {
		if err := s.server.Serve(lis); err != nil && !errors.Is(err, grpc.ErrServerStopped) {
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
			logging.ComponentErrorEvent("extproc", "server_stopped_with_error", map[string]interface{}{
				"port":  s.port,
				"error": err.Error(),
			})
			return err
		}
	case <-signalChan:
		logging.ComponentEvent("extproc", "server_shutdown_requested", map[string]interface{}{
			"port": s.port,
		})
	}

	s.Stop()
	return nil
}

// Stop gracefully stops the gRPC server, draining in-flight requests, then
// runs every registered shutdown hook. Hooks run only after the drain
// completes, so process-wide cleanup (e.g. releasing the vector store) never
// races an in-flight request against resources it still needs.
func (s *Server) Stop() {
	if s.server != nil {
		s.server.GracefulStop()
		logging.ComponentEvent("extproc", "server_stopped", map[string]interface{}{
			"port": s.port,
		})
	}
	if s.service != nil {
		if err := s.service.Shutdown(defaultRouterDrainTimeout); err != nil {
			logging.ComponentWarnEvent("extproc", "router_shutdown_close_failed", map[string]interface{}{
				"error": err.Error(),
			})
		}
	}
	for _, hook := range s.shutdownHooks {
		hook()
	}
}

// defaultRouterDrainTimeout bounds how long Retire waits for a retired
// router's in-flight requests to finish before closing its resources
// anyway, so a reload is never blocked indefinitely by a stuck request.
const defaultRouterDrainTimeout = 30 * time.Second

// RouterService is a delegating gRPC service that forwards to the current router implementation.
type RouterService struct {
	current atomic.Pointer[routerLease]
}

func NewRouterService(r *OpenAIRouter) *RouterService {
	rs := &RouterService{}
	rs.current.Store(newRouterLease(r))
	return rs
}

// Swap replaces the current router implementation and returns the lease it
// replaced, so the caller can Retire that lease once its in-flight requests
// have drained.
func (rs *RouterService) Swap(r *OpenAIRouter) *routerLease {
	return rs.current.Swap(newRouterLease(r))
}

// Retire stops admitting new calls through lease, waits up to drainTimeout
// for its in-flight calls to finish, then closes its router's owned
// resources. Pass the lease returned by Swap. A nil lease (e.g. no router
// was previously set) is a no-op.
func (rs *RouterService) Retire(lease *routerLease, drainTimeout time.Duration) error {
	if lease == nil {
		return nil
	}
	lease.retire(drainTimeout)
	return lease.router.Close()
}

// Shutdown retires the currently active router: stops admitting new calls,
// waits up to drainTimeout for its in-flight calls to finish, then closes
// its owned resources. Used for final process shutdown, where — unlike a
// reload — there is no replacement router to Swap in first.
func (rs *RouterService) Shutdown(drainTimeout time.Duration) error {
	return rs.Retire(rs.current.Load(), drainTimeout)
}

// GetRouter returns the current router implementation.
func (rs *RouterService) GetRouter() *OpenAIRouter {
	lease := rs.current.Load()
	if lease == nil {
		return nil
	}
	return lease.router
}

// Process delegates to the current router, holding a lease on it for the
// duration of the call so a concurrent reload can't close its resources out
// from under an in-flight request. If a reload retires the loaded lease
// between load and acquire, it retries against the new current lease.
func (rs *RouterService) Process(stream ext_proc.ExternalProcessor_ProcessServer) error {
	for {
		lease := rs.current.Load()
		if lease == nil {
			return nil
		}
		if !lease.acquire() {
			continue
		}
		defer lease.release()
		return lease.router.Process(stream)
	}
}

func (s *Server) reloadRouterFromFile(configPath string) error {
	candidateCfg, err := parseReloadConfig(configPath)
	if err != nil {
		return err
	}

	return s.reloadRouterFromConfig("file", configPath, candidateCfg)
}

func (s *Server) reloadRouterFromConfig(
	source string,
	configPath string,
	candidateCfg *config.RouterConfig,
) error {
	if source == "file" {
		if err := ensureReloadConfigModels(candidateCfg); err != nil {
			return fmt.Errorf("model download preflight failed: %w", err)
		}
	}

	runtimeState, err := prepareReloadRuntime(candidateCfg)
	if err != nil {
		return fmt.Errorf("runtime dependency init failed: %w", err)
	}

	newRouter, err := buildReloadRouter(candidateCfg)
	if err != nil {
		return err
	}
	attachRuntimeRegistry(newRouter, s.runtime)
	if err := warmupReloadRouter(newRouter, runtimeState); err != nil {
		_ = newRouter.Close()
		return fmt.Errorf("runtime warmup failed: %w", err)
	}

	// Kubernetes updates are already published through config.Replace in the
	// controller callback. Replacing again here would re-enqueue the same config
	// update and can cause duplicate reload notifications.
	if source != "kubernetes" && s.runtime == nil {
		replaceReloadConfig(candidateCfg)
	}
	logLoadedRouterConfig(configPath, candidateCfg)
	oldLease := s.service.Swap(newRouter)
	if err := s.service.Retire(oldLease, defaultRouterDrainTimeout); err != nil {
		logging.ComponentWarnEvent("extproc", "router_retire_close_failed", map[string]interface{}{
			"error": err.Error(),
		})
	}
	publishRouterState(candidateCfg, newRouter, s.runtime)
	return nil
}

func (s *Server) configuredGRPCMaxMessageSize() int {
	cfg := resolveServerConfig(s)
	if cfg == nil {
		return (&config.LooperConfig{}).GetGRPCMaxMsgSize()
	}
	return cfg.Looper.GetGRPCMaxMsgSize()
}

func resolveServerConfig(s *Server) *config.RouterConfig {
	if s != nil && s.service != nil {
		if router := s.service.GetRouter(); router != nil && router.Config != nil {
			return router.Config
		}
	}
	if s != nil && s.runtime != nil {
		return s.runtime.CurrentConfig()
	}
	return config.Get()
}

func (s *Server) usesKubernetesConfigSource() bool {
	cfg := resolveServerConfig(s)
	return cfg != nil && cfg.ConfigSource == config.ConfigSourceKubernetes
}

func logReloadRuntimeLifecycleEvent(event modelruntime.Event) {
	if event.Status != modelruntime.TaskFailed && event.Status != modelruntime.TaskSkipped {
		return
	}

	payload := map[string]interface{}{
		"task":        event.Task,
		"best_effort": event.BestEffort,
	}
	if event.Error != nil {
		payload["error"] = event.Error.Error()
	}
	if event.Status == modelruntime.TaskSkipped {
		logging.ComponentWarnEvent("extproc", "runtime_lifecycle_task_skipped", payload)
		return
	}
	if event.BestEffort {
		logging.ComponentWarnEvent("extproc", "runtime_lifecycle_task_failed", payload)
		return
	}
	logging.ComponentErrorEvent("extproc", "runtime_lifecycle_task_failed", payload)
}

func attachRuntimeRegistry(router *OpenAIRouter, runtimeRegistry *routerruntime.Registry) {
	if router == nil {
		return
	}
	router.RuntimeRegistry = runtimeRegistry
}

func publishRouterState(
	cfg *config.RouterConfig,
	router *OpenAIRouter,
	runtimeRegistry *routerruntime.Registry,
) {
	if router == nil {
		return
	}
	if runtimeRegistry != nil {
		runtimeRegistry.PublishRouterRuntime(cfg, router.ClassificationService, router.MemoryStore)
		runtimeRegistry.SetModelSelector(router.ModelSelector)
		runtimeRegistry.SetLearningRuntime(router.routerLearningRuntimeState())
		runtimeRegistry.SetResponseCache(router.responseCacheService())
		runtimeRegistry.SetContextCompression(
			router.contextCompressionService(),
			router.CompressionRecovery,
		)
		return
	}
	services.SetGlobalClassificationService(router.ClassificationService)
	memory.SetGlobalMemoryStore(router.MemoryStore)
}
