package extproc

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"net"
	"path/filepath"
	"sync"
	"sync/atomic"
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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
	tlsutil "github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/tls"
)

const defaultGenerationDrainTimeout = 30 * time.Second

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
	configPath  string
	service     *RouterService
	server      *grpc.Server
	port        int
	secure      bool
	certPath    string
	runtime     *routerruntime.Registry
	stopOnce    sync.Once
	stopErr     error
	reloadMu    sync.Mutex
	stopping    atomic.Bool
	watchMu     sync.Mutex
	watchCancel context.CancelFunc
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
	service := NewRouterService(router)
	publishRouterState(router.Config, router, runtimeRegistry)
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

// Start serves requests until Stop is called or the gRPC server fails.
func (s *Server) Start() error {
	return s.StartContext(context.Background())
}

// StartContext serves requests until ctx is cancelled or the gRPC server fails.
func (s *Server) StartContext(ctx context.Context) error {
	if ctx == nil {
		ctx = context.Background()
	}
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", s.port))
	if err != nil {
		s.Stop()
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
				_ = lis.Close()
				s.Stop()
				return fmt.Errorf("failed to load TLS certificate from %s: %w", s.certPath, err)
			}
			logging.ComponentEvent("extproc", "tls_certificate_loaded", map[string]interface{}{
				"path": s.certPath,
			})
		} else {
			// Create self-signed certificate
			cert, err = tlsutil.CreateSelfSignedTLSCertificate()
			if err != nil {
				_ = lis.Close()
				s.Stop()
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
	watchCtx, cancel := context.WithCancel(ctx)
	s.watchMu.Lock()
	if s.stopping.Load() {
		cancel()
	} else {
		s.watchCancel = cancel
	}
	s.watchMu.Unlock()
	defer func() {
		cancel()
		s.watchMu.Lock()
		s.watchCancel = nil
		s.watchMu.Unlock()
	}()
	go s.watchConfigAndReload(watchCtx)

	// Process signal ownership belongs to the command entrypoint. This server
	// only observes the lifecycle context supplied by its caller.
	select {
	case err := <-serverErrCh:
		if err != nil {
			logging.ComponentErrorEvent("extproc", "server_stopped_with_error", map[string]interface{}{
				"port":  s.port,
				"error": err.Error(),
			})
			s.Stop()
			return err
		}
	case <-ctx.Done():
		logging.ComponentEvent("extproc", "server_shutdown_requested", map[string]interface{}{
			"port": s.port,
		})
	}
	return nil
}

// Stop stops the gRPC server
func (s *Server) Stop() {
	ctx, cancel := context.WithTimeout(context.Background(), defaultGenerationDrainTimeout)
	defer cancel()
	_ = s.Shutdown(ctx)
}

func (s *Server) Shutdown(ctx context.Context) error {
	if ctx == nil {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(context.Background(), defaultGenerationDrainTimeout)
		defer cancel()
	}

	s.stopOnce.Do(func() { s.stopErr = s.shutdown(ctx) })
	return s.stopErr
}

func (s *Server) shutdown(ctx context.Context) error {
	s.stopping.Store(true)
	s.watchMu.Lock()
	if s.watchCancel != nil {
		s.watchCancel()
	}
	s.watchMu.Unlock()
	if s.server != nil {
		gracefulDone := make(chan struct{})
		go func() {
			s.server.GracefulStop()
			close(gracefulDone)
		}()
		select {
		case <-gracefulDone:
		case <-ctx.Done():
			s.server.Stop()
			<-gracefulDone
		}
		logging.ComponentEvent("extproc", "server_stopped", map[string]interface{}{
			"port": s.port,
		})
	}
	if s.service != nil {
		return s.service.Shutdown(ctx)
	}
	return nil
}

// RouterService is a delegating gRPC service that forwards to the current router implementation.
type RouterService struct {
	current atomic.Pointer[routerGeneration]
	mu      sync.Mutex
	closed  bool
	retired sync.WaitGroup
	errMu   sync.Mutex
	errors  []error
}

type routerGeneration struct {
	router *OpenAIRouter

	mu      sync.Mutex
	refs    sync.WaitGroup
	retired bool
	drained chan struct{}
}

func NewRouterService(r *OpenAIRouter) *RouterService {
	rs := &RouterService{}
	rs.current.Store(newRouterGeneration(r))
	return rs
}

func newRouterGeneration(router *OpenAIRouter) *routerGeneration {
	generation := &routerGeneration{
		router:  router,
		drained: make(chan struct{}),
	}
	if router != nil {
		router.routerLearningMu.Lock()
		router.generation = generation
		if router.routerLearningRuntime != nil {
			router.routerLearningRuntime.generation = generation
		}
		router.routerLearningMu.Unlock()
	}
	return generation
}

func (g *routerGeneration) acquire() (func(), bool) {
	if g == nil {
		return nil, false
	}
	g.mu.Lock()
	if g.retired {
		g.mu.Unlock()
		return nil, false
	}
	g.refs.Add(1)
	g.mu.Unlock()

	var once sync.Once
	return func() {
		once.Do(g.refs.Done)
	}, true
}

func (g *routerGeneration) retire() {
	if g == nil {
		return
	}
	g.mu.Lock()
	if g.retired {
		g.mu.Unlock()
		return
	}
	g.retired = true
	g.mu.Unlock()

	go func() {
		g.refs.Wait()
		close(g.drained)
	}()
}

// Swap replaces the current router implementation and closes the retired
// generation after every stream that leased it has returned. The current
// pointer is swapped before publish, but Process must take rs.mu and therefore
// cannot observe the new generation until the management snapshot is also
// published and this critical section ends.
func (rs *RouterService) Swap(r *OpenAIRouter, publish func()) error {
	rs.mu.Lock()
	if rs.closed {
		rs.mu.Unlock()
		if r != nil {
			_ = r.Close()
		}
		return errors.New("router service is shutting down")
	}
	old := rs.current.Swap(newRouterGeneration(r))
	if publish != nil {
		publish()
	}
	if old != nil {
		old.retire()
		rs.retired.Add(1)
	}
	rs.mu.Unlock()
	if old != nil {
		go rs.closeRetiredGeneration(old)
	}
	return nil
}

// GetRouter returns the current router implementation.
func (rs *RouterService) GetRouter() *OpenAIRouter {
	generation := rs.current.Load()
	if generation == nil {
		return nil
	}
	return generation.router
}

// Process delegates to the current router.
func (rs *RouterService) Process(stream ext_proc.ExternalProcessor_ProcessServer) error {
	rs.mu.Lock()
	generation := rs.current.Load()
	if generation == nil {
		rs.mu.Unlock()
		return errors.New("router is shutting down")
	}
	release, acquired := generation.acquire()
	rs.mu.Unlock()
	if !acquired {
		return errors.New("router is shutting down")
	}
	defer release()
	return generation.router.Process(stream)
}

func (rs *RouterService) Close() error {
	ctx, cancel := context.WithTimeout(context.Background(), defaultGenerationDrainTimeout)
	defer cancel()
	return rs.Shutdown(ctx)
}

func (rs *RouterService) Shutdown(ctx context.Context) error {
	if ctx == nil {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(context.Background(), defaultGenerationDrainTimeout)
		defer cancel()
	}

	rs.mu.Lock()
	if !rs.closed {
		rs.closed = true
		generation := rs.current.Swap(nil)
		if generation != nil {
			generation.retire()
			rs.retired.Add(1)
			go rs.closeRetiredGeneration(generation)
		}
	}
	rs.mu.Unlock()

	done := make(chan struct{})
	go func() {
		rs.retired.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-ctx.Done():
		return ctx.Err()
	}

	rs.errMu.Lock()
	defer rs.errMu.Unlock()
	return errors.Join(rs.errors...)
}

func (rs *RouterService) closeRetiredGeneration(generation *routerGeneration) {
	defer rs.retired.Done()
	<-generation.drained
	if generation.router == nil {
		return
	}
	if err := generation.router.Close(); err != nil {
		rs.errMu.Lock()
		rs.errors = append(rs.errors, err)
		rs.errMu.Unlock()
		logging.ComponentErrorEvent("extproc", "retired_router_close_failed", map[string]interface{}{
			"error": err.Error(),
		})
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
	s.reloadMu.Lock()
	defer s.reloadMu.Unlock()
	if s.stopping.Load() {
		return errors.New("router server is shutting down")
	}
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
	inheritRouterLearningState(s.service.GetRouter(), newRouter)

	// Kubernetes updates are already published through config.Replace in the
	// controller callback. Replacing again here would re-enqueue the same config
	// update and can cause duplicate reload notifications.
	if source != "kubernetes" && s.runtime == nil {
		replaceReloadConfig(candidateCfg)
	}
	logLoadedRouterConfig(configPath, candidateCfg)
	if err := s.service.Swap(newRouter, func() {
		publishRouterState(candidateCfg, newRouter, s.runtime)
	}); err != nil {
		return err
	}
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
	publishRouterLearningStateStore(router)
	if runtimeRegistry != nil {
		runtimeRegistry.PublishRouterRuntimeSnapshot(routerruntime.RouterRuntimeSnapshot{
			Config:                cfg,
			ClassificationService: router.ClassificationService,
			MemoryStore:           router.MemoryStore,
			ModelSelector:         router.ModelSelector,
			LearningRuntime:       router.routerLearningRuntimeState(),
			ReplayRuntime:         router,
			ResponseCache:         router.responseCacheService(),
			ContextCompression:    router.contextCompressionService(),
			CompressionRecovery:   router.CompressionRecovery,
		})
		return
	}
	services.SetGlobalClassificationService(router.ClassificationService)
	memory.SetGlobalMemoryStore(router.MemoryStore)
	selection.SetGlobalRegistry(router.ModelSelector)
}
