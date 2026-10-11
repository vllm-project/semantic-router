package extproc

import (
	"bytes"
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modeldownload"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/pluginruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
	tlsutil "github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/tls"
)

const (
	defaultGenerationDrainTimeout = 30 * time.Second
	generationDrainReserve        = 5 * time.Second
)

var (
	parseReloadConfig               = config.Parse
	ensureReloadConfigModels        = modeldownload.EnsureModelsForConfig
	buildReloadRouter               = buildOpenAIRouterFromConfig
	buildReloadRouterSharingSignals = buildOpenAIRouterSharingSignals
	replaceReloadConfig             = config.Replace

	warmupReloadRouter = func(router *OpenAIRouter) error {
		if router == nil {
			return nil
		}
		state := router.embeddingRuntimeState()
		_, err := modelruntime.WarmupRouter(context.Background(), []modelruntime.RouterWarmupTask{
			{
				Name:       "tools_database",
				Ready:      state.ToolsReady,
				SkipReason: "embedding_runtime_not_ready_for_tools",
				Load:       router.LoadToolsDatabase,
			},
			{
				Name:       "knowledge_bases",
				Ready:      state.KnowledgeBasesReady,
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
	modelPool    *binding.Pool
	configPath   string
	service      *RouterService
	server       *grpc.Server
	port         int
	secure       bool
	certPath     string
	runtime      *routerruntime.Registry
	reloadMu     sync.Mutex
	servingMu    sync.Mutex
	servingReady chan struct{}
	healthServer *health.Server
	lifecycle    serverLifecycle
	configs      *configsnapshot.Manager
	configsOnce  sync.Once
	// gateway is the mode client traffic reaches the server in; a reload
	// that needs a capability the mode lacks is rejected.
	gateway config.GatewayMode
}

// NewServer creates a new ExtProc gRPC server
func NewServer(
	configPath string,
	port int,
	secure bool,
	certPath string,
	runtimeRegistry *routerruntime.Registry,
	opts ...ServerOption,
) (*Server, error) {
	options := serverOptions{historyLimit: configsnapshot.DefaultHistoryLimit, gateway: config.GatewayExtProc}
	for _, opt := range opts {
		opt(&options)
	}
	modelPool := binding.NewPool()
	if runtimeRegistry != nil {
		modelPool = runtimeRegistry.ModelPool()
	}
	cfg, publishGlobal, err := resolveInitialRouterConfig(configPath, runtimeRegistry)
	if err != nil {
		return nil, err
	}
	var router *OpenAIRouter
	if cfg.RoutingEnabled() {
		router, err = buildOpenAIRouterFromConfig(cfg, modelPool)
		if err != nil {
			return nil, err
		}
	}
	if publishGlobal {
		config.Replace(cfg)
	}
	logLoadedRouterConfig(configPath, cfg)
	attachRuntimeRegistry(router, runtimeRegistry)
	server := &Server{
		modelPool:  modelPool,
		configPath: configPath,
		port:       port,
		secure:     secure,
		certPath:   certPath,
		runtime:    runtimeRegistry,
		gateway:    options.gateway,
	}
	server.configs = server.newConfigManager(openConfigHistory(configPath, options.historyLimit), options.parts...)
	snapshot, err := server.configManager().Install(context.Background(), configsnapshot.Update{
		Origin:   configsnapshot.Origin{Source: configsnapshot.SourceStartup},
		Config:   cfg,
		Document: parsedDocument(configPath, cfg),
	})
	if err != nil {
		return nil, errors.Join(err, router.Close())
	}
	router.nameSignals(snapshot.ComponentKey(configsnapshot.ComponentSignals))
	server.service = newRouterServiceWithSnapshot(router, snapshot)
	publishSnapshotState(cfg, snapshot, router, runtimeRegistry, server.service.current.Load().acquire)
	return server, nil
}

// GetRouter returns the current router instance
func (s *Server) GetRouter() *OpenAIRouter {
	return s.service.GetRouter()
}

// WarmupRouter loads generation-owned runtime data before serving requests.
func (s *Server) WarmupRouter(
	ctx context.Context,
	options modelruntime.WarmupRouterOptions,
) error {
	if s == nil || s.service == nil {
		return nil
	}
	generation := s.service.current.Load()
	if generation == nil || generation.router == nil {
		return nil
	}
	state := generation.router.embeddingRuntimeState()
	_, err := modelruntime.WarmupRouter(ctx, []modelruntime.RouterWarmupTask{
		{
			Name:       "tools_database",
			Ready:      state.ToolsReady,
			SkipReason: "embedding_runtime_not_ready_for_tools",
			Load: func() error {
				return generation.withLease(generation.router.LoadToolsDatabase)
			},
		},
		{
			Name:       "knowledge_bases",
			Ready:      state.KnowledgeBasesReady,
			SkipReason: "embedding_runtime_not_ready_for_knowledge_bases",
			Load: func() error {
				return generation.withLease(generation.router.PreloadKnowledgeBases)
			},
		},
	}, options)
	return err
}

// Start serves requests until Stop is called or the gRPC server fails.
func (s *Server) Start() error {
	return s.StartContext(context.Background())
}

// StartContext serves requests until ctx is cancelled or the gRPC server fails.
func (s *Server) StartContext(ctx context.Context) error {
	return s.StartContextWithReady(ctx, nil)
}

// StartContextWithReady calls onServing once the listener is accepting requests.
func (s *Server) StartContextWithReady(ctx context.Context, onServing func()) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if s.lifecycle.isStopping() {
		return errors.New("router server is shutting down")
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
	grpcServer := grpc.NewServer(serverOpts...)
	ext_proc.RegisterExternalProcessorServer(grpcServer, s.service)
	s.servingMu.Lock()
	if s.lifecycle.isStopping() {
		s.servingMu.Unlock()
		_ = lis.Close()
		return errors.New("router server is shutting down")
	}
	s.server = grpcServer
	s.healthServer = health.NewServer()
	s.healthServer.SetServingStatus("", healthpb.HealthCheckResponse_NOT_SERVING)
	healthpb.RegisterHealthServer(grpcServer, s.healthServer)
	ready := s.servingReadyLocked()
	s.servingMu.Unlock()
	lis = &servingListener{Listener: lis, onServing: func() {
		close(ready)
		if !s.usesKubernetesConfigSource() {
			s.markServingReady()
		}
		if onServing != nil {
			onServing()
		}
	}}

	// Run the server in a separate goroutine
	serverErrCh := make(chan error, 1)
	go func() {
		if err := grpcServer.Serve(lis); err != nil && !errors.Is(err, grpc.ErrServerStopped) {
			serverErrCh <- err
		} else {
			serverErrCh <- nil
		}
	}()

	// Start config file watcher in background
	watchCtx, watcherDone := s.lifecycle.startWatcher(ctx)
	defer s.lifecycle.beginShutdown()
	go func() {
		defer watcherDone()
		s.watchConfigAndReload(watchCtx)
	}()

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
		grpcServer.Stop()
		<-serverErrCh
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
	return errors.Join(s.ShutdownServing(ctx), s.ShutdownResources(ctx))
}

// ShutdownServing stops accepting ExtProc requests and drains active streams.
func (s *Server) ShutdownServing(ctx context.Context) error {
	if ctx == nil {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(context.Background(), defaultGenerationDrainTimeout)
		defer cancel()
	}

	return s.lifecycle.serving.run(ctx, func() error { return s.shutdownServing(ctx) })
}

func (s *Server) shutdownServing(ctx context.Context) error {
	s.lifecycle.beginShutdown()
	var shutdownErr error
	s.servingMu.Lock()
	grpcServer := s.server
	if s.healthServer != nil {
		s.healthServer.Shutdown()
	}
	s.servingMu.Unlock()
	if grpcServer != nil {
		gracefulCtx := ctx
		cancelGraceful := func() {}
		if deadline, ok := ctx.Deadline(); ok {
			reserve := generationDrainReserve
			budget := time.Until(deadline)
			switch {
			case budget <= 0:
				reserve = 0
			case budget < 2*reserve:
				reserve = budget / 2
			}
			gracefulCtx, cancelGraceful = context.WithDeadline(ctx, deadline.Add(-reserve))
		}
		defer cancelGraceful()

		gracefulDone := make(chan struct{})
		go func() {
			grpcServer.GracefulStop()
			close(gracefulDone)
		}()
		select {
		case <-gracefulDone:
		case <-gracefulCtx.Done():
			shutdownErr = gracefulCtx.Err()
			grpcServer.Stop()
			<-gracefulDone
		}
		logging.ComponentEvent("extproc", "server_stopped", map[string]interface{}{
			"port": s.port,
		})
	}
	return shutdownErr
}

// ShutdownResources retires the active router generation and closes it after drain.
func (s *Server) ShutdownResources(ctx context.Context) error {
	if ctx == nil {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(context.Background(), defaultGenerationDrainTimeout)
		defer cancel()
	}
	return s.lifecycle.resources.run(ctx, func() error {
		if err := s.lifecycle.stopAndWaitForBackgroundWork(context.Background()); err != nil {
			return err
		}
		if s.service != nil {
			return s.service.Shutdown(context.Background())
		}
		return nil
	})
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

// routerGeneration is the serving instance of one configuration snapshot. A
// request that leases it reads the router and the snapshot of one version.
type routerGeneration struct {
	router *OpenAIRouter
	// snapshot is nil for routers installed without the lifecycle.
	snapshot *configsnapshot.Snapshot

	mu      sync.Mutex
	refs    sync.WaitGroup
	retired bool
	drained chan struct{}
}

// AcquireFunc registers a reference on the live generation for the duration of
// one acquire. It reports false once the generation is retired, so a caller that
// loses the race against a reload falls back instead of using a closing router.
type AcquireFunc func() (release func(), ok bool)

func NewRouterService(r *OpenAIRouter) *RouterService {
	return newRouterServiceWithSnapshot(r, nil)
}

// NewRouterServiceForSnapshot returns a service whose router serves
// snapshot, as the lifecycle installs a Router's startup configuration.
func NewRouterServiceForSnapshot(r *OpenAIRouter, snapshot *configsnapshot.Snapshot) *RouterService {
	return newRouterServiceWithSnapshot(r, snapshot)
}

func newRouterServiceWithSnapshot(r *OpenAIRouter, snapshot *configsnapshot.Snapshot) *RouterService {
	rs := &RouterService{}
	rs.current.Store(newRouterGeneration(r, snapshot))
	return rs
}

func newRouterGeneration(router *OpenAIRouter, snapshot *configsnapshot.Snapshot) *routerGeneration {
	generation := &routerGeneration{
		router:   router,
		snapshot: snapshot,
		drained:  make(chan struct{}),
	}
	if router != nil {
		if snapshot != nil {
			router.configVersion.Store(snapshot.Version())
		}
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

func (g *routerGeneration) withLease(work func() error) error {
	release, acquired := g.acquire()
	if !acquired {
		return errors.New("router generation is shutting down")
	}
	defer release()
	return work()
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
func (rs *RouterService) Swap(r *OpenAIRouter, publish func(acquire AcquireFunc)) error {
	return rs.swapSnapshot(r, nil, publish)
}

// swapSnapshot is Swap for the router built from snapshot.
func (rs *RouterService) swapSnapshot(r *OpenAIRouter, snapshot *configsnapshot.Snapshot, publish func(acquire AcquireFunc)) error {
	rs.mu.Lock()
	if rs.closed {
		rs.mu.Unlock()
		if r != nil {
			_ = r.Close()
		}
		return errors.New("router service is shutting down")
	}
	generation := newRouterGeneration(r, snapshot)
	old := rs.current.Swap(generation)
	if publish != nil {
		publish(generation.acquire)
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

// Snapshot returns the configuration snapshot the current router serves, or
// nil when it was installed without the lifecycle.
func (rs *RouterService) Snapshot() *configsnapshot.Snapshot {
	generation := rs.current.Load()
	if generation == nil {
		return nil
	}
	return generation.snapshot
}

// Process delegates to the current router.
func (rs *RouterService) Process(stream ext_proc.ExternalProcessor_ProcessServer) error {
	lease, err := rs.Pin()
	if err != nil {
		return err
	}
	defer lease.Release()
	if lease.Router == nil {
		// Send an HTTP decision, not a transport failure an Envoy configured
		// with failure_mode_allow could bypass.
		if _, err := stream.Recv(); err != nil {
			return err
		}
		return stream.Send(createImmediateJSONResponse(http.StatusNotFound, []byte(`{"error":{"code":"routing_disabled","message":"Chat routing is disabled for this instance."}}`)))
	}
	return lease.Router.Process(stream)
}

// lease pins the current router generation until release is called.
func (rs *RouterService) lease() (*OpenAIRouter, func(), error) {
	pin, err := rs.Pin()
	if err != nil {
		return nil, nil, err
	}
	if pin.Router == nil {
		pin.Release()
		return nil, nil, errors.New("routing is disabled for this instance")
	}
	return pin.Router, pin.Release, nil
}

// Lease is one request's pin on a router generation: the router and the
// configuration snapshot it serves stay the same until Release, and the
// generation drains only after every lease is released.
type Lease struct {
	Router   *OpenAIRouter
	Snapshot *configsnapshot.Snapshot
	release  func()
}

// Release ends the lease. Only the first call counts.
func (l *Lease) Release() { l.release() }

// Pin leases the serving generation for one request.
func (rs *RouterService) Pin() (*Lease, error) {
	rs.mu.Lock()
	generation := rs.current.Load()
	if generation == nil {
		rs.mu.Unlock()
		return nil, errors.New("router is shutting down")
	}
	release, acquired := generation.acquire()
	rs.mu.Unlock()
	if !acquired {
		return nil, errors.New("router is shutting down")
	}
	return &Lease{Router: generation.router, Snapshot: generation.snapshot, release: release}, nil
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
			// #nosec G118 -- Cleanup must outlive the caller's wait deadline so retained requests can drain safely.
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
		return errors.Join(ctx.Err(), rs.retiredGenerationErrors())
	}

	return rs.retiredGenerationErrors()
}

func (rs *RouterService) retiredGenerationErrors() error {
	rs.errMu.Lock()
	defer rs.errMu.Unlock()
	return errors.Join(rs.errors...)
}

func (rs *RouterService) closeRetiredGeneration(generation *routerGeneration) {
	defer rs.retired.Done()
	<-generation.drained
	var err error
	if generation.router != nil {
		err = generation.router.Close()
	}
	// The snapshot's parts serve only through this generation's leases, which
	// have all ended; parts a newer snapshot shares stay open.
	releaseCtx, cancel := context.WithTimeout(context.Background(), defaultGenerationDrainTimeout)
	defer cancel()
	if releaseErr := generation.snapshot.Release(releaseCtx); releaseErr != nil {
		err = errors.Join(err, releaseErr)
	}
	if err != nil {
		rs.errMu.Lock()
		rs.errors = append(rs.errors, err)
		rs.errMu.Unlock()
		logging.ComponentErrorEvent("extproc", "retired_router_close_failed", map[string]interface{}{
			"error": err.Error(),
		})
	}
}

func (s *Server) reloadRouterFromFile(configPath string) error {
	s.reloadMu.Lock()
	defer s.reloadMu.Unlock()
	before, _ := os.ReadFile(configPath)
	candidateCfg, err := parseReloadConfig(configPath)
	if err != nil {
		// Do not attribute an older parse failure to a concurrent replacement.
		if after, readErr := os.ReadFile(configPath); readErr == nil && bytes.Equal(after, before) {
			return s.configManager().Reject(configsnapshot.Update{
				Origin: configsnapshot.Origin{Source: configsnapshot.SourceFile}, Document: before,
			}, configsnapshot.Reject(configsnapshot.StageParse, configsnapshot.CodeInvalidDocument, err))
		}
		return err
	}

	return s.reloadRouterFromConfigLocked("file", configPath, candidateCfg)
}

func (s *Server) reloadRouterFromConfig(
	source string,
	configPath string,
	candidateCfg *config.RouterConfig,
) error {
	s.reloadMu.Lock()
	defer s.reloadMu.Unlock()
	return s.reloadRouterFromConfigLocked(source, configPath, candidateCfg)
}

func (s *Server) reloadRouterFromConfigLocked(
	source string,
	configPath string,
	candidateCfg *config.RouterConfig,
) error {
	return s.reloadRouterFromConfigLockedContext(context.Background(), source, configPath, candidateCfg)
}

// reloadRouterFromConfigLockedContext hands a source's candidate to the
// configuration lifecycle, which validates, warms and activates it as the
// next snapshot, or rejects it while the active one keeps serving.
func (s *Server) reloadRouterFromConfigLockedContext(ctx context.Context, source, configPath string, candidateCfg *config.RouterConfig) error {
	if candidateCfg == nil {
		return errors.New("config reload candidate is nil")
	}
	_, err := s.configManager().Apply(ctx, sourceUpdate(source, configPath, candidateCfg))
	return err
}

// CurrentConfig returns the published generation's configuration, including
// while a Kubernetes candidate has been received but has not become ready.
func (s *Server) CurrentConfig() *config.RouterConfig { return resolveServerConfig(s) }

func (s *Server) configuredGRPCMaxMessageSize() int {
	cfg := resolveServerConfig(s)
	if cfg == nil {
		return (&config.LooperConfig{}).GetGRPCMaxMsgSize()
	}
	return cfg.Looper.GetGRPCMaxMsgSize()
}

func resolveServerConfig(s *Server) *config.RouterConfig {
	if s != nil && s.service != nil {
		if snapshot := s.service.Snapshot(); snapshot != nil {
			return snapshot.Config()
		}
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
	acquire AcquireFunc,
) {
	publishSnapshotState(cfg, nil, router, runtimeRegistry, acquire)
}

// publishSnapshotState publishes a router generation, and the snapshot it
// serves, as the management runtime.
func publishSnapshotState(
	cfg *config.RouterConfig,
	snapshot *configsnapshot.Snapshot,
	router *OpenAIRouter,
	runtimeRegistry *routerruntime.Registry,
	acquire AcquireFunc,
) {
	if router == nil {
		if runtimeRegistry != nil {
			runtimeRegistry.PublishRouterRuntimeSnapshot(routerruntime.RouterRuntimeSnapshot{
				Config: cfg, ConfigSnapshot: snapshot, AcquireClassification: routerruntime.AcquireClassification(acquire),
			})
		} else {
			services.SetGlobalClassificationService(nil)
			memory.SetGlobalMemoryStore(nil)
			selection.SetGlobalRegistry(nil)
		}
		return
	}
	publishRouterLearningStateStore(router)
	router.startServedContextWindowCheck()
	if runtimeRegistry != nil {
		runtimeRegistry.PublishRouterRuntimeSnapshot(routerruntime.RouterRuntimeSnapshot{
			Config:                cfg,
			ConfigSnapshot:        snapshot,
			ClassificationService: router.ClassificationService,
			AcquireClassification: routerruntime.AcquireClassification(acquire),
			MemoryStore:           router.MemoryStore,
			ModelSelector:         router.ModelSelector,
			LearningRuntime:       router.routerLearningRuntimeState(),
			ReplayRuntime:         router,
			ResponseCache:         router.responseCacheService(),
			ContextCompression:    router.contextCompressionService(),
			CompressionRecovery:   router.CompressionRecovery,
			Plugins:               pluginruntime.Capabilities{Guards: router, Retrieval: router, Inspector: router},
			NativeRouter:          router,
		})
		return
	}
	services.SetGlobalClassificationService(router.ClassificationService)
	memory.SetGlobalMemoryStore(router.MemoryStore)
	selection.SetGlobalRegistry(router.ModelSelector)
}

func (s *Server) EmbeddingRuntimeState() modelruntime.EmbeddingRuntimeState {
	if s == nil || s.service == nil {
		return modelruntime.EmbeddingRuntimeState{}
	}
	router := s.service.GetRouter()
	if router == nil {
		return modelruntime.EmbeddingRuntimeState{}
	}
	return router.embeddingRuntimeState()
}

func (r *OpenAIRouter) embeddingRuntimeState() modelruntime.EmbeddingRuntimeState {
	state := modelruntime.EmbeddingState(r.Config, r.Embeddings)
	state.AnyReady = state.AnyReady || r.serviceEmbeddings.Ready() || r.cacheEmbeddings.Ready() || r.RecipeClassifiers.HasAnyPreparedEmbeddings()
	state.ToolsReady = r.ToolsDatabase != nil && r.ToolsDatabase.IsEnabled() && r.serviceEmbeddings.Has("")
	if r.RecipeClassifiers != nil {
		state.KnowledgeBasesReady = r.RecipeClassifiers.HasPreparedKnowledgeBases()
	} else {
		state.KnowledgeBasesReady = r.Classifier.HasPreparedKnowledgeBases()
	}
	return state
}
