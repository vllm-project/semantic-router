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
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/status"

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
	port       int
	secure     bool
	certPath   string
	runtime    *routerruntime.Registry

	// mu guards the shutdown state below, which Start writes and Stop reads
	// — possibly from different goroutines.
	mu sync.Mutex
	// server is the running gRPC server, published by Start once it is ready
	// to serve. Stop may run concurrently with (or entirely before) Start, so
	// it is only ever read under mu.
	server *grpc.Server
	// stopped records that Stop has already run, so a Start racing it never
	// begins serving a socket that shutdown can no longer close.
	stopped bool
	// watcherCancel stops the config-reload watcher Start launched. Stop
	// calls it before draining so a reload cannot land mid-shutdown.
	watcherCancel context.CancelFunc
	// stopOnce makes Stop idempotent. Both the process signal handler (which
	// lives in cmd, the only SIGINT/SIGTERM owner) and Start's own path after
	// serving ends call Stop, so the two can overlap; Once also blocks the
	// second caller until the first drain finishes, which is exactly the
	// contract the signal handler needs before tearing down the rest of the
	// process.
	stopOnce sync.Once
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
	grpcServer := grpc.NewServer(serverOpts...)
	ext_proc.RegisterExternalProcessorServer(grpcServer, s.service)

	// Publish the gRPC server and the config watcher's cancel under one lock,
	// and give up if Stop already ran. The signal handler owns Stop and can
	// fire at any point during startup — including after NewServer but before
	// Start — so without this check a late Start would serve traffic that
	// shutdown has already finished draining and can no longer stop.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	s.mu.Lock()
	if s.stopped {
		s.mu.Unlock()
		_ = lis.Close()
		logging.ComponentEvent("extproc", "server_start_aborted_after_stop", map[string]interface{}{
			"port": s.port,
		})
		return nil
	}
	s.server = grpcServer
	s.watcherCancel = cancel
	s.mu.Unlock()

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
	go s.watchConfigAndReload(ctx)

	// Serve until the server fails or Stop is called. Process signals are not
	// watched here: SIGINT/SIGTERM can arrive long before the server exists
	// (model init, classifier construction and warmup all run first), so the
	// handler belongs in cmd where it is live from the first line of main.
	// Stop is what ends this wait — it makes Serve return grpc.ErrServerStopped.
	if err := <-serverErrCh; err != nil {
		logging.ComponentErrorEvent("extproc", "server_stopped_with_error", map[string]interface{}{
			"port":  s.port,
			"error": err.Error(),
		})
		return err
	}

	s.Stop()
	return nil
}

// Stop gracefully stops the gRPC server, draining in-flight requests, then
// releases the resources the server owns. It runs at most once and is safe to
// call concurrently: a second caller blocks until the first Stop has finished,
// so a caller that tears down shared process resources afterwards (the signal
// handler in cmd) can never do so while requests are still draining.
func (s *Server) Stop() {
	s.stopOnce.Do(s.stop)
}

func (s *Server) stop() {
	// Flag the shutdown before reading anything Start publishes. Ordered the
	// other way, a Stop that beats Start reads a nil watcherCancel, skips the
	// cancel, and only then flags — leaving Start free to publish and launch
	// a watcher nothing ever stops. Flagging first makes the two outcomes
	// exhaustive: either Start aborts and no watcher exists, or Start won the
	// lock and published server and watcherCancel together, so both are
	// visible here.
	grpcServer := s.markStopped()

	// Stop config reloads before draining. The watcher outlives Stop
	// otherwise (Start only cancels it once Start itself returns), and a
	// reload landing mid-shutdown would Swap a fresh router into an
	// already-retired service — leaking that router and republishing runtime
	// state for a process that is going away.
	s.stopConfigWatcher()

	if grpcServer != nil {
		s.gracefulStopWithin(grpcServer, defaultRouterDrainTimeout)
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
}

func (s *Server) stopConfigWatcher() {
	s.mu.Lock()
	cancel := s.watcherCancel
	s.mu.Unlock()
	if cancel != nil {
		cancel()
	}
}

// markStopped records that shutdown has run and returns the running gRPC
// server, or nil if Start never got as far as publishing one. Flagging and
// reading under a single lock is what lets a Stop that beats Start guarantee
// Start never starts serving afterwards.
func (s *Server) markStopped() *grpc.Server {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.stopped = true
	return s.server
}

// gracefulStopWithin drains the gRPC server's in-flight streams, falling
// back to a hard Stop once timeout elapses. GracefulStop on its own is
// unbounded: ExtProc serves bidirectional streams, so a single stream that
// never ends would block shutdown until the orchestrator SIGKILLs the
// process — losing the router teardown that follows it here, and the
// process-wide cleanup the signal handler runs after Stop returns.
func (s *Server) gracefulStopWithin(grpcServer *grpc.Server, timeout time.Duration) {
	stopped := make(chan struct{})
	go func() {
		grpcServer.GracefulStop()
		close(stopped)
	}()

	select {
	case <-stopped:
	case <-time.After(timeout):
		logging.ComponentWarnEvent("extproc", "server_graceful_stop_timed_out", map[string]interface{}{
			"port":            s.port,
			"timeout_seconds": timeout.Seconds(),
		})
		grpcServer.Stop()
		<-stopped
	}
}

// defaultRouterDrainTimeout bounds how long shutdown waits for in-flight
// requests — both the gRPC graceful drain and a retired router's lease —
// before proceeding anyway, so neither a reload nor process shutdown is
// blocked indefinitely by a stuck request.
const defaultRouterDrainTimeout = 30 * time.Second

// maxLeaseAcquireAttempts bounds how many times Process re-reads the
// current lease when it finds one already retiring. Each retry means a
// reload swapped in a replacement mid-call; a handful of consecutive
// reloads is conceivable, an unbounded number is not, and bounding the loop
// guarantees Process can never spin.
const maxLeaseAcquireAttempts = 8

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
// from under an in-flight request.
func (rs *RouterService) Process(stream ext_proc.ExternalProcessor_ProcessServer) error {
	lease, err := rs.acquireCurrentLease()
	if err != nil {
		return err
	}
	defer lease.release()
	return lease.router.Process(stream)
}

// acquireCurrentLease returns the current lease with one in-flight call
// admitted against it. If a reload retires the loaded lease between load and
// acquire, it retries against the replacement.
//
// Retrying only makes sense while the service still has a live router to
// retry against. Shutdown retires the current lease without replacing it, so
// once acquire fails against the lease that is still installed, it will keep
// failing: reject the call instead of retrying forever.
func (rs *RouterService) acquireCurrentLease() (*routerLease, error) {
	for attempt := 0; attempt < maxLeaseAcquireAttempts; attempt++ {
		lease := rs.current.Load()
		if lease == nil {
			return nil, status.Error(codes.Unavailable, "router is not initialized")
		}
		if lease.acquire() {
			return lease, nil
		}
		if rs.current.Load() == lease {
			return nil, status.Error(codes.Unavailable, "router is shutting down")
		}
	}
	return nil, status.Error(codes.Unavailable, "router is reloading, please retry")
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
	// Publish before draining. The new router serves traffic the instant
	// Swap returns, so holding the runtime-registry update behind a drain
	// that can take up to defaultRouterDrainTimeout would leave the control
	// plane describing the old router while the data plane already runs the
	// new one.
	publishRouterState(candidateCfg, newRouter, s.runtime)
	if err := s.service.Retire(oldLease, defaultRouterDrainTimeout); err != nil {
		logging.ComponentWarnEvent("extproc", "router_retire_close_failed", map[string]interface{}{
			"error": err.Error(),
		})
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
	if runtimeRegistry != nil {
		runtimeRegistry.PublishRouterRuntime(cfg, router.ClassificationService, router.MemoryStore)
		runtimeRegistry.SetModelSelector(router.ModelSelector)
		runtimeRegistry.SetLearningRuntime(router.routerLearningRuntimeState())
		return
	}
	services.SetGlobalClassificationService(router.ClassificationService)
	memory.SetGlobalMemoryStore(router.MemoryStore)
}
