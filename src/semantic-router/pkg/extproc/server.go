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
	"runtime/debug"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/fsnotify/fsnotify"
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
		_, err := modelruntime.WarmupToolsDatabase(context.Background(), state.ToolsReady, router.LoadToolsDatabase, modelruntime.WarmupToolsOptions{
			Component:      "extproc",
			MaxParallelism: 1,
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
}

// NewServer creates a new ExtProc gRPC server
func NewServer(
	configPath string,
	port int,
	secure bool,
	certPath string,
	runtimeRegistry *routerruntime.Registry,
) (*Server, error) {
	router, err := NewOpenAIRouter(configPath)
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

	maxMsgSize := config.Get().Looper.GetGRPCMaxMsgSize()
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

// Stop stops the gRPC server
func (s *Server) Stop() {
	if s.server != nil {
		s.server.GracefulStop()
		logging.ComponentEvent("extproc", "server_stopped", map[string]interface{}{
			"port": s.port,
		})
	}
}

func shouldReloadForConfigEvent(cfgFile, cfgDir, eventPath string) bool {
	if eventPath == "" {
		return false
	}

	cleanEventPath := filepath.Clean(eventPath)
	if cleanEventPath == filepath.Clean(cfgFile) {
		return true
	}

	if filepath.Dir(cleanEventPath) != filepath.Clean(cfgDir) {
		return false
	}

	base := filepath.Base(cleanEventPath)
	if base == filepath.Base(cfgFile) {
		return true
	}
	if strings.HasPrefix(base, ".vllm-sr-write-check-") {
		return false
	}

	return strings.HasPrefix(base, "..data")
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

// GetRouter returns the current router implementation.
func (rs *RouterService) GetRouter() *OpenAIRouter {
	return rs.current.Load()
}

// Process delegates to the current router.
func (rs *RouterService) Process(stream ext_proc.ExternalProcessor_ProcessServer) error {
	r := rs.current.Load()
	return r.Process(stream)
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
	if source != "kubernetes" {
		replaceReloadConfig(candidateCfg)
	}
	logLoadedRouterConfig(configPath, candidateCfg)
	oldRouter := s.service.GetRouter()
	s.service.Swap(newRouter)
	if oldRouter != nil {
		_ = oldRouter.Close()
	}
	publishRouterState(candidateCfg, newRouter, s.runtime)
	return nil
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

// watchConfigAndReload watches the config file and reloads router on changes.
func (s *Server) watchConfigAndReload(ctx context.Context) {
	// Check if we're using Kubernetes config source
	cfg := config.Get()
	if cfg != nil && cfg.ConfigSource == config.ConfigSourceKubernetes {
		logging.ComponentEvent("extproc", "config_update_watch_started", map[string]interface{}{
			"source": "kubernetes",
		})
		// Watch for config updates from the Kubernetes controller
		s.watchKubernetesConfigUpdates(ctx)
		return
	}

	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		logging.ComponentErrorEvent("extproc", "config_watcher_error", map[string]interface{}{
			"stage": "create_watcher",
			"error": err.Error(),
		})
		return
	}
	defer func() {
		_ = watcher.Close()
	}()

	cfgFile := s.configPath
	cfgDir := filepath.Dir(cfgFile)

	// Watch both the file and its directory to handle symlink swaps (Kubernetes ConfigMap)
	if err := watcher.Add(cfgDir); err != nil {
		logging.ComponentErrorEvent("extproc", "config_watcher_error", map[string]interface{}{
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
		logging.ComponentDebugEvent("extproc", "config_reload_triggered", map[string]interface{}{
			"file": cfgFile,
		})

		if info, err := os.Stat(cfgFile); err == nil {
			logging.ComponentDebugEvent("extproc", "config_file_stat", map[string]interface{}{
				"file":       cfgFile,
				"size_bytes": info.Size(),
				"mod_time":   info.ModTime().Format("2006-01-02 15:04:05"),
			})
		} else {
			logging.ComponentDebugEvent("extproc", "config_file_stat_failed", map[string]interface{}{
				"file":  cfgFile,
				"error": err.Error(),
			})
		}

		err := s.reloadRouterFromFile(cfgFile)
		if err != nil {
			event := map[string]interface{}{
				"file":  cfgFile,
				"error": err.Error(),
			}
			if strings.Contains(err.Error(), "model download preflight failed") {
				event["stage"] = "model_download"
			}
			logging.ComponentErrorEvent("extproc", "config_reload_failed", event)
			return
		}

		newRouter := s.service.GetRouter()
		event := map[string]interface{}{
			"file": cfgFile,
		}
		if newRouter != nil && newRouter.Config != nil {
			event["decision_count"] = len(newRouter.Config.Decisions)
		}
		logging.ComponentEvent("extproc", "config_reloaded", event)
	}

	for {
		select {
		case <-ctx.Done():
			return
		case ev, ok := <-watcher.Events:
			if !ok {
				return
			}
			logging.ComponentDebugEvent("extproc", "config_watcher_event", map[string]interface{}{
				"name": ev.Name,
				"op":   ev.Op.String(),
			})
			if ev.Op&(fsnotify.Write|fsnotify.Create|fsnotify.Rename|fsnotify.Remove|fsnotify.Chmod) != 0 {
				if shouldReloadForConfigEvent(cfgFile, cfgDir, ev.Name) {
					if !pending || time.Since(last) > 250*time.Millisecond {
						pending = true
						last = time.Now()
						logging.ComponentEvent("extproc", "config_reload_scheduled", map[string]interface{}{
							"file":     ev.Name,
							"event":    ev.Op.String(),
							"delay_ms": 300,
						})
						// Slight delay to let file settle
						go func() {
							defer func() {
								if rec := recover(); rec != nil {
									logging.Errorf("Config reload goroutine: recovered panic: %v\n%s", rec, debug.Stack())
								}
							}()
							time.Sleep(300 * time.Millisecond)
							reload()
						}()
					} else {
						logging.ComponentDebugEvent("extproc", "config_reload_debounced", map[string]interface{}{
							"file": ev.Name,
						})
					}
				}
			}
		case err, ok := <-watcher.Errors:
			if !ok {
				return
			}
			logging.ComponentErrorEvent("extproc", "config_watcher_error", map[string]interface{}{
				"stage": "watch_loop",
				"error": err.Error(),
			})
		}
	}
}

// watchKubernetesConfigUpdates watches for config updates from the Kubernetes controller
func (s *Server) watchKubernetesConfigUpdates(ctx context.Context) {
	subscription := config.SubscribeConfigUpdates(1)
	defer subscription.Close()
	updateCh := subscription.Updates()

	for {
		select {
		case <-ctx.Done():
			return
		case newCfg := <-updateCh:
			if newCfg == nil {
				continue
			}

			err := s.reloadRouterFromConfig("kubernetes", s.configPath, newCfg)
			if err != nil {
				logging.ComponentErrorEvent("extproc", "config_reload_failed", map[string]interface{}{
					"source": "kubernetes",
					"error":  err.Error(),
				})
				continue
			}

			logging.ComponentEvent("extproc", "config_reloaded", map[string]interface{}{
				"source":         "kubernetes",
				"decision_count": len(newCfg.Decisions),
			})
		}
	}
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
		return
	}
	services.SetGlobalClassificationService(router.ClassificationService)
	memory.SetGlobalMemoryStore(router.MemoryStore)
}
