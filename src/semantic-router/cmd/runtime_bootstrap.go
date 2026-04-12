package main

import (
	"context"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apiserver"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
)

type runtimeOptions struct {
	configPath   string
	certPath     string
	kubeconfig   string
	namespace    string
	port         int
	apiPort      int
	metricsPort  int
	enableAPI    bool
	secure       bool
	downloadOnly bool
}

func parseRuntimeOptions() runtimeOptions {
	var (
		configPath   = flag.String("config", "config/config.yaml", "Path to the configuration file")
		port         = flag.Int("port", 50051, "Port to listen on for gRPC ExtProc")
		apiPort      = flag.Int("api-port", 8080, "Port to listen on for the router apiserver")
		metricsPort  = flag.Int("metrics-port", 9190, "Port for Prometheus metrics")
		enableAPI    = flag.Bool("enable-api", true, "Enable the router apiserver")
		secure       = flag.Bool("secure", false, "Enable secure gRPC server with TLS")
		certPath     = flag.String("cert-path", "", "Path to TLS certificate directory (containing tls.crt and tls.key)")
		kubeconfig   = flag.String("kubeconfig", "", "Path to kubeconfig file (optional, uses in-cluster config if not specified)")
		namespace    = flag.String("namespace", "default", "Kubernetes namespace to watch for CRDs")
		downloadOnly = flag.Bool("download-only", false, "Download required models and exit (useful for CI/testing)")
	)
	flag.Parse()

	return runtimeOptions{
		configPath:   *configPath,
		certPath:     *certPath,
		kubeconfig:   *kubeconfig,
		namespace:    *namespace,
		port:         *port,
		apiPort:      *apiPort,
		metricsPort:  *metricsPort,
		enableAPI:    *enableAPI,
		secure:       *secure,
		downloadOnly: *downloadOnly,
	}
}

func initializeRuntimeLogger() {
	if _, err := logging.InitLoggerFromEnv(); err != nil {
		fmt.Fprintf(os.Stderr, "failed to initialize logger: %v\n", err)
	}
}

func loadRuntimeConfigOrFatal(configPath string) *config.RouterConfig {
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		logging.ComponentFatalEvent("router", "runtime_config_missing", map[string]interface{}{
			"config_path": configPath,
		})
	}

	cfg, err := config.Parse(configPath)
	if err != nil {
		logging.ComponentFatalEvent("router", "runtime_config_load_failed", map[string]interface{}{
			"config_path": configPath,
			"error":       err.Error(),
		})
	}
	logging.ComponentDebugEvent("router", "runtime_config_loaded", map[string]interface{}{
		"config_path":    configPath,
		"config_source":  cfg.ConfigSource,
		"decision_count": len(cfg.Decisions),
	})
	return cfg
}

func newStartupWriter(cfg *config.RouterConfig, configPath string) startupstatus.StatusWriter {
	writer := buildStartupWriter(cfg, configPath)
	writeStartupState(writer, startupstatus.State{
		Phase:   "starting",
		Ready:   false,
		Message: "Router process booting...",
	}, "Failed to write initial startup status")
	return writer
}

func buildStartupWriter(cfg *config.RouterConfig, configPath string) startupstatus.StatusWriter {
	if cfg.StartupStatus.StoreBackend == "redis" && cfg.StartupStatus.Redis != nil {
		rw, err := startupstatus.NewRedisWriter(startupstatus.RedisWriterConfig{
			Address:  cfg.StartupStatus.Redis.Address,
			Password: cfg.StartupStatus.Redis.Password,
			DB:       cfg.StartupStatus.Redis.DB,
		})
		if err != nil {
			logging.ComponentWarnEvent("router", "startup_status_redis_fallback", map[string]interface{}{
				"error":    err.Error(),
				"fallback": "file",
			})
			return startupstatus.NewFileWriter(configPath)
		}
		logging.ComponentEvent("router", "startup_status_backend", map[string]interface{}{
			"backend": "redis",
			"address": cfg.StartupStatus.Redis.Address,
		})
		return rw
	}

	logging.ComponentWarnEvent("router", "startup_status_file_backend", map[string]interface{}{
		"backend": "file",
		"message": "Startup status using local file backend. Status is not shared across replicas or visible to the dashboard in containerized deployments. Set startup_status.backend: redis for production use.",
	})
	return startupstatus.NewFileWriter(configPath)
}

func writeStartupState(writer startupstatus.StatusWriter, state startupstatus.State, warning string) {
	if err := writer.Write(state); err != nil {
		logging.ComponentWarnEvent("router", "startup_state_write_failed", map[string]interface{}{
			"warning": warning,
			"phase":   state.Phase,
			"ready":   state.Ready,
			"error":   err.Error(),
		})
	}
}

func failStartup(writer startupstatus.StatusWriter, format string, args ...interface{}) {
	message := fmt.Sprintf(format, args...)
	_ = writer.Write(startupstatus.State{
		Phase:   "error",
		Ready:   false,
		Message: message,
	})
	logging.ComponentFatalEvent("router", "startup_failed", map[string]interface{}{
		"message": message,
	})
}

func ensureModelsDownloadedOrFatal(cfg *config.RouterConfig, writer startupstatus.StatusWriter) {
	if err := ensureModelsDownloaded(cfg, writer); err != nil {
		failStartup(writer, "Failed to ensure models are downloaded: %v", err)
	}
}

func exitIfDownloadOnly(downloadOnly bool) {
	if !downloadOnly {
		return
	}

	logging.ComponentEvent("router", "download_only_complete", map[string]interface{}{
		"mode": "download_only",
	})
	os.Exit(0)
}

func initializeTracing(cfg *config.RouterConfig) func() {
	if !cfg.Observability.Tracing.Enabled {
		return func() {}
	}

	tracingCfg := tracing.TracingConfig{
		Enabled:               cfg.Observability.Tracing.Enabled,
		Provider:              cfg.Observability.Tracing.Provider,
		ExporterType:          cfg.Observability.Tracing.Exporter.Type,
		ExporterEndpoint:      cfg.Observability.Tracing.Exporter.Endpoint,
		ExporterInsecure:      cfg.Observability.Tracing.Exporter.Insecure,
		SamplingType:          cfg.Observability.Tracing.Sampling.Type,
		SamplingRate:          cfg.Observability.Tracing.Sampling.Rate,
		ServiceName:           cfg.Observability.Tracing.Resource.ServiceName,
		ServiceVersion:        cfg.Observability.Tracing.Resource.ServiceVersion,
		DeploymentEnvironment: cfg.Observability.Tracing.Resource.DeploymentEnvironment,
	}
	if err := tracing.InitTracing(context.Background(), tracingCfg); err != nil {
		logging.ComponentWarnEvent("router", "tracing_init_failed", map[string]interface{}{
			"provider":          tracingCfg.Provider,
			"exporter_type":     tracingCfg.ExporterType,
			"exporter_endpoint": tracingCfg.ExporterEndpoint,
			"error":             err.Error(),
		})
	}
	return shutdownTracing
}

func shutdownTracing() {
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := tracing.ShutdownTracing(shutdownCtx); err != nil {
		logging.ComponentErrorEvent("router", "tracing_shutdown_failed", map[string]interface{}{
			"error": err.Error(),
		})
	}
}

func initializeWindowedMetricsIfEnabled(cfg *config.RouterConfig) {
	if !cfg.Observability.Metrics.WindowedMetrics.Enabled {
		return
	}

	if err := metrics.InitializeWindowedMetrics(cfg.Observability.Metrics.WindowedMetrics); err != nil {
		logging.ComponentWarnEvent("router", "windowed_metrics_init_failed", map[string]interface{}{
			"error": err.Error(),
		})
		return
	}
	logging.ComponentEvent("router", "windowed_metrics_initialized", map[string]interface{}{
		"mode": "load_balancing",
	})
}

func registerSignalHandler(shutdownHooks *[]func()) {
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	go func() {
		sig := <-sigChan
		logging.ComponentEvent("router", "shutdown_signal_received", map[string]interface{}{
			"signal": sig.String(),
		})
		for _, hook := range *shutdownHooks {
			hook()
		}
		shutdownTracing()
		os.Exit(0)
	}()
}

func startMetricsServerIfEnabled(cfg *config.RouterConfig, metricsPort int) {
	metricsEnabled := true
	if cfg.Observability.Metrics.Enabled != nil {
		metricsEnabled = *cfg.Observability.Metrics.Enabled
	}
	if metricsPort <= 0 {
		metricsEnabled = false
	}
	if !metricsEnabled {
		logging.ComponentEvent("router", "metrics_server_disabled", map[string]interface{}{
			"metrics_port": metricsPort,
		})
		return
	}

	go func() {
		http.Handle("/metrics", promhttp.Handler())
		metricsAddr := fmt.Sprintf(":%d", metricsPort)
		logging.ComponentEvent("router", "metrics_server_starting", map[string]interface{}{
			"address": metricsAddr,
		})
		if err := http.ListenAndServe(metricsAddr, nil); err != nil {
			logging.ComponentErrorEvent("router", "metrics_server_failed", map[string]interface{}{
				"address": metricsAddr,
				"error":   err.Error(),
			})
		}
	}()
}

func initializeRuntimeDependencies(
	cfg *config.RouterConfig,
	writer startupstatus.StatusWriter,
	shutdownHooks *[]func(),
	runtimeRegistry *routerruntime.Registry,
) modelruntime.EmbeddingRuntimeState {
	writeStartupState(writer, startupstatus.State{
		Phase:   "initializing_models",
		Ready:   false,
		Message: "Initializing embedding models and router dependencies...",
	}, "Failed to write initialization startup status")

	embeddingState, err := modelruntime.PrepareRouterRuntime(context.Background(), cfg, modelruntime.PrepareRouterRuntimeOptions{
		Component:                  "router",
		MaxParallelism:             modelruntime.DefaultParallelism(5),
		OnEvent:                    logRuntimeLifecycleEvent,
		InitModalityClassifierFunc: extproc.InitModalityClassifier,
	})
	if err != nil {
		failStartup(writer, "Failed to initialize runtime dependencies: %v", err)
	}

	initializeVectorStoreIfEnabled(cfg, shutdownHooks, runtimeRegistry)
	return embeddingState
}

func logRuntimeLifecycleEvent(event modelruntime.Event) {
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
		logging.ComponentWarnEvent("router", "runtime_lifecycle_task_skipped", payload)
		return
	}
	if event.BestEffort {
		logging.ComponentWarnEvent("router", "runtime_lifecycle_task_failed", payload)
		return
	}
	logging.ComponentErrorEvent("router", "runtime_lifecycle_task_failed", payload)
}

func initializeVectorStoreIfEnabled(
	cfg *config.RouterConfig,
	shutdownHooks *[]func(),
	runtimeRegistry *routerruntime.Registry,
) {
	if cfg.VectorStore == nil || !cfg.VectorStore.Enabled {
		return
	}

	logging.ComponentEvent("router", "vector_store_init_started", map[string]interface{}{
		"backend": cfg.VectorStore.BackendType,
	})
	if err := cfg.VectorStore.Validate(); err != nil {
		logging.ComponentFatalEvent("router", "vector_store_config_invalid", map[string]interface{}{
			"backend": cfg.VectorStore.BackendType,
			"error":   err.Error(),
		})
	}
	vectorStoreRuntime, err := routerruntime.NewVectorStoreRuntime(cfg)
	if err != nil {
		logging.ComponentFatalEvent("router", "vector_store_runtime_create_failed", map[string]interface{}{
			"backend": cfg.VectorStore.BackendType,
			"error":   err.Error(),
		})
	}
	if runtimeRegistry != nil {
		runtimeRegistry.SetVectorStoreRuntime(vectorStoreRuntime)
	}
	vectorStoreRuntime.LogInitialized("router", cfg)
	registerVectorStoreShutdownHook(shutdownHooks, vectorStoreRuntime)
}

func registerVectorStoreShutdownHook(
	shutdownHooks *[]func(),
	vectorStoreRuntime *routerruntime.VectorStoreRuntime,
) {
	*shutdownHooks = append(*shutdownHooks, func() {
		logging.ComponentEvent("router", "vector_store_shutdown_started", map[string]interface{}{})
		if err := vectorStoreRuntime.Shutdown(); err != nil {
			logging.ComponentErrorEvent("router", "vector_store_shutdown_failed", map[string]interface{}{
				"error": err.Error(),
			})
		}
	})
}

func newExtProcServerOrFatal(
	opts runtimeOptions,
	writer startupstatus.StatusWriter,
	runtimeRegistry *routerruntime.Registry,
) *extproc.Server {
	server, err := extproc.NewServer(opts.configPath, opts.port, opts.secure, opts.certPath, runtimeRegistry)
	if err != nil {
		failStartup(writer, "Failed to create ExtProc server: %v", err)
	}

	return server
}

func warmupRouterRuntime(server *extproc.Server, embeddingState modelruntime.EmbeddingRuntimeState) {
	router := server.GetRouter()
	if router == nil {
		return
	}
	_, _ = modelruntime.WarmupToolsDatabase(context.Background(), embeddingState.ToolsReady, router.LoadToolsDatabase, modelruntime.WarmupToolsOptions{
		Component:      "router",
		MaxParallelism: 1,
		OnEvent:        logRuntimeLifecycleEvent,
	})
}

func startAPIServerIfEnabled(opts runtimeOptions, runtimeRegistry *routerruntime.Registry) {
	if !opts.enableAPI {
		return
	}

	go func() {
		logging.ComponentEvent("router", "api_server_starting", map[string]interface{}{
			"api_port": opts.apiPort,
		})
		if err := apiserver.InitWithRuntime(opts.configPath, opts.apiPort, runtimeRegistry); err != nil {
			logging.ComponentErrorEvent("router", "api_server_failed", map[string]interface{}{
				"api_port": opts.apiPort,
				"error":    err.Error(),
			})
		}
	}()
}

func markRouterReady(writer startupstatus.StatusWriter) {
	writeStartupState(writer, startupstatus.State{
		Phase:   "ready",
		Ready:   true,
		Message: "Router models are ready. Starting router services...",
	}, "Failed to write ready startup status")
}

func startKubernetesControllerIfNeeded(cfg *config.RouterConfig, kubeconfig, namespace string) {
	if cfg.ConfigSource == config.ConfigSourceKubernetes {
		go startKubernetesController(cfg, kubeconfig, namespace)
	}
}

func startExtProcServerOrFatal(server *extproc.Server, writer startupstatus.StatusWriter) {
	if err := server.Start(); err != nil {
		failStartup(writer, "ExtProc server error: %v", err)
	}
}
