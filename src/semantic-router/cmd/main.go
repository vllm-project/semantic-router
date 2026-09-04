package main

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apiserver"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/k8s"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/logo"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modeldownload"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
)

const processShutdownTimeout = 30 * time.Second

type contextShutdowner interface {
	Shutdown(context.Context) error
}

func main() {
	logo.PrintVLLMLogo()
	opts := parseRuntimeOptions()
	initializeRuntimeLogger()
	applyBackendRuntimeTuningDefaults()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	runErr := runRouterProcess(ctx, opts)
	stop()

	if runErr != nil {
		logging.ComponentErrorEvent("router", "router_process_failed", map[string]interface{}{
			"error": runErr.Error(),
		})
		os.Exit(1)
	}
}

func runRouterProcess(ctx context.Context, opts runtimeOptions) (runErr error) {
	cfg := loadRuntimeConfigOrFatal(opts.configPath)
	config.Replace(cfg)
	runtimeRegistry := routerruntime.NewRegistry(cfg)

	startupWriter := newStartupWriter(cfg, opts.configPath)
	resolvedOpts, err := resolveRuntimeManagementOptions(opts, cfg)
	if err != nil {
		failStartup(startupWriter, "Failed to resolve management API: %v", err)
	}
	opts = resolvedOpts

	// Start the API server early so /startup-status is available during
	// model downloads and initialization.
	apiServer, err := startAPIServerIfEnabled(opts, runtimeRegistry)
	if err != nil {
		failStartup(startupWriter, "Failed to start management API: %v", err)
	}
	var routerServer *extproc.Server
	var metricsServer *http.Server
	shutdownHooks := make([]func(context.Context) error, 0)
	shutdownTracing := func(context.Context) error { return nil }
	// Return errors below so deferred shutdown can release started resources.
	defer func() {
		shutdownCtx, cancel := context.WithTimeout(context.Background(), processShutdownTimeout)
		defer cancel()
		runErr = errors.Join(runErr, shutdownRouterProcess(
			shutdownCtx,
			apiServer,
			routerServer,
			metricsServer,
			&shutdownHooks,
			shutdownTracing,
		))
	}()

	if err := ensureModelsDownloaded(ctx, cfg, startupWriter); err != nil {
		return recordStartupError(startupWriter, "ensure models are downloaded", err)
	}
	if opts.downloadOnly {
		logging.ComponentEvent("router", "download_only_complete", map[string]interface{}{
			"mode": "download_only",
		})
		return nil
	}

	shutdownTracing = initializeTracing(cfg)
	initializeWindowedMetricsIfEnabled(cfg)

	metricsServer = startMetricsServerIfEnabled(cfg, opts.metricsPort)
	startProfilingServerIfEnabled(cfg, opts, &shutdownHooks)

	embeddingRuntime, err := initializeRuntimeDependencies(ctx, cfg, startupWriter, &shutdownHooks, runtimeRegistry)
	if err != nil {
		return recordStartupError(startupWriter, "initialize runtime dependencies", err)
	}
	routerServer, err = extproc.NewServer(opts.configPath, opts.port, opts.secure, opts.certPath, runtimeRegistry)
	if err != nil {
		return recordStartupError(startupWriter, "create ExtProc server", err)
	}

	if err := warmupRouterRuntime(ctx, routerServer, embeddingRuntime); err != nil {
		return recordStartupError(startupWriter, "warm up router runtime", err)
	}
	markRouterReady(startupWriter, startupEmbeddingProviderStatus(embeddingRuntime))
	logStartupSummary(cfg, opts, embeddingRuntime.AnyReady)
	startKubernetesControllerIfNeeded(cfg, opts.kubeconfig, opts.namespace)
	return startExtProcServer(ctx, routerServer, startupWriter)
}

func shutdownRouterProcess(
	ctx context.Context,
	apiServer *apiserver.Server,
	routerServer *extproc.Server,
	metricsServer *http.Server,
	shutdownHooks *[]func(context.Context) error,
	shutdownTracing func(context.Context) error,
) error {
	servers := make([]contextShutdowner, 0, 3)
	if apiServer != nil {
		servers = append(servers, apiServer)
	}
	if routerServer != nil {
		servers = append(servers, routerServer)
	}
	if metricsServer != nil {
		servers = append(servers, metricsServer)
	}
	err := shutdownServers(ctx, servers...)
	err = errors.Join(err, runShutdownHooks(ctx, shutdownHooks))
	err = errors.Join(err, shutdownTracing(ctx))
	return err
}

func recordStartupError(writer startupstatus.StatusWriter, operation string, cause error) error {
	err := fmt.Errorf("%s: %w", operation, cause)
	_ = writer.Write(startupstatus.State{
		Phase:   "error",
		Ready:   false,
		Message: err.Error(),
	})
	logging.ComponentErrorEvent("router", "startup_failed", map[string]interface{}{
		"message": err.Error(),
	})
	return err
}

func shutdownServers(ctx context.Context, servers ...contextShutdowner) error {
	errorsByServer := make(chan error, len(servers))
	for _, server := range servers {
		go func(server contextShutdowner) {
			errorsByServer <- server.Shutdown(ctx)
		}(server)
	}
	shutdownErrors := make([]error, 0, len(servers))
	for range servers {
		shutdownErrors = append(shutdownErrors, <-errorsByServer)
	}
	return errors.Join(shutdownErrors...)
}

var (
	ensureKubernetesConfigModels   = modeldownload.EnsureModelsForConfig
	replaceKubernetesRuntimeConfig = config.Replace
)

func applyBackendRuntimeTuningDefaults() {
	backend := strings.TrimSpace(strings.ToLower(os.Getenv("EMBEDDING_BACKEND_OVERRIDE")))
	if backend != "candle" {
		return
	}

	defaults := map[string]string{
		"OMP_NUM_THREADS":        "1",
		"MKL_NUM_THREADS":        "1",
		"OPENBLAS_NUM_THREADS":   "1",
		"RAYON_NUM_THREADS":      "1",
		"TOKENIZERS_PARALLELISM": "false",
	}
	applied := make(map[string]string)
	for key, value := range defaults {
		if _, exists := os.LookupEnv(key); exists {
			continue
		}
		if err := os.Setenv(key, value); err != nil {
			logging.ComponentWarnEvent("router", "backend_runtime_tuning_setenv_failed", map[string]interface{}{
				"backend": backend,
				"env":     key,
				"error":   err.Error(),
			})
			continue
		}
		applied[key] = value
	}
	if len(applied) == 0 {
		return
	}
	logging.ComponentEvent("router", "backend_runtime_tuning_applied", map[string]interface{}{
		"backend": backend,
		"env":     applied,
	})
}

func ensureModelsDownloaded(ctx context.Context, cfg *config.RouterConfig, startupWriter startupstatus.StatusWriter) error {
	reporter := func(progress modeldownload.ProgressState) {
		state := startupstatus.State{
			Ready:            false,
			DownloadingModel: progress.DownloadingModel,
			PendingModels:    progress.PendingModels,
			ReadyModels:      progress.ReadyModels,
			TotalModels:      progress.TotalModels,
			Message:          progress.Message,
		}

		switch progress.Phase {
		case "downloading":
			state.Phase = "downloading_models"
		case "completed":
			state.Phase = "initializing_models"
			state.Message = "Required router models downloaded. Continuing startup..."
		case "skipped":
			state.Phase = "initializing_models"
		default:
			state.Phase = "checking_models"
		}

		if err := startupWriter.Write(state); err != nil {
			logging.ComponentWarnEvent("router", "model_download_progress_persist_failed", map[string]interface{}{
				"phase":             state.Phase,
				"downloading_model": state.DownloadingModel,
				"ready_models":      state.ReadyModels,
				"total_models":      state.TotalModels,
				"error":             err.Error(),
			})
		}
	}

	return modeldownload.EnsureModelsForConfigWithProgressContext(ctx, cfg, reporter)
}

func applyKubernetesConfigUpdate(newConfig *config.RouterConfig) error {
	if err := ensureKubernetesConfigModels(newConfig); err != nil {
		return fmt.Errorf("failed to ensure models for kubernetes config update: %w", err)
	}

	replaceKubernetesRuntimeConfig(newConfig)
	logging.ComponentEvent("router", "kubernetes_config_applied", map[string]interface{}{
		"config_source":  newConfig.ConfigSource,
		"decision_count": len(newConfig.Decisions),
	})
	return nil
}

// startKubernetesController starts the Kubernetes controller for watching CRDs
func startKubernetesController(staticConfig *config.RouterConfig, kubeconfig, namespace string) {
	// Import k8s package here to avoid import errors when k8s dependencies are not available
	// This is a lazy import pattern
	logging.ComponentEvent("router", "kubernetes_controller_starting", map[string]interface{}{
		"namespace":      namespace,
		"has_kubeconfig": kubeconfig != "",
	})

	controller, err := k8s.NewController(k8s.ControllerConfig{
		Namespace:      namespace,
		Kubeconfig:     kubeconfig,
		StaticConfig:   staticConfig,
		OnConfigUpdate: applyKubernetesConfigUpdate,
	})
	if err != nil {
		logging.ComponentFatalEvent("router", "kubernetes_controller_create_failed", map[string]interface{}{
			"namespace": namespace,
			"error":     err.Error(),
		})
	}

	ctx := context.Background()
	if err := controller.Start(ctx); err != nil {
		logging.ComponentFatalEvent("router", "kubernetes_controller_failed", map[string]interface{}{
			"namespace": namespace,
			"error":     err.Error(),
		})
	}
}

// logStartupSummary emits a single structured log line summarizing the router
// startup state — making it trivial for agents and log aggregators to determine
// what the router is serving and on which ports.
func logStartupSummary(cfg *config.RouterConfig, opts runtimeOptions, embeddingModelsReady bool) {
	decisionNames := make([]string, 0, len(cfg.Decisions))
	for _, d := range cfg.Decisions {
		decisionNames = append(decisionNames, d.Name)
	}

	logging.ComponentEvent("router", "startup_complete", map[string]interface{}{
		"extproc_port":        opts.port,
		"api_port":            opts.apiPort,
		"metrics_port":        opts.metricsPort,
		"secure":              opts.secure,
		"config_source":       cfg.ConfigSource,
		"decisions":           strings.Join(decisionNames, ","),
		"embedding_ready":     embeddingModelsReady,
		"sem_cache_enabled":   cfg.Enabled,
		"model_selection":     cfg.ModelSelection.Enabled,
		"authz_providers":     len(cfg.Authz.Providers),
		"ratelimit_providers": len(cfg.RateLimit.Providers),
	})
}
