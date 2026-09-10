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
	var (
		routerServer     *extproc.Server
		metricsServer    *http.Server
		servingLifecycle *servingComponentLifecycle
		shutdownHooks    = make([]func(context.Context) error, 0)
		shutdownTracing  = func(context.Context) error { return nil }
	)
	// Return errors below so deferred shutdown can release started resources.
	defer func() {
		shutdownCtx, cancel := context.WithTimeout(context.Background(), processShutdownTimeout)
		defer cancel()
		runErr = errors.Join(runErr, shutdownRouterProcess(
			shutdownCtx,
			apiServer,
			routerServer,
			metricsServer,
			servingLifecycle,
			&shutdownHooks,
			shutdownTracing,
		))
	}()

	if err = ensureModelsDownloaded(ctx, cfg, startupWriter); err != nil {
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

	if err = warmupRouterRuntime(ctx, routerServer, embeddingRuntime); err != nil {
		return recordStartupError(startupWriter, "warm up router runtime", err)
	}
	markRouterReady(startupWriter, startupEmbeddingProviderStatus(embeddingRuntime))
	logStartupSummary(cfg, opts, embeddingRuntime.AnyReady)
	servingLifecycle, err = runRouterServing(ctx, cfg, opts, routerServer, startupWriter)
	return err
}

func shutdownRouterProcess(
	ctx context.Context,
	apiServer *apiserver.Server,
	routerServer *extproc.Server,
	metricsServer *http.Server,
	servingLifecycle *servingComponentLifecycle,
	shutdownHooks *[]func(context.Context) error,
	shutdownTracing func(context.Context) error,
) error {
	var managementShutdown func(context.Context) error
	if apiServer != nil {
		managementShutdown = apiServer.Shutdown
	}
	var resourceShutdown func(context.Context) error
	servingShutdowns := make([]func(context.Context) error, 0, 2)
	if routerServer != nil {
		servingShutdowns = append(servingShutdowns, routerServer.ShutdownServing)
		resourceShutdown = routerServer.ShutdownResources
	}
	if servingLifecycle != nil {
		routerResourceShutdown := resourceShutdown
		resourceShutdown = func(ctx context.Context) error {
			if err := servingLifecycle.Shutdown(ctx); err != nil {
				return err
			}
			if routerResourceShutdown != nil {
				return routerResourceShutdown(ctx)
			}
			return nil
		}
	}
	if metricsServer != nil {
		servingShutdowns = append(servingShutdowns, metricsServer.Shutdown)
	}
	return shutdownRouterComponents(
		ctx,
		managementShutdown,
		resourceShutdown,
		shutdownHooks,
		shutdownTracing,
		servingShutdowns...,
	)
}

func shutdownRouterComponents(
	ctx context.Context,
	managementShutdown func(context.Context) error,
	resourceShutdown func(context.Context) error,
	shutdownHooks *[]func(context.Context) error,
	shutdownTracing func(context.Context) error,
	servingShutdowns ...func(context.Context) error,
) error {
	managementDone := make(chan error, 1)
	go func() {
		if managementShutdown == nil {
			managementDone <- nil
			return
		}
		managementDone <- managementShutdown(ctx)
	}()

	servingErr := shutdownConcurrently(ctx, servingShutdowns...)
	managementErr := <-managementDone
	shutdownErr := errors.Join(managementErr, servingErr)
	if errors.Is(managementErr, context.Canceled) || errors.Is(managementErr, context.DeadlineExceeded) {
		return shutdownErr
	}
	if resourceShutdown != nil {
		shutdownErr = errors.Join(shutdownErr, resourceShutdown(ctx))
	}
	shutdownErr = errors.Join(shutdownErr, runShutdownHooks(ctx, shutdownHooks))
	shutdownErr = errors.Join(shutdownErr, shutdownTracing(ctx))
	return shutdownErr
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

func shutdownConcurrently(ctx context.Context, shutdowns ...func(context.Context) error) error {
	shutdownResults := make(chan error, len(shutdowns))
	for _, shutdown := range shutdowns {
		go func(shutdown func(context.Context) error) {
			shutdownResults <- shutdown(ctx)
		}(shutdown)
	}
	shutdownErrors := make([]error, 0, len(shutdowns))
	for range shutdowns {
		shutdownErrors = append(shutdownErrors, <-shutdownResults)
	}
	return errors.Join(shutdownErrors...)
}

var (
	ensureKubernetesConfigModels = func(ctx context.Context, cfg *config.RouterConfig) error {
		return modeldownload.EnsureModelsForConfigWithProgressContext(ctx, cfg, nil)
	}
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

func applyKubernetesConfigUpdate(ctx context.Context, newConfig *config.RouterConfig) error {
	if err := ensureKubernetesConfigModels(ctx, newConfig); err != nil {
		return fmt.Errorf("failed to ensure models for kubernetes config update: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	replaceKubernetesRuntimeConfig(newConfig)
	logging.ComponentEvent("router", "kubernetes_config_applied", map[string]interface{}{
		"config_source":  newConfig.ConfigSource,
		"decision_count": len(newConfig.Decisions),
	})
	return nil
}

func runRouterServing(
	ctx context.Context,
	cfg *config.RouterConfig,
	opts runtimeOptions,
	routerServer *extproc.Server,
	startupWriter startupstatus.StatusWriter,
) (*servingComponentLifecycle, error) {
	components := []func(context.Context) error{
		func(ctx context.Context) error {
			return startExtProcServer(ctx, routerServer, startupWriter)
		},
	}
	if cfg.ConfigSource == config.ConfigSourceKubernetes {
		components = append(components, func(ctx context.Context) error {
			return startKubernetesController(ctx, cfg, opts.kubeconfig, opts.namespace)
		})
	}
	lifecycle := startServingComponents(ctx, components...)
	return lifecycle, lifecycle.Wait(ctx)
}

type servingComponentLifecycle struct {
	cancel context.CancelFunc
	first  <-chan error
	done   <-chan struct{}
}

func startServingComponents(ctx context.Context, components ...func(context.Context) error) *servingComponentLifecycle {
	runCtx, cancel := context.WithCancel(ctx)
	results := make(chan error, len(components))
	for _, component := range components {
		go func(component func(context.Context) error) {
			results <- component(runCtx)
		}(component)
	}
	first := make(chan error, 1)
	done := make(chan struct{})
	if len(components) == 0 {
		first <- nil
		close(done)
		return &servingComponentLifecycle{cancel: cancel, first: first, done: done}
	}
	go func() {
		first <- <-results
		cancel()
		for range len(components) - 1 {
			<-results
		}
		close(done)
	}()
	return &servingComponentLifecycle{cancel: cancel, first: first, done: done}
}

func (l *servingComponentLifecycle) Wait(ctx context.Context) error {
	select {
	case <-ctx.Done():
		l.cancel()
		return nil
	case err := <-l.first:
		return err
	}
}

func (l *servingComponentLifecycle) Shutdown(ctx context.Context) error {
	if l == nil {
		return nil
	}
	l.cancel()
	select {
	case <-l.done:
		return nil
	case <-ctx.Done():
		select {
		case <-l.done:
			return nil
		default:
			return ctx.Err()
		}
	}
}

// startKubernetesController runs the Kubernetes controller until ctx is canceled or it fails.
func startKubernetesController(
	ctx context.Context,
	staticConfig *config.RouterConfig,
	kubeconfig,
	namespace string,
) error {
	logging.ComponentEvent("router", "kubernetes_controller_starting", map[string]interface{}{
		"namespace":      namespace,
		"has_kubeconfig": kubeconfig != "",
	})

	controller, err := k8s.NewController(k8s.ControllerConfig{
		Namespace:    namespace,
		Kubeconfig:   kubeconfig,
		StaticConfig: staticConfig,
		OnConfigUpdate: func(newConfig *config.RouterConfig) error {
			return applyKubernetesConfigUpdate(ctx, newConfig)
		},
	})
	if err != nil {
		return fmt.Errorf("create Kubernetes controller: %w", err)
	}

	if err := controller.Start(ctx); err != nil {
		return fmt.Errorf("serve Kubernetes controller: %w", err)
	}
	return nil
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
