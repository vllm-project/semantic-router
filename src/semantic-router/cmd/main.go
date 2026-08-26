package main

import (
	"context"
	"os"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/logo"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modeldownload"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/runtimecapabilities"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
)

func main() {
	logo.PrintVLLMLogo()
	opts := parseRuntimeOptions()
	initializeRuntimeLogger()
	applyBackendRuntimeTuningDefaults()

	connectionCompiler, err := productionModelConnectionCompiler()
	if err != nil {
		logging.Fatalf("Failed to compose Provider Integrations: %v", err)
	}
	configParser := config.NewParser(connectionCompiler)
	cfg := loadRuntimeConfigOrFatal(opts.configPath, configParser)
	config.Replace(cfg)
	runtimeRegistry := routerruntime.NewRegistry(cfg)
	capabilities, err := runtimecapabilities.Derive(cfg)
	if err != nil {
		logging.Fatalf("Failed to derive runtime capabilities: %v", err)
	}

	startupWriter := newStartupWriter(cfg, opts.configPath)
	resolvedOpts, err := resolveRuntimeManagementOptions(opts, cfg)
	if err != nil {
		failStartup(startupWriter, "Failed to resolve management API: %v", err)
	}
	opts = resolvedOpts
	if capabilities.ManagementAPI && !opts.enableAPI {
		failStartup(startupWriter, "Management API is enabled but its listener was disabled")
	}
	if opts.downloadOnly {
		ensureModelsDownloadedOrFatal(cfg, startupWriter)
		exitIfDownloadOnly(true)
	}
	processContext, cancelProcess := context.WithCancel(context.Background())
	processRuntime, err := composeProcessRuntime(processContext, cfg)
	if err != nil {
		cancelProcess()
		failStartup(startupWriter, "Failed to compose process runtime: %v", err)
	}

	// Start the API server early so /startup-status is available during
	// model downloads and initialization.
	apiLifecycle, err := startAPIServerIfEnabled(
		processContext, opts, runtimeRegistry, processRuntime, processRuntime.ManagementAPI(),
	)
	if err != nil {
		cancelProcess()
		_ = processRuntime.Close()
		failStartup(startupWriter, "Failed to start management API: %v", err)
	}
	defer func() {
		cancelProcess()
		shutdownContext, cancelShutdown := context.WithTimeout(context.Background(), 12*time.Second)
		defer cancelShutdown()
		if err := apiLifecycle.Close(shutdownContext); err != nil {
			logging.ComponentWarnEvent("router", "api_server_shutdown_failed", map[string]interface{}{
				"error": err.Error(),
			})
		}
		if err := processRuntime.Close(); err != nil {
			logging.ComponentWarnEvent("router", "runtime_shutdown_failed", map[string]interface{}{
				"error": err.Error(),
			})
		}
	}()

	ensureModelsDownloadedOrFatal(cfg, startupWriter)

	defer initializeTracing(cfg)()
	initializeWindowedMetricsIfEnabled(cfg)

	shutdownHooks := make([]func(), 0)
	defer runShutdownHooks(&shutdownHooks)
	startMetricsServerIfEnabled(cfg, opts.metricsPort)

	embeddingRuntime := initializeRuntimeDependencies(cfg, startupWriter, &shutdownHooks, runtimeRegistry)
	if err := processRuntime.Start(processContext); err != nil {
		failStartup(startupWriter, "Failed to start process runtime: %v", err)
	}
	server := newExtProcServerOrFatal(opts, startupWriter, runtimeRegistry, extproc.ServerDependencies{
		DurableRoutingRequests: processRuntime.DurableRoutingRequests(),
		FileRequests:           processRuntime.FileRequests(),
		DispatchCapabilities:   processRuntime.DispatchCapabilities(),
		OutcomeFeedback:        processRuntime.OutcomeFeedback(),
		OutcomeProjection:      processRuntime.OutcomeProjection(),
		ResponseTerminals:      processRuntime.ResponseTerminals(),
		ProtocolCodecs:         processRuntime.ProtocolCodecs(),
		ParseConfig:            configParser.Parse,
	})

	warmupRouterRuntime(server, embeddingRuntime)
	markRouterReady(startupWriter, startupEmbeddingProviderStatus(embeddingRuntime))
	logStartupSummary(cfg, opts, embeddingRuntime.AnyReady)
	startExtProcServerOrFatal(server, startupWriter)
}

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

func ensureModelsDownloaded(cfg *config.RouterConfig, startupWriter startupstatus.StatusWriter) error {
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

	return modeldownload.EnsureModelsForConfigWithProgress(cfg, reporter)
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
		"extproc_port":      opts.port,
		"api_port":          opts.apiPort,
		"metrics_port":      opts.metricsPort,
		"secure":            opts.secure,
		"decisions":         strings.Join(decisionNames, ","),
		"embedding_ready":   embeddingModelsReady,
		"sem_cache_enabled": cfg.Enabled,
		"model_selection":   cfg.ModelSelection.Enabled,
	})
}
