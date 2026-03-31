package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/k8s"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/logo"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modeldownload"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
)

func main() {
	logo.PrintVLLMLogo()
	opts := parseRuntimeOptions()
	initializeRuntimeLogger()

	cfg := loadRuntimeConfigOrFatal(opts.configPath)
	config.Replace(cfg)

	startupWriter := newStartupWriter(opts.configPath)
	ensureModelsDownloadedOrFatal(cfg, startupWriter)
	exitIfDownloadOnly(opts.downloadOnly)

	defer initializeTracing(cfg)()
	initializeWindowedMetricsIfEnabled(cfg)

	shutdownHooks := make([]func(), 0)
	registerSignalHandler(&shutdownHooks)
	startMetricsServerIfEnabled(cfg, opts.metricsPort)

	embeddingModelsInitialized := initializeRuntimeDependencies(cfg, startupWriter, &shutdownHooks)
	server := newExtProcServerOrFatal(opts, startupWriter)

	loadToolsDatabaseIfReady(server, embeddingModelsInitialized)
	startAPIServerIfEnabled(opts)
	markRouterReady(startupWriter)
	logStartupSummary(cfg, opts, embeddingModelsInitialized)
	startKubernetesControllerIfNeeded(cfg, opts.kubeconfig, opts.namespace)
	startExtProcServerOrFatal(server, startupWriter)
}

var (
	ensureKubernetesConfigModels   = modeldownload.EnsureModelsForConfig
	replaceKubernetesRuntimeConfig = config.Replace
)

func ensureModelsDownloaded(cfg *config.RouterConfig, startupWriter *startupstatus.Writer) error {
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
