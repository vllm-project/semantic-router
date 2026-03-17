package main

import (
	"context"
	"fmt"

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
			logging.Warnf("Failed to persist model download progress: %v", err)
		}
	}

	return modeldownload.EnsureModelsForConfigWithProgress(cfg, reporter)
}

func applyKubernetesConfigUpdate(newConfig *config.RouterConfig) error {
	if err := ensureKubernetesConfigModels(newConfig); err != nil {
		return fmt.Errorf("failed to ensure models for kubernetes config update: %w", err)
	}

	replaceKubernetesRuntimeConfig(newConfig)
	logging.Infof("Configuration updated from Kubernetes CRDs")
	return nil
}

// startKubernetesController starts the Kubernetes controller for watching CRDs
func startKubernetesController(staticConfig *config.RouterConfig, kubeconfig, namespace string) {
	// Import k8s package here to avoid import errors when k8s dependencies are not available
	// This is a lazy import pattern
	logging.Infof("Initializing Kubernetes controller for namespace: %s", namespace)

	logging.Infof("Starting Kubernetes controller for namespace: %s", namespace)

	controller, err := k8s.NewController(k8s.ControllerConfig{
		Namespace:      namespace,
		Kubeconfig:     kubeconfig,
		StaticConfig:   staticConfig,
		OnConfigUpdate: applyKubernetesConfigUpdate,
	})
	if err != nil {
		logging.Fatalf("Failed to create Kubernetes controller: %v", err)
	}

	ctx := context.Background()
	if err := controller.Start(ctx); err != nil {
		logging.Fatalf("Kubernetes controller error: %v", err)
	}
}
