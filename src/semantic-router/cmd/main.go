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

// ensureModelsDownloaded checks and downloads required models
func ensureModelsDownloaded(cfg *config.RouterConfig, startupWriter *startupstatus.Writer) error {
	logging.Infof("Installing required models...")

	// Build model specs from config
	specs, err := modeldownload.BuildModelSpecs(cfg)
	if err != nil {
		return fmt.Errorf("failed to build model specs: %w", err)
	}

	// Skip download if no local models are configured (API-only mode)
	if len(specs) == 0 {
		_ = startupWriter.Write(startupstatus.State{
			Phase:   "initializing_models",
			Ready:   false,
			Message: "No local models configured. Skipping model download.",
		})
		logging.Infof("No local models configured, skipping model download (API-only mode)")
		return nil
	}

	// Calculate unique models based on RepoID
	uniqueModels := make(map[string]bool)
	for _, repoID := range cfg.MoMRegistry {
		uniqueModels[repoID] = true
	}

	// Print model registry configuration
	logging.Infof("MoM Families: %d unique models (total %d registry aliases)", len(uniqueModels), len(cfg.MoMRegistry))
	logging.Debugf("Registry Details:")
	for localPath, repoID := range cfg.MoMRegistry {
		logging.Debugf("  %s -> %s", localPath, repoID)
	}

	// Check if huggingface-cli is available
	if err := modeldownload.CheckHuggingFaceCLI(); err != nil {
		return fmt.Errorf("huggingface-cli check failed: %w", err)
	}

	// Get download configuration from environment
	downloadConfig := modeldownload.GetDownloadConfig()

	// Log environment configuration (mask sensitive token)
	maskedToken := "***"
	if downloadConfig.HFToken == "" {
		maskedToken = "<not set>"
	}
	logging.Infof("HF_ENDPOINT: %s; HF_TOKEN: %s; HF_HOME: %s", downloadConfig.HFEndpoint, maskedToken, downloadConfig.HFHome)

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
		default:
			state.Phase = "checking_models"
		}

		if err := startupWriter.Write(state); err != nil {
			logging.Warnf("Failed to persist model download progress: %v", err)
		}
	}

	// Ensure all models are downloaded
	if err := modeldownload.EnsureModelsWithProgress(specs, downloadConfig, reporter); err != nil {
		return fmt.Errorf("failed to download models: %w", err)
	}

	logging.Infof("All required models are ready")
	return nil
}

// startKubernetesController starts the Kubernetes controller for watching CRDs
func startKubernetesController(staticConfig *config.RouterConfig, kubeconfig, namespace string) {
	// Import k8s package here to avoid import errors when k8s dependencies are not available
	// This is a lazy import pattern
	logging.Infof("Initializing Kubernetes controller for namespace: %s", namespace)

	logging.Infof("Starting Kubernetes controller for namespace: %s", namespace)

	controller, err := k8s.NewController(k8s.ControllerConfig{
		Namespace:    namespace,
		Kubeconfig:   kubeconfig,
		StaticConfig: staticConfig,
		OnConfigUpdate: func(newConfig *config.RouterConfig) error {
			config.Replace(newConfig)
			logging.Infof("Configuration updated from Kubernetes CRDs")
			return nil
		},
	})
	if err != nil {
		logging.Fatalf("Failed to create Kubernetes controller: %v", err)
	}

	ctx := context.Background()
	if err := controller.Start(ctx); err != nil {
		logging.Fatalf("Kubernetes controller error: %v", err)
	}
}
