package modeldownload

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// EnsureModelsForConfig ensures all local models required by cfg are ready.
func EnsureModelsForConfig(cfg *config.RouterConfig) error {
	return EnsureModelsForConfigWithProgress(cfg, nil)
}

// EnsureModelsForConfigWithProgress ensures all local models required by cfg
// are ready and optionally reports progress.
func EnsureModelsForConfigWithProgress(
	cfg *config.RouterConfig,
	reporter ProgressReporter,
) error {
	logging.Infof("Installing required models...")

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		return fmt.Errorf("failed to build model specs: %w", err)
	}

	if len(specs) == 0 {
		reportProgress(reporter, ProgressState{
			Phase:       "skipped",
			ReadyModels: 0,
			TotalModels: 0,
			Message:     "No local models configured. Skipping model download.",
		})
		logging.Infof("No local models configured, skipping model download (API-only mode)")
		return nil
	}

	logModelRegistry(cfg.MoMRegistry)

	if err := CheckHuggingFaceCLI(); err != nil {
		return fmt.Errorf("huggingface-cli check failed: %w", err)
	}

	downloadConfig := GetDownloadConfig()
	maskedToken := "***"
	if downloadConfig.HFToken == "" {
		maskedToken = "<not set>"
	}
	logging.Infof(
		"HF_ENDPOINT: %s; HF_TOKEN: %s; HF_HOME: %s",
		downloadConfig.HFEndpoint,
		maskedToken,
		downloadConfig.HFHome,
	)

	if err := EnsureModelsWithProgress(specs, downloadConfig, reporter); err != nil {
		return fmt.Errorf("failed to download models: %w", err)
	}

	logging.Infof("All required models are ready")
	return nil
}

func logModelRegistry(registry map[string]string) {
	uniqueModels := make(map[string]bool)
	for _, repoID := range registry {
		uniqueModels[repoID] = true
	}

	logging.Infof(
		"MoM Families: %d unique models (total %d registry aliases)",
		len(uniqueModels),
		len(registry),
	)
	logging.Debugf("Registry Details:")
	for localPath, repoID := range registry {
		logging.Debugf("  %s -> %s", localPath, repoID)
	}
}
