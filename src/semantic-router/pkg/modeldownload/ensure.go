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
	logging.ComponentEvent("router", "required_models_check_started", map[string]interface{}{})

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
		logging.ComponentEvent("router", "required_models_check_skipped", map[string]interface{}{
			"reason": "no_local_models_configured",
			"mode":   "api_only",
		})
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
	logging.ComponentDebugEvent("router", "huggingface_download_config_loaded", map[string]interface{}{
		"hf_endpoint": downloadConfig.HFEndpoint,
		"hf_token":    maskedToken,
		"hf_home":     downloadConfig.HFHome,
	})

	if err := EnsureModelsWithProgress(specs, downloadConfig, reporter); err != nil {
		return fmt.Errorf("failed to download models: %w", err)
	}

	logging.ComponentEvent("router", "required_models_ready", map[string]interface{}{})
	return nil
}

func logModelRegistry(registry map[string]string) {
	uniqueModels := make(map[string]bool)
	for _, repoID := range registry {
		uniqueModels[repoID] = true
	}

	logging.ComponentDebugEvent("router", "model_registry_summary", map[string]interface{}{
		"unique_models":    len(uniqueModels),
		"registry_aliases": len(registry),
	})
}
