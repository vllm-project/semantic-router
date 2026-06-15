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

	// Only require huggingface-cli when there are actually missing models to
	// download. If all configured models are already present locally (e.g.
	// pre-mounted into the container), skip the CLI check so the router can
	// start in environments without huggingface-cli installed.
	missing, missingErr := GetMissingModels(specs)
	if missingErr != nil {
		return fmt.Errorf("failed to check models: %w", missingErr)
	}
	if len(missing) == 0 {
		logging.ComponentEvent("router", "required_models_already_present", map[string]interface{}{
			"total_models": len(specs),
		})
		reportProgress(reporter, ProgressState{
			Phase:       "completed",
			ReadyModels: len(specs),
			TotalModels: len(specs),
			Message:     "All required router models are already present locally.",
		})
		return nil
	}

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
