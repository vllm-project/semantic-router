package modeldownload

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modellifecycle"
)

// ExtractModelPaths returns the router-owned local model directories configured
// by the shared model lifecycle plan.
func ExtractModelPaths(cfg *config.RouterConfig) []string {
	return modellifecycle.BuildPlan(cfg).ConfiguredModelPaths()
}

// isModelDirectory checks if a path looks like a model directory (not a file).
func isModelDirectory(path string) bool {
	return filepath.Ext(filepath.Base(path)) == ""
}

// BuildModelSpecs builds ModelSpec list from config and registry.
func BuildModelSpecs(cfg *config.RouterConfig) ([]ModelSpec, error) {
	plan := modellifecycle.BuildPlan(cfg)
	assets := plan.DownloadAssets()

	if len(assets) == 0 {
		return []ModelSpec{}, nil
	}

	registry := cfg.MoMRegistry
	if len(registry) == 0 {
		return nil, fmt.Errorf("mom_registry is empty in configuration")
	}

	var specs []ModelSpec
	for _, asset := range assets {
		repoID, ok := registry[asset.LocalPath]
		if !ok {
			return nil, fmt.Errorf("model path %s not found in mom_registry", asset.LocalPath)
		}

		requiredFiles := append([]string{}, DefaultRequiredFiles...)
		for _, extra := range asset.RequiredFiles {
			if extra != "" && !slices.Contains(requiredFiles, extra) {
				requiredFiles = append(requiredFiles, extra)
			}
		}

		specs = append(specs, ModelSpec{
			LocalPath:     asset.LocalPath,
			RepoID:        repoID,
			Revision:      "main",
			RequiredFiles: requiredFiles,
		})
	}

	return specs, nil
}

// ExtractRequiredFilesByModel derives per-model completeness requirements from
// config-owned companion files such as category/jailbreak/PII mappings.
func ExtractRequiredFilesByModel(cfg *config.RouterConfig) map[string][]string {
	return modellifecycle.BuildPlan(cfg).RequiredFilesByModel()
}

// GetDownloadConfig creates DownloadConfig from environment variables.
func GetDownloadConfig() DownloadConfig {
	return DownloadConfig{
		HFEndpoint: getEnvOrDefault("HF_ENDPOINT", "https://huggingface.co"),
		HFToken:    os.Getenv("HF_TOKEN"),
		HFHome:     getEnvOrDefault("HF_HOME", ""),
	}
}

func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}
