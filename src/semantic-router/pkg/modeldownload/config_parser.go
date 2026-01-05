package modeldownload

import (
	"fmt"
	"os"
	"reflect"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// ExtractModelPaths extracts all model paths from the configuration
// It recursively searches for fields named "ModelID", "Qwen3ModelPath", "GemmaModelPath",
// or any field ending with "ModelPath" (but excludes non-model paths like mapping_path, tools_db_path)
func ExtractModelPaths(cfg *config.RouterConfig) []string {
	var paths []string
	seen := make(map[string]bool)

	// Use reflection to traverse the config structure
	extractFromValue(reflect.ValueOf(cfg), &paths, seen)

	return paths
}

// extractFromValue recursively extracts model paths from a reflect.Value
func extractFromValue(v reflect.Value, paths *[]string, seen map[string]bool) {
	if !v.IsValid() {
		return
	}

	// Dereference pointers
	if v.Kind() == reflect.Ptr {
		if v.IsNil() {
			return
		}
		v = v.Elem()
	}

	switch v.Kind() {
	case reflect.Struct:
		t := v.Type()
		for i := 0; i < v.NumField(); i++ {
			field := v.Field(i)
			fieldType := t.Field(i)
			fieldName := fieldType.Name

			// Check if this is a model path field
			// Matches: ModelID, Qwen3ModelPath, GemmaModelPath, or any field ending with "ModelPath"
			isModelField := fieldName == "ModelID" ||
				fieldName == "Qwen3ModelPath" ||
				fieldName == "GemmaModelPath" ||
				strings.HasSuffix(fieldName, "ModelPath")

			if isModelField && field.Kind() == reflect.String {
				path := field.String()
				if path != "" && strings.HasPrefix(path, "models/") && !seen[path] {
					// Only add if it looks like a model directory (not a file path)
					if !strings.Contains(path[7:], "/") || isModelDirectory(path) {
						*paths = append(*paths, path)
						seen[path] = true
					}
				}
			}

			// Recursively process nested structs
			extractFromValue(field, paths, seen)
		}

	case reflect.Slice, reflect.Array:
		for i := 0; i < v.Len(); i++ {
			extractFromValue(v.Index(i), paths, seen)
		}

	case reflect.Map:
		for _, key := range v.MapKeys() {
			extractFromValue(v.MapIndex(key), paths, seen)
		}
	}
}

// isModelDirectory checks if a path looks like a model directory (not a file)
func isModelDirectory(path string) bool {
	// If path ends with a file extension, it's not a model directory
	if strings.HasSuffix(path, ".json") || strings.HasSuffix(path, ".txt") ||
		strings.HasSuffix(path, ".yaml") || strings.HasSuffix(path, ".yml") {
		return false
	}
	return true
}

// BuildModelSpecs builds ModelSpec list from config and registry
func BuildModelSpecs(cfg *config.RouterConfig) ([]ModelSpec, error) {
	// Extract all model paths from config
	paths := ExtractModelPaths(cfg)

	if len(paths) == 0 {
		return nil, fmt.Errorf("no model paths found in configuration")
	}

	// Get model registry from config
	registry := cfg.MoMRegistry
	if len(registry) == 0 {
		return nil, fmt.Errorf("mom_registry is empty in configuration")
	}

	// Build specs
	var specs []ModelSpec
	for _, path := range paths {
		repoID, ok := registry[path]
		if !ok {
			return nil, fmt.Errorf("model path %s not found in mom_registry", path)
		}

		specs = append(specs, ModelSpec{
			LocalPath:     path,
			RepoID:        repoID,
			Revision:      "main",
			RequiredFiles: DefaultRequiredFiles,
		})
	}

	return specs, nil
}

// GetDownloadConfig creates DownloadConfig from environment variables
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
