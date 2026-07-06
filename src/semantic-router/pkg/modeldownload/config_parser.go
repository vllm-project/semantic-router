package modeldownload

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"slices"
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
			recordModelPath(t.Field(i).Name, field, paths, seen)
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

func recordModelPath(fieldName string, field reflect.Value, paths *[]string, seen map[string]bool) {
	if !isModelPathField(fieldName) || field.Kind() != reflect.String {
		return
	}

	path := field.String()
	if path == "" || !strings.HasPrefix(path, "models/") || seen[path] {
		return
	}

	if !isModelDirectory(path) {
		return
	}

	*paths = append(*paths, path)
	seen[path] = true
}

func isModelPathField(fieldName string) bool {
	return fieldName == "ModelID" ||
		fieldName == "Qwen3ModelPath" ||
		fieldName == "GemmaModelPath" ||
		strings.HasSuffix(fieldName, "ModelPath")
}

// isModelDirectory checks if a path looks like a model directory (not a file)
func isModelDirectory(path string) bool {
	// If the basename has a file extension, treat it as a file rather than a model directory.
	if filepath.Ext(filepath.Base(path)) != "" {
		return false
	}
	return true
}

// BuildModelSpecs builds ModelSpec list from config and registry
func BuildModelSpecs(cfg *config.RouterConfig) ([]ModelSpec, error) {
	// Extract all model paths from config
	paths := filterDisabledOptionalModelPaths(cfg, ExtractModelPaths(cfg))
	requiredFilesByModel := ExtractRequiredFilesByModel(cfg)
	addEmbeddingModelRequiredFiles(cfg, requiredFilesByModel)

	// Allow empty paths for API-only configurations
	if len(paths) == 0 {
		return []ModelSpec{}, nil
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

		requiredFiles := append([]string{}, DefaultRequiredFiles...)
		for _, extra := range requiredFilesByModel[path] {
			if extra != "" && !slices.Contains(requiredFiles, extra) {
				requiredFiles = append(requiredFiles, extra)
			}
		}

		specs = append(specs, ModelSpec{
			LocalPath:     path,
			RepoID:        repoID,
			Revision:      "main",
			RequiredFiles: requiredFiles,
		})
	}

	return specs, nil
}

// embeddingModelWeightFiles are the files the candle embedding runtime loads to bring a
// semantic embedding model up. They are deliberately stricter than the nested-weight
// heuristic in IsModelComplete: a directory holding only config.json + onnx/ (the layout
// shipped in the image) otherwise satisfies that heuristic via the nested *.onnx files and
// the safetensors/tokenizer download is never triggered, leaving embedding_ready=false (#2172).
var embeddingModelWeightFiles = []string{"model.safetensors", "tokenizer.json"}

// addEmbeddingModelRequiredFiles marks the configured semantic embedding model as requiring
// its safetensors weights and tokenizer so a partial (ONNX-only) directory is detected as
// incomplete and the full snapshot is re-downloaded.
func addEmbeddingModelRequiredFiles(cfg *config.RouterConfig, requiredFilesByModel map[string][]string) {
	if cfg.EmbeddingModels.UsesRemoteEmbeddingBackend() {
		return
	}

	// MmBertModelPath holds the configured semantic embedding model directory.
	path := cfg.MmBertModelPath
	if path == "" || !strings.HasPrefix(path, "models/") {
		return
	}

	existing := requiredFilesByModel[path]
	for _, fileName := range embeddingModelWeightFiles {
		if !slices.Contains(existing, fileName) {
			existing = append(existing, fileName)
		}
	}
	requiredFilesByModel[path] = existing
}

// ExtractRequiredFilesByModel derives per-model completeness requirements from
// config-owned companion files such as category/jailbreak/PII mappings.
func ExtractRequiredFilesByModel(cfg *config.RouterConfig) map[string][]string {
	requiredFilesByModel := make(map[string][]string)
	collectRequiredFilesByModel(reflect.ValueOf(cfg), requiredFilesByModel)
	return requiredFilesByModel
}

func collectRequiredFilesByModel(v reflect.Value, requiredFilesByModel map[string][]string) {
	if !v.IsValid() {
		return
	}

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

			if strings.HasSuffix(fieldName, "MappingPath") && field.Kind() == reflect.String {
				recordRequiredMappingFile(requiredFilesByModel, field.String())
			}

			collectRequiredFilesByModel(field, requiredFilesByModel)
		}
	case reflect.Slice, reflect.Array:
		for i := 0; i < v.Len(); i++ {
			collectRequiredFilesByModel(v.Index(i), requiredFilesByModel)
		}
	case reflect.Map:
		for _, key := range v.MapKeys() {
			collectRequiredFilesByModel(v.MapIndex(key), requiredFilesByModel)
		}
	}
}

func recordRequiredMappingFile(requiredFilesByModel map[string][]string, mappingPath string) {
	if !strings.HasPrefix(mappingPath, "models/") {
		return
	}

	modelPath := filepath.Dir(mappingPath)
	fileName := filepath.Base(mappingPath)
	if modelPath == "." || modelPath == "models" || fileName == "" || fileName == "." {
		return
	}

	requiredFiles := requiredFilesByModel[modelPath]
	if !slices.Contains(requiredFiles, fileName) {
		requiredFilesByModel[modelPath] = append(requiredFiles, fileName)
	}
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
