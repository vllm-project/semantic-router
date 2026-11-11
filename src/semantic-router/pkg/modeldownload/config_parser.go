package modeldownload

import (
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// ExtractModelPaths extracts canonical local model paths from the configuration.
// It recursively searches for fields named "ModelID", "Qwen3ModelPath",
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
	if v.Kind() == reflect.Pointer {
		if v.IsNil() {
			return
		}
		v = v.Elem()
	}

	switch v.Kind() {
	case reflect.Struct:
		t := v.Type()
		for i := 0; i < v.NumField(); i++ {
			// Opaque compiled registries contain remote provider model IDs,
			// which are not router-owned artifacts even when an alias matches.
			if !t.Field(i).IsExported() {
				continue
			}
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

	// Registry aliases are local artifacts too. Resolve before filtering and
	// deduplication so bare aliases use the same provisioning contract as paths.
	path := config.ResolveModelPath(field.String())
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
	return fieldName == "ModelID" || strings.HasSuffix(fieldName, "ModelPath")
}

// isModelDirectory checks if a path looks like a model directory (not a file)
func isModelDirectory(path string) bool {
	// Versioned model directories can contain dots (for example Vela-1.0).
	// Only known artifact extensions identify a file; the model registry owns
	// whether a directory is actually provisionable.
	ext := strings.ToLower(filepath.Ext(filepath.Base(path)))
	return !slices.Contains([]string{
		".json", ".yaml", ".yml", ".txt", ".bin", ".pt", ".pth",
		".safetensors", ".onnx", ".data", ".xml", ".model", ".gguf",
	}, ext)
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

	if v.Kind() == reflect.Pointer {
		if v.IsNil() {
			return
		}
		v = v.Elem()
	}

	switch v.Kind() {
	case reflect.Struct:
		t := v.Type()
		for i := 0; i < v.NumField(); i++ {
			fieldType := t.Field(i)
			if !fieldType.IsExported() {
				continue
			}
			field := v.Field(i)
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

	modelPath := config.ResolveModelPath(filepath.Dir(mappingPath))
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
