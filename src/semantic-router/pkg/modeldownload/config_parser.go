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
	return fieldName == "ModelID" ||
		fieldName == "Qwen3ModelPath" ||
		strings.HasSuffix(fieldName, "ModelPath")
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

// embeddingModelWeightFiles are the files the model runtime's native engine loads to bring
// a semantic embedding model up. They are deliberately stricter than the nested-weight
// heuristic in IsModelComplete: a directory holding only config.json + onnx/ (the layout
// shipped in the image) otherwise satisfies that heuristic via the nested *.onnx files and
// the safetensors/tokenizer download is never triggered, leaving embedding_ready=false (#2172).
var embeddingModelWeightFiles = []string{"model.safetensors", "tokenizer.json"}

// embeddingModelRequiredFiles returns, per canonical embedding model path, the files the
// runtime hard-loads at startup. The qwen3 and multimodal paths share the
// non-healing completeness defect fixed for mmbert in #2195 (#2531).
func embeddingModelRequiredFiles(cfg *config.RouterConfig) map[string][]string {
	required := make(map[string][]string)
	add := func(path string, files []string) {
		path = config.ResolveModelPath(path)
		for _, fileName := range files {
			if !slices.Contains(required[path], fileName) {
				required[path] = append(required[path], fileName)
			}
		}
	}

	add(cfg.MmBertModelPath, embeddingModelWeightFiles)
	add(cfg.Qwen3ModelPath, embeddingModelWeightFiles)
	add(cfg.MultiModalModelPath, embeddingModelWeightFiles)
	return required
}

// onnxWeightExcludePatterns match the ONNX inference exports published beside the
// safetensors weights in the embedding model repositories. The runtime's native engine
// never opens them, yet they dominate the snapshot size (about 4.3 GB of the 4.9 GB
// mmbert-embed-32k-2d-matryoshka repository), so downloads skip them. Small manifests
// such as onnx/model_config.json are not matched and stay in the snapshot.
var onnxWeightExcludePatterns = []string{
	"*.onnx",
	"*.onnx.data",
	"*.onnx_data",
	"onnx/weights.data",
}

// embeddingModelExcludePatterns returns, per configured embedding model path, the
// download exclude globs for artifacts the runtime never loads. The remote backend
// provisions no local embedding models.
//
// Keys are canonical registry paths (config.ResolveModelPath), matching how the
// embedding runtime resolves the same fields before loading. Callers look the map
// up by the resolved path too, so the narrowing holds whether the configured value
// is the canonical directory or a registry alias, and whether or not the collected
// provisioning paths have already been canonicalized upstream.
func embeddingModelExcludePatterns(cfg *config.RouterConfig) map[string][]string {
	excluded := make(map[string][]string)
	if cfg.EmbeddingModels.EmbeddingBackend() != config.EmbeddingBackendModelRuntime {
		return excluded
	}

	for path := range embeddingModelRequiredFiles(cfg) {
		resolved := config.ResolveModelPath(path)
		if resolved == "" || !strings.HasPrefix(resolved, "models/") {
			continue
		}
		excluded[resolved] = append([]string(nil), onnxWeightExcludePatterns...)
	}
	return excluded
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
