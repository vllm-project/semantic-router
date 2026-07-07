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
// config-owned companion files such as category/jailbreak/PII mappings, plus
// the files the configured embedding runtime hard-loads at startup.
func ExtractRequiredFilesByModel(cfg *config.RouterConfig) map[string][]string {
	requiredFilesByModel := make(map[string][]string)
	collectRequiredFilesByModel(reflect.ValueOf(cfg), requiredFilesByModel)
	collectCandleEmbeddingRequiredFiles(cfg, requiredFilesByModel)
	return requiredFilesByModel
}

// candleEmbeddingRequiredFiles are the files the candle embedding runtime
// hard-loads from each configured embedding model directory at startup:
// candle-binding builds its VarBuilder from <model>/model.safetensors and the
// tokenizer from <model>/tokenizer.json, with no fallback formats.
var candleEmbeddingRequiredFiles = []string{"model.safetensors", "tokenizer.json"}

// gemmaEmbeddingRequiredFiles extends the common candle set with the dense
// bottleneck weights GemmaEmbeddingModel additionally hard-loads from the
// 2_Dense/ and 3_Dense/ subdirectories (BottleneckDenseNet::load_from_path).
var gemmaEmbeddingRequiredFiles = []string{
	"model.safetensors",
	"tokenizer.json",
	"2_Dense/model.safetensors",
	"3_Dense/model.safetensors",
}

// collectCandleEmbeddingRequiredFiles records the candle runtime's startup
// files as per-model completeness requirements for a superset of the embedding
// models the candle runtime may load: the qwen3/gemma/mmbert paths under the
// candle backend, and the multimodal path whenever model_type selects it or
// complexity rules declare image_candidates (which loads the multimodal model
// regardless of model_type).
//
// Without this, an interrupted download that fetched only companion artifacts
// (config.json, nested onnx/ exports) satisfies the generic weight heuristic
// in IsModelComplete: the model is treated as complete, is never re-downloaded
// on restart, and the router crash-loops at classifier init with
// "failed to load safetensors from <model>/model.safetensors". Recording the
// runtime-required files makes such partial downloads read as incomplete so
// the next startup resumes the download instead of crash-looping.
//
// The embedding path fields are enumerated explicitly (unlike the reflective
// ExtractModelPaths walk) because the requirement is loader-specific: only the
// paths whose loaders hard-code these files are covered. A new embedding path
// field needs a matching entry here to gain partial-download protection.
func collectCandleEmbeddingRequiredFiles(cfg *config.RouterConfig, requiredFilesByModel map[string][]string) {
	// Use the canonical backend resolution (EmbeddingModels.EmbeddingBackend):
	// only the candle backend hard-loads these local files at startup. The
	// remote backend (openai_compatible, including model_type=remote with no
	// backend set) and openvino load nothing from these paths, so requiring
	// the files there would force downloads of models that never load.
	if cfg.EmbeddingModels.EmbeddingBackend() == config.EmbeddingBackendCandle {
		appendRequiredFiles(requiredFilesByModel, cfg.EmbeddingModels.Qwen3ModelPath, candleEmbeddingRequiredFiles)
		appendRequiredFiles(requiredFilesByModel, cfg.EmbeddingModels.GemmaModelPath, gemmaEmbeddingRequiredFiles)
		appendRequiredFiles(requiredFilesByModel, cfg.EmbeddingModels.MmBertModelPath, candleEmbeddingRequiredFiles)
	}

	// The multimodal model is loaded independently of the backend value:
	// classification's initializer branches run it whenever model_type
	// resolves to "multimodal", and the complexity classifier additionally
	// loads it whenever any complexity rule declares image_candidates,
	// without any model_type gate. Keep these triggers in sync with
	// classifier_option_rules.go.
	modelType := strings.ToLower(strings.TrimSpace(cfg.EmbeddingModels.EmbeddingConfig.ModelType))
	if modelType == "multimodal" || config.HasImageCandidatesInRules(cfg.ComplexityRules) {
		appendRequiredFiles(requiredFilesByModel, cfg.EmbeddingModels.MultiModalModelPath, candleEmbeddingRequiredFiles)
	}
}

// appendRequiredFiles records files as completeness requirements for a local
// model path, deduplicating against already-recorded requirements.
func appendRequiredFiles(requiredFilesByModel map[string][]string, modelPath string, files []string) {
	if !strings.HasPrefix(modelPath, "models/") || modelPath == "models/" {
		return
	}
	requiredFiles := requiredFilesByModel[modelPath]
	for _, file := range files {
		if !slices.Contains(requiredFiles, file) {
			requiredFiles = append(requiredFiles, file)
		}
	}
	requiredFilesByModel[modelPath] = requiredFiles
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
