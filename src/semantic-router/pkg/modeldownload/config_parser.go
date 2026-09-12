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
	// Versioned model directories can contain dots (for example Vela-1.0).
	// Only known artifact extensions identify a file; the model registry owns
	// whether a directory is actually provisionable.
	ext := strings.ToLower(filepath.Ext(filepath.Base(path)))
	return !slices.Contains([]string{
		".json", ".yaml", ".yml", ".txt", ".bin", ".pt", ".pth",
		".safetensors", ".onnx", ".data", ".xml", ".model", ".gguf",
	}, ext)
}

// BuildModelSpecs builds ModelSpec list from config and registry
func BuildModelSpecs(cfg *config.RouterConfig) ([]ModelSpec, error) {
	// Extract shared/default paths plus paths owned by request-reachable named
	// recipes. Declared but unmapped recipes remain validated by config and
	// classifier construction without forcing unused model snapshots onto disk.
	paths := filterDisabledOptionalModelPaths(cfg, extractProvisioningModelPaths(cfg))
	requiredFilesByModel := ExtractRequiredFilesByModel(cfg)
	addEmbeddingModelRequiredFiles(cfg, requiredFilesByModel)
	excludePatternsByModel := runtimeEmbeddingModelExcludePatterns(cfg)

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
			LocalPath:       path,
			RepoID:          repoID,
			Revision:        modelRevision(path, repoID),
			RequiredFiles:   requiredFiles,
			ExcludePatterns: modelDownloadExcludePatterns(path, repoID, excludePatternsByModel[config.ResolveModelPath(path)]),
		})
	}

	return specs, nil
}

func modelDownloadExcludePatterns(path, repoID string, runtimePatterns []string) []string {
	patterns := slices.Clone(runtimePatterns)
	if model := config.GetModelByPath(path); model != nil && model.RepoID == repoID {
		for _, pattern := range model.DownloadExcludePatterns {
			if !slices.Contains(patterns, pattern) {
				patterns = append(patterns, pattern)
			}
		}
	}
	return patterns
}

func modelRevision(path, repoID string) string {
	// A user mapping this path to a different repository owns that repository's
	// revision. Never attach a built-in model's commit to a custom override.
	if model := config.GetModelByPath(path); model != nil && model.RepoID == repoID && model.Revision != "" {
		return model.Revision
	}
	return "main"
}

func extractProvisioningModelPaths(cfg *config.RouterConfig) []string {
	if cfg == nil {
		return nil
	}

	paths := make([]string, 0)
	seen := make(map[string]bool)

	// Canonical configs mirror the default recipe into the flat routing fields.
	// Strip the normalized recipe registry before walking shared/default state so
	// named recipes can be added back according to request reachability.
	sharedAndDefault := *cfg
	sharedAndDefault.Recipes = nil
	sharedAndDefault.Entrypoints = nil
	if !cfg.IsRecipeReachableForRouting(config.DefaultRecipeName) {
		sharedAndDefault.Signals = config.Signals{}
		sharedAndDefault.Projections = config.Projections{}
		sharedAndDefault.Decisions = nil
	}
	extractFromValue(reflect.ValueOf(&sharedAndDefault), &paths, seen)

	for _, recipe := range cfg.ReachableRoutingRecipes() {
		if recipe == nil || recipe.Name == config.DefaultRecipeName {
			continue
		}
		extractFromValue(reflect.ValueOf(recipe.Profile), &paths, seen)
	}
	return paths
}

// embeddingModelWeightFiles are the files the candle embedding runtime loads to bring a
// semantic embedding model up. They are deliberately stricter than the nested-weight
// heuristic in IsModelComplete: a directory holding only config.json + onnx/ (the layout
// shipped in the image) otherwise satisfies that heuristic via the nested *.onnx files and
// the safetensors/tokenizer download is never triggered, leaving embedding_ready=false (#2172).
var embeddingModelWeightFiles = []string{"model.safetensors", "tokenizer.json"}

// gemmaDenseWeightFiles are the dense-bottleneck weights the gemma embedding model
// additionally hard-loads at startup (candle-binding dense_layers: 2_Dense + 3_Dense).
var gemmaDenseWeightFiles = []string{
	"2_Dense/model.safetensors",
	"3_Dense/model.safetensors",
}

// candleEmbeddingModelRequiredFiles returns, per configured candle embedding model path,
// the files the runtime hard-loads at startup. The qwen3, gemma, and multimodal paths
// share the non-healing completeness defect fixed for mmbert in #2195 (#2531).
func candleEmbeddingModelRequiredFiles(cfg *config.RouterConfig) map[string][]string {
	required := make(map[string][]string)
	add := func(path string, files []string) {
		for _, fileName := range files {
			if !slices.Contains(required[path], fileName) {
				required[path] = append(required[path], fileName)
			}
		}
	}

	add(cfg.MmBertModelPath, embeddingModelWeightFiles)
	add(cfg.Qwen3ModelPath, embeddingModelWeightFiles)
	add(cfg.GemmaModelPath, embeddingModelWeightFiles)
	add(cfg.GemmaModelPath, gemmaDenseWeightFiles)
	add(cfg.MultiModalModelPath, embeddingModelWeightFiles)
	return required
}

// onnxWeightExcludePatterns match the ONNX inference exports published beside the
// safetensors weights in the embedding model repositories. The candle runtime never
// opens them, yet they dominate the snapshot size (about 4.3 GB of the 4.9 GB
// mmbert-embed-32k-2d-matryoshka repository), so a candle deployment skips them at
// download time. Small manifests such as onnx/model_config.json, which
// config.MmBertAvailableLayers reads, are not matched and stay in the snapshot.
var onnxWeightExcludePatterns = []string{
	"*.onnx",
	"*.onnx.data",
	"*.onnx_data",
}

// candleEmbeddingModelExcludePatterns returns, per configured embedding model path,
// the download exclude globs for artifacts the selected embedding backend never
// loads. Only the candle backend is narrowed: OpenVINO consumes the ONNX exports
// and the remote backend provisions no local embedding models.
//
// Keys are canonical registry paths (config.ResolveModelPath), matching how the
// embedding runtime resolves the same fields before loading. Callers look the map
// up by the resolved path too, so the narrowing holds whether the configured value
// is the canonical directory or a registry alias, and whether or not the collected
// provisioning paths have already been canonicalized upstream.
func candleEmbeddingModelExcludePatterns(cfg *config.RouterConfig) map[string][]string {
	excluded := make(map[string][]string)
	if cfg.EmbeddingModels.EmbeddingBackend() != config.EmbeddingBackendCandle {
		return excluded
	}

	for path := range candleEmbeddingModelRequiredFiles(cfg) {
		resolved := config.ResolveModelPath(path)
		if resolved == "" || !strings.HasPrefix(resolved, "models/") {
			continue
		}
		excluded[resolved] = append([]string(nil), onnxWeightExcludePatterns...)
	}
	return excluded
}

// addEmbeddingModelRequiredFiles applies the completeness contract of the
// embedding runtime compiled into this binary. The public "candle" backend
// name is intentionally stable when an ONNX build replaces candle-binding at
// link time, so the contract must be selected at compile time too.
func addEmbeddingModelRequiredFiles(cfg *config.RouterConfig, requiredFilesByModel map[string][]string) {
	if cfg.EmbeddingModels.UsesRemoteEmbeddingBackend() {
		return
	}

	for path, files := range runtimeEmbeddingModelRequiredFiles(cfg) {
		if path == "" || !strings.HasPrefix(path, "models/") {
			continue
		}

		existing := requiredFilesByModel[path]
		for _, fileName := range files {
			if !slices.Contains(existing, fileName) {
				existing = append(existing, fileName)
			}
		}
		requiredFilesByModel[path] = existing
	}
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
