package extproc

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	. "github.com/onsi/ginkgo/v2"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
)

var extprocTestModelWeightCandidates = []string{
	"model.safetensors",
	"model.safetensors.index.json",
	"pytorch_model.bin",
	"adapter_model.safetensors",
}

// CreateTestRouter creates a properly initialized router for testing.
func CreateTestRouter(cfg *config.RouterConfig) (*OpenAIRouter, error) {
	classifierCfg := cloneRouterConfigForTest(cfg)
	categoryMapping, err := loadTestCategoryMapping(classifierCfg)
	if err != nil {
		return nil, err
	}
	if !extprocTestModelArtifactsAvailable(classifierCfg.CategoryModel.ModelID) {
		classifierCfg.CategoryModel.ModelID = ""
		classifierCfg.CategoryMappingPath = ""
		categoryMapping = nil
	}

	piiMapping, err := loadTestPIIMapping(classifierCfg)
	if err != nil {
		return nil, err
	}

	err = initTestBERTModel(classifierCfg)
	if err != nil {
		return nil, err
	}

	semanticCache, err := newTestSemanticCache(classifierCfg)
	if err != nil {
		return nil, err
	}

	toolsDatabase, err := newTestToolsDatabase(classifierCfg)
	if err != nil {
		return nil, err
	}

	classifier, err := classification.NewClassifier(classifierCfg, categoryMapping, piiMapping, nil)
	if err != nil {
		return nil, err
	}

	return &OpenAIRouter{
		Config:               cfg,
		CategoryDescriptions: cfg.GetCategoryDescriptions(),
		Classifier:           classifier,
		Cache:                semanticCache,
		ToolsDatabase:        toolsDatabase,
		ResponseAPIFilter:    newTestResponseAPIFilter(cfg),
		CredentialResolver:   newTestCredentialResolver(cfg),
	}, nil
}

func cloneRouterConfigForTest(cfg *config.RouterConfig) *config.RouterConfig {
	if cfg == nil {
		return nil
	}
	clone := *cfg
	return &clone
}

func loadTestCategoryMapping(cfg *config.RouterConfig) (*classification.CategoryMapping, error) {
	if cfg == nil || cfg.CategoryMappingPath == "" {
		return nil, nil
	}
	if _, err := os.Stat(cfg.CategoryMappingPath); err != nil {
		return nil, nil
	}
	return classification.LoadCategoryMapping(cfg.CategoryMappingPath)
}

func loadTestPIIMapping(cfg *config.RouterConfig) (*classification.PIIMapping, error) {
	if cfg.PIIMappingPath == "" {
		return nil, nil
	}
	if _, err := os.Stat(cfg.PIIMappingPath); err != nil {
		return nil, nil
	}
	return classification.LoadPIIMapping(cfg.PIIMappingPath)
}

func initTestBERTModel(cfg *config.RouterConfig) error {
	if err := candle_binding.InitModel(cfg.BertModelPath, cfg.UseCPU); err != nil {
		return fmt.Errorf("failed to initialize BERT model: %w", err)
	}
	return nil
}

func newTestSemanticCache(cfg *config.RouterConfig) (cache.CacheBackend, error) {
	return cache.NewCacheBackend(cache.CacheConfig{
		BackendType:         cache.InMemoryCacheType,
		Enabled:             cfg.Enabled,
		SimilarityThreshold: cfg.GetCacheSimilarityThreshold(),
		MaxEntries:          cfg.MaxEntries,
		TTLSeconds:          cfg.TTLSeconds,
		EvictionPolicy:      cache.EvictionPolicyType(cfg.EvictionPolicy),
		EmbeddingModel:      cfg.EmbeddingModel,
	})
}

func newTestToolsDatabase(cfg *config.RouterConfig) (*tools.ToolsDatabase, error) {
	toolCfg := cfg.Tools
	toolsSimilarityThreshold := float32(0.2)
	if toolCfg.SimilarityThreshold != nil {
		toolsSimilarityThreshold = *toolCfg.SimilarityThreshold
	}

	toolsDatabase := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
		SimilarityThreshold: toolsSimilarityThreshold,
		Enabled:             toolCfg.Enabled,
		ModelType:           cfg.EmbeddingConfig.ModelType,
		TargetDimension:     cfg.EmbeddingConfig.TargetDimension,
	})
	if !toolCfg.Enabled || toolCfg.ToolsDBPath == "" {
		return toolsDatabase, nil
	}
	if err := toolsDatabase.LoadToolsFromFile(toolCfg.ToolsDBPath); err != nil {
		return nil, fmt.Errorf("failed to load tools database: %w", err)
	}
	return toolsDatabase, nil
}

func newTestResponseAPIFilter(cfg *config.RouterConfig) *ResponseAPIFilter {
	if !cfg.ResponseAPI.Enabled {
		return nil
	}
	return NewResponseAPIFilter(NewMockResponseStore())
}

func newTestCredentialResolver(cfg *config.RouterConfig) *authz.CredentialResolver {
	credResolver := authz.NewCredentialResolver(
		authz.NewHeaderInjectionProvider(authz.DefaultHeaderMap()),
		authz.NewStaticConfigProvider(cfg),
	)
	credResolver.SetFailOpen(true)
	return credResolver
}

func findExtprocTestProjectRoot() string {
	wd, err := os.Getwd()
	if err != nil {
		return ""
	}
	for current := wd; current != filepath.Dir(current); current = filepath.Dir(current) {
		if _, err := os.Stat(filepath.Join(current, "models")); err == nil {
			return current
		}
	}
	return ""
}

func resolveExtprocTestPath(relativePath string) string {
	if _, err := os.Stat(relativePath); err == nil {
		return relativePath
	}
	root := findExtprocTestProjectRoot()
	if root == "" {
		return relativePath
	}
	trimmed := strings.TrimPrefix(relativePath, "../../../../")
	trimmed = strings.TrimPrefix(trimmed, "../../../../../")
	absolute := filepath.Join(root, trimmed)
	if _, err := os.Stat(absolute); err == nil {
		return absolute
	}
	return relativePath
}

func extprocTestModelArtifactsAvailable(modelPath string) bool {
	info, err := os.Stat(modelPath)
	if err != nil || !info.IsDir() {
		return false
	}
	if _, err := os.Stat(filepath.Join(modelPath, "config.json")); err != nil {
		return false
	}
	for _, candidate := range extprocTestModelWeightCandidates {
		if _, err := os.Stat(filepath.Join(modelPath, candidate)); err == nil {
			return true
		}
	}
	if matches, _ := filepath.Glob(filepath.Join(modelPath, "*.safetensors")); len(matches) > 0 {
		return true
	}
	if matches, _ := filepath.Glob(filepath.Join(modelPath, "*.bin")); len(matches) > 0 {
		return true
	}
	return false
}

func skipExtprocSpecIfModelArtifactsMissing(label string, modelPath string) {
	if extprocTestModelArtifactsAvailable(modelPath) {
		return
	}
	Skip(fmt.Sprintf("%s artifacts not available at %s (missing model weights)", label, modelPath))
}
