package classification

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// NewLegacyClassifierFromConfig loads mapping assets and builds the legacy
// classifier runtime for callers that still use the non-unified path.
func NewLegacyClassifierFromConfig(cfg *config.RouterConfig, runtimeOptions ...RecipeRuntimeOptions) (*Classifier, error) {
	if cfg == nil {
		return nil, fmt.Errorf("config is nil")
	}
	plan, err := config.CompileModelBindings(cfg)
	if err != nil {
		return nil, err
	}
	if cfg.RoutingScope == "" {
		cfg = cfg.ConfigForRecipe(cfg.DefaultRecipe())
	}
	cfg, err = config.ProjectRecipeModelBindings(cfg, plan, cfg.RoutingScope)
	if err != nil {
		return nil, err
	}

	categoryMapping, err := loadLegacyCategoryMapping(cfg)
	if err != nil {
		return nil, err
	}
	piiMapping, err := loadLegacyPIIMapping(cfg)
	if err != nil {
		return nil, err
	}
	jailbreakMapping, err := loadLegacyJailbreakMapping(cfg)
	if err != nil {
		return nil, err
	}

	classifier, err := buildClassifierWithAdmission(cfg, categoryMapping, piiMapping, jailbreakMapping, nil, runtimeOptions...)
	if err != nil {
		return nil, fmt.Errorf("failed to create classifier: %w", err)
	}
	if err := classifier.InitializeRuntime(); err != nil {
		_ = classifier.Close()
		return nil, fmt.Errorf("failed to initialize classifier: %w", err)
	}
	return classifier, nil
}

func loadLegacyCategoryMapping(cfg *config.RouterConfig) (*CategoryMapping, error) {
	useMCPCategories := cfg.CategoryModel.ModelID == "" && cfg.MCPCategoryModel.Enabled
	if useMCPCategories && cfg.UsesSignalTypeInRouting(config.SignalTypeDomain) {
		logging.Infof("Category mapping will be loaded from MCP server")
		return nil, nil
	}
	if !cfg.NeedsCategoryMappingForRouting() {
		return nil, nil
	}

	categoryMapping, err := LoadCategoryMapping(cfg.CategoryMappingPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load category mapping: %w", err)
	}
	return categoryMapping, nil
}

func loadLegacyPIIMapping(cfg *config.RouterConfig) (*PIIMapping, error) {
	if !cfg.NeedsPIIMappingForRouting() {
		return nil, nil
	}

	piiMapping, err := LoadPIIMapping(cfg.PIIMappingPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load PII mapping: %w", err)
	}
	return piiMapping, nil
}

func loadLegacyJailbreakMapping(cfg *config.RouterConfig) (*JailbreakMapping, error) {
	if !cfg.NeedsJailbreakMappingForRouting() {
		return nil, nil
	}

	jailbreakMapping, err := LoadJailbreakMapping(cfg.PromptGuard.JailbreakMappingPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load jailbreak mapping: %w", err)
	}
	return jailbreakMapping, nil
}
