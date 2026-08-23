package classification

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// NewRecipeClassifiersFromConfig builds and initializes the classifier graph
// from explicit Recipes in one canonical Router config. It never interprets
// root-level routing fields as an implicit Recipe.
func NewRecipeClassifiersFromConfig(cfg *config.RouterConfig) (*RecipeClassifiers, error) {
	if cfg == nil {
		return nil, fmt.Errorf("config is nil")
	}

	categoryMapping, err := loadRecipeCategoryMapping(cfg)
	if err != nil {
		return nil, err
	}
	piiMapping, err := loadRecipePIIMapping(cfg)
	if err != nil {
		return nil, err
	}
	jailbreakMapping, err := loadRecipeJailbreakMapping(cfg)
	if err != nil {
		return nil, err
	}

	classifiers, err := BuildRecipeClassifiers(cfg, categoryMapping, piiMapping, jailbreakMapping)
	if err != nil {
		return nil, fmt.Errorf("build Recipe classifiers: %w", err)
	}
	if err := classifiers.InitializeRuntime(); err != nil {
		return nil, fmt.Errorf("initialize Recipe classifiers: %w", err)
	}
	return classifiers, nil
}

func loadRecipeCategoryMapping(cfg *config.RouterConfig) (*CategoryMapping, error) {
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

func loadRecipePIIMapping(cfg *config.RouterConfig) (*PIIMapping, error) {
	if !cfg.NeedsPIIMappingForRouting() {
		return nil, nil
	}

	piiMapping, err := LoadPIIMapping(cfg.PIIMappingPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load PII mapping: %w", err)
	}
	return piiMapping, nil
}

func loadRecipeJailbreakMapping(cfg *config.RouterConfig) (*JailbreakMapping, error) {
	if !cfg.NeedsJailbreakMappingForRouting() {
		return nil, nil
	}

	jailbreakMapping, err := LoadJailbreakMapping(cfg.PromptGuard.JailbreakMappingPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load jailbreak mapping: %w", err)
	}
	return jailbreakMapping, nil
}
