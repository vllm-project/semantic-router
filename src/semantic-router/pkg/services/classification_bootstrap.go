package services

import (
	"fmt"
	"os"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type legacyClassifierMappings struct {
	category  *classification.CategoryMapping
	pii       *classification.PIIMapping
	jailbreak *classification.JailbreakMapping
}

// NewClassificationServiceWithAutoDiscovery creates a service with auto-discovery.
func NewClassificationServiceWithAutoDiscovery(routerConfig *config.RouterConfig) (*ClassificationService, error) {
	wd, _ := os.Getwd()
	logging.Debugf("Debug: Current working directory: %s", wd)
	logging.Debugf("Debug: Attempting to discover models in: ./models")

	unifiedClassifier, ucErr := classification.AutoInitializeUnifiedClassifierWithRegistry(
		resolveUnifiedModelsPath(routerConfig),
		classificationModelRegistry(routerConfig),
	)
	if ucErr != nil {
		logging.Infof("Unified classifier auto-discovery failed: %v", ucErr)
	}

	legacyClassifier, lcErr := createLegacyClassifier(routerConfig)
	if lcErr != nil {
		logging.Warnf("Legacy classifier initialization failed: %v", lcErr)
	}
	if unifiedClassifier == nil && legacyClassifier == nil {
		logging.Warnf("No classifier initialized. Using placeholder service.")
	}
	return NewUnifiedClassificationService(unifiedClassifier, legacyClassifier, routerConfig), nil
}

func classificationModelRegistry(routerConfig *config.RouterConfig) map[string]string {
	if routerConfig == nil {
		return nil
	}
	return routerConfig.MoMRegistry
}

func resolveUnifiedModelsPath(routerConfig *config.RouterConfig) string {
	modelsPath := "./models"
	if routerConfig == nil || routerConfig.CategoryModel.ModelID == "" {
		return modelsPath
	}

	// Extract the models directory from the model path.
	// Example: "models/mom-domain-classifier" -> "models".
	if idx := strings.Index(routerConfig.CategoryModel.ModelID, "/"); idx > 0 {
		return routerConfig.CategoryModel.ModelID[:idx]
	}
	return modelsPath
}

func createLegacyClassifier(routerConfig *config.RouterConfig) (*classification.Classifier, error) {
	if routerConfig == nil {
		return nil, fmt.Errorf("config is nil")
	}

	mappings, err := loadLegacyClassifierMappings(routerConfig)
	if err != nil {
		return nil, err
	}

	classifier, err := classification.NewClassifier(
		routerConfig,
		mappings.category,
		mappings.pii,
		mappings.jailbreak,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create classifier: %w", err)
	}

	return classifier, nil
}

func loadLegacyClassifierMappings(routerConfig *config.RouterConfig) (*legacyClassifierMappings, error) {
	mappings := &legacyClassifierMappings{}

	if useMCPCategories(routerConfig) {
		// Categories will be loaded from the MCP server during classifier runtime initialization.
		logging.Infof("Category mapping will be loaded from MCP server")
	} else if path := strings.TrimSpace(routerConfig.CategoryMappingPath); path != "" {
		categoryMapping, err := classification.LoadCategoryMapping(path)
		if err != nil {
			return nil, fmt.Errorf("failed to load category mapping: %w", err)
		}
		mappings.category = categoryMapping
	}

	if path := strings.TrimSpace(routerConfig.PIIMappingPath); path != "" {
		piiMapping, err := classification.LoadPIIMapping(path)
		if err != nil {
			return nil, fmt.Errorf("failed to load PII mapping: %w", err)
		}
		mappings.pii = piiMapping
	}

	if path := strings.TrimSpace(routerConfig.PromptGuard.JailbreakMappingPath); path != "" {
		jailbreakMapping, err := classification.LoadJailbreakMapping(path)
		if err != nil {
			return nil, fmt.Errorf("failed to load jailbreak mapping: %w", err)
		}
		mappings.jailbreak = jailbreakMapping
	}

	return mappings, nil
}

func useMCPCategories(routerConfig *config.RouterConfig) bool {
	return routerConfig != nil &&
		routerConfig.CategoryModel.ModelID == "" &&
		routerConfig.MCPCategoryModel.Enabled
}
