package classification

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (b *classifierOptionBuilder) addCategoryClassifier(categoryMapping *CategoryMapping) {
	if b.cfg.CategoryModel.ModelID == "" {
		return
	}
	var categoryInitializer CategoryInitializer
	var categoryInference CategoryInference
	if b.cfg.CategoryModel.UseMmBERT32K {
		logging.ComponentEvent("classifier", "category_classifier_backend_selected", map[string]interface{}{
			"backend": "mmbert_32k",
		})
		categoryInitializer = createMmBERT32KCategoryInitializer()
		categoryInference = createMmBERT32KCategoryInference()
	} else {
		categoryInitializer = createCategoryInitializer()
		categoryInference = createCategoryInference()
	}
	b.options = append(b.options, withCategory(categoryMapping, categoryInitializer, categoryInference))
}

func (b *classifierOptionBuilder) addMCPCategoryClassifier() {
	if !b.cfg.MCPCategoryModel.Enabled {
		return
	}
	mcpInit := createMCPCategoryInitializer()
	mcpInf := createMCPCategoryInference(mcpInit)
	b.options = append(b.options, withMCPCategory(mcpInit, mcpInf))
}

// addComplexityModelClassifier wires the trained complexity classifier when at
// least one complexity rule opts into it via method: model. Embedding-mode rules
// are unaffected.
//
// Config validation (validateComplexityRules) already requires model_id and
// complexity_mapping_path whenever a model-mode rule exists, so the guards below
// are defensive: they turn any residual gap into a hard startup error instead of
// silently wiring nothing and leaving the signal permanently inert.
func (b *classifierOptionBuilder) addComplexityModelClassifier() error {
	if !config.HasModelComplexityRule(b.cfg.ComplexityRules) {
		return nil
	}
	cfg := b.cfg.ComplexityModel.Classifier
	if cfg.ModelID == "" {
		return fmt.Errorf("complexity rules use method: model but global.model_catalog.modules.complexity.classifier.model_id is not set")
	}
	if cfg.ComplexityMappingPath == "" {
		return fmt.Errorf("complexity classifier model_id is set but complexity_mapping_path is empty; a class-index -> difficulty mapping is required")
	}

	mapping, err := LoadComplexityMapping(cfg.ComplexityMappingPath)
	if err != nil {
		return fmt.Errorf("failed to load complexity mapping: %w", err)
	}

	logging.ComponentEvent("classifier", "complexity_model_classifier_selected", map[string]interface{}{
		"model_ref": cfg.ModelID,
	})
	b.options = append(b.options, withComplexityModel(mapping, createComplexityInitializer(), createComplexityInference()))
	return nil
}

func buildJailbreakDependencies(cfg *config.RouterConfig) (JailbreakInitializer, JailbreakInference, error) {
	jailbreakInference, err := createJailbreakInference(&cfg.PromptGuard, cfg)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create jailbreak inference: %w", err)
	}
	if cfg.PromptGuard.UseVLLM {
		return nil, jailbreakInference, nil
	}
	if cfg.PromptGuard.UseMmBERT32K {
		return createMmBERT32KJailbreakInitializer(), jailbreakInference, nil
	}
	return createJailbreakInitializer(), jailbreakInference, nil
}

func buildPIIDependencies(cfg *config.RouterConfig) (PIIInitializer, PIIInference) {
	if cfg.PIIModel.UseMmBERT32K {
		logging.ComponentEvent("classifier", "pii_detector_backend_selected", map[string]interface{}{
			"backend": "mmbert_32k",
		})
		return createMmBERT32KPIIInitializer(), createMmBERT32KPIIInference()
	}
	return createPIIInitializer(), createPIIInference()
}
