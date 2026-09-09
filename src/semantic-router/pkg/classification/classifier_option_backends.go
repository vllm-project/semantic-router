package classification

import (
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (b *classifierOptionBuilder) addCategoryClassifier(categoryMapping *CategoryMapping) error {
	// Keep the construction seam on the same validator as config loading and
	// BuildClassifier. This prevents an already-decoded config from bypassing
	// backend/model compatibility checks when this builder is used directly.
	if err := config.ValidateCategoryModelBackend(b.cfg); err != nil {
		return err
	}
	if b.cfg.CategoryModel.ModelID == "" && b.cfg.CategoryModel.Backend == nil {
		return nil
	}
	if b.cfg.CategoryModel.Backend != nil {
		return b.addRemoteCategoryClassifier(categoryMapping)
	}
	return b.addLocalCategoryClassifier(categoryMapping)
}

func (b *classifierOptionBuilder) addRemoteCategoryClassifier(categoryMapping *CategoryMapping) error {
	backendCfg := b.cfg.CategoryModel.Backend
	external, err := config.ResolveRemoteClassifierBackend(
		b.cfg,
		backendCfg,
		config.ModelRoleClassification,
		config.RemoteClassifierContractLabelDistribution,
	)
	if err != nil {
		return fmt.Errorf("failed to resolve category backend: %w", err)
	}
	if backendCfg.Protocol != config.RemoteClassifierProtocolHTTPClassify {
		return fmt.Errorf("category backend protocol %q is not supported", backendCfg.Protocol)
	}
	timeout := time.Duration(backendCfg.EffectiveDeadlineMs()) * time.Millisecond
	backend, err := newCategoryHTTPBackend(external, categoryMapping, timeout)
	if err != nil {
		return err
	}
	b.options = append(b.options, withCategory(categoryMapping, nil, backend))
	return nil
}

func (b *classifierOptionBuilder) addLocalCategoryClassifier(categoryMapping *CategoryMapping) error {
	variant, err := b.cfg.CategoryModel.EffectiveVariant()
	if err != nil {
		return err
	}
	categoryInitializer, categoryInference := categoryDependenciesForVariant(variant)
	b.options = append(b.options, withCategory(categoryMapping, categoryInitializer, categoryInference))
	return nil
}

func categoryDependenciesForVariant(variant string) (CategoryInitializer, CategoryInference) {
	switch variant {
	case config.CategoryVariantMmBERT32K:
		logging.ComponentEvent("classifier", "category_classifier_backend_selected", map[string]interface{}{
			"backend": "mmbert_32k",
		})
		return createMmBERT32KCategoryInitializer(), createMmBERT32KCategoryInference()
	case config.CategoryVariantModernBERT:
		logging.ComponentEvent("classifier", "category_classifier_backend_selected", map[string]interface{}{
			"backend": "modernbert",
		})
		return createModernBERTCategoryInitializer(), createModernBERTCategoryInference()
	case config.CategoryVariantCandle:
		logging.ComponentEvent("classifier", "category_classifier_backend_selected", map[string]interface{}{
			"backend": "candle",
		})
		return createCandleCategoryInitializer(), CandleCategoryInferenceImpl{}
	default:
		return createCategoryInitializer(), createCategoryInference()
	}
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

func buildJailbreakDependencies(cfg *config.RouterConfig, jailbreakMapping *JailbreakMapping) (JailbreakInitializer, SequenceClassifierBackend, error) {
	jailbreakInference, err := createJailbreakInference(&cfg.PromptGuard, cfg, jailbreakMapping)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create jailbreak inference: %w", err)
	}
	if cfg.PromptGuard.Protocol != "" {
		// Remote backends have no local model to initialize.
		return nil, jailbreakInference, nil
	}
	switch cfg.PromptGuard.Variant {
	case config.PromptGuardVariantMmBERT32K:
		return createMmBERT32KJailbreakInitializer(), jailbreakInference, nil
	default:
		return createJailbreakInitializer(), jailbreakInference, nil
	}
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
