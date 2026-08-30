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
	timeout := time.Duration(backendCfg.EffectiveTimeoutSeconds()) * time.Second
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
	var categoryInitializer CategoryInitializer
	var categoryInference CategoryInference
	switch variant {
	case config.CategoryVariantMmBERT32K:
		logging.ComponentEvent("classifier", "category_classifier_backend_selected", map[string]interface{}{
			"backend": "mmbert_32k",
		})
		categoryInitializer = createMmBERT32KCategoryInitializer()
		categoryInference = createMmBERT32KCategoryInference()
	case config.CategoryVariantModernBERT:
		logging.ComponentEvent("classifier", "category_classifier_backend_selected", map[string]interface{}{
			"backend": "modernbert",
		})
		categoryInitializer = createModernBERTCategoryInitializer()
		categoryInference = createCategoryInference()
	case config.CategoryVariantCandle:
		logging.ComponentEvent("classifier", "category_classifier_backend_selected", map[string]interface{}{
			"backend": "candle",
		})
		categoryInitializer = createCandleCategoryInitializer()
		categoryInference = CandleCategoryInferenceImpl{}
	default:
		categoryInitializer = createCategoryInitializer()
		categoryInference = createCategoryInference()
	}
	b.options = append(b.options, withCategory(categoryMapping, categoryInitializer, categoryInference))
	return nil
}

func (b *classifierOptionBuilder) addMCPCategoryClassifier() {
	if !b.cfg.MCPCategoryModel.Enabled {
		return
	}
	mcpInit := createMCPCategoryInitializer()
	mcpInf := createMCPCategoryInference(mcpInit)
	b.options = append(b.options, withMCPCategory(mcpInit, mcpInf))
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
