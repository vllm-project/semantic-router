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
	cbCfg := backendCfg.CircuitBreaker

	httpBackend, err := newHTTPClassifierInference(external, categoryMapping, timeout)
	if err != nil {
		return fmt.Errorf("failed to create category http_classify backend: %w", err)
	}
	var innerBackend SequenceClassifierBackend = httpBackend
	if cbCfg != nil && cbCfg.Enabled {
		innerBackend = newCircuitBreakingBackend(httpBackend, cbCfg, external.ModelName)
	}
	cbWrappedBackend := &categoryHTTPBackend{backend: innerBackend}
	b.options = append(b.options, withCategory(categoryMapping, nil, cbWrappedBackend))
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

func buildJailbreakDependencies(cfg *config.RouterConfig, jailbreakMapping *JailbreakMapping) (JailbreakInitializer, SequenceClassifierBackend, error) {
	jailbreakInference, err := createJailbreakInference(&cfg.PromptGuard, cfg, jailbreakMapping)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create jailbreak inference: %w", err)
	}
	if cfg.PromptGuard.Protocol != "" {
		externalCfg := cfg.FindExternalModelByRole(config.ModelRoleGuardrail)
		if externalCfg != nil {
			cbCfg := cfg.PromptGuard.CircuitBreaker
			if cbCfg != nil && cbCfg.Enabled {
				jailbreakInference = newCircuitBreakingBackend(jailbreakInference, cbCfg, externalCfg.ModelName)
			}
		}
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

func buildPIIDependencies(cfg *config.RouterConfig, piiMapping *PIIMapping) (PIIInitializer, PIIInference, error) {
	if cfg.PIIModel.Backend != nil {
		if piiMapping == nil {
			// The mapping loader is skipped on purpose when no reachable routing
			// decision consumes the PII signal. With no consumer there is nothing
			// to build: IsPIIEnabled stays false and a dormant remote backend must
			// not fail startup on the missing mapping.
			logging.ComponentEvent("classifier", "pii_detector_backend_dormant", map[string]interface{}{
				"backend": "token_spans_http",
				"reason":  "no_reachable_pii_signal",
			})
			return nil, nil, nil
		}
		// Remote inference is fully constructed here and has no local model
		// lifecycle, so the initializer is nil, as it is for a remote category
		// backend.
		backendCfg := cfg.PIIModel.Backend
		external, err := config.ResolveRemoteClassifierBackend(
			cfg,
			backendCfg,
			config.ModelRoleClassification,
			config.RemoteClassifierContractTokenSpans,
		)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to resolve PII backend: %w", err)
		}
		if backendCfg.Protocol != config.RemoteClassifierProtocolHTTPClassify {
			return nil, nil, fmt.Errorf("PII backend protocol %q is not supported", backendCfg.Protocol)
		}
		timeout := time.Duration(backendCfg.EffectiveDeadlineMs()) * time.Millisecond
		inference, err := newPIIHTTPBackend(external, piiMapping, timeout)
		if err != nil {
			return nil, nil, err
		}
		logging.ComponentEvent("classifier", "pii_detector_backend_selected", map[string]interface{}{
			"backend": "token_spans_http",
		})
		return nil, inference, nil
	}
	if cfg.PIIModel.UseMmBERT32K {
		logging.ComponentEvent("classifier", "pii_detector_backend_selected", map[string]interface{}{
			"backend": "mmbert_32k",
		})
		return createMmBERT32KPIIInitializer(), createMmBERT32KPIIInference(), nil
	}
	return createPIIInitializer(), createPIIInference(), nil
}

// addComplexityBackend attaches the complexity signal's remote scorer, if one
// is configured. A nil backend leaves the local prototype path in place, so
// this is a no-op for every existing config.
//
// The contract decides which reader is built: score.v1 needs no label mapping
// and turns its number into a verdict through each rule's boundaries, while
// label_distribution.v1 reuses the shared sequence backend with the verdict
// vocabulary declared inline, exactly as the generic classifier signal does.
func (b *classifierOptionBuilder) addComplexityBackend() error {
	// The same validator globalConfigContractValidators runs at config load, so
	// a directly-built classifier cannot bypass the backend and boundary
	// checks either.
	if err := config.ValidateComplexityModelBackend(b.cfg); err != nil {
		return err
	}
	backendCfg := b.cfg.ComplexityModel.Backend
	if backendCfg == nil {
		return nil
	}

	external, err := config.ResolveRemoteClassifierBackend(
		b.cfg,
		backendCfg,
		config.ModelRoleClassification,
		config.ComplexityBackendContracts...,
	)
	if err != nil {
		return fmt.Errorf("failed to resolve complexity backend: %w", err)
	}
	deadline := time.Duration(backendCfg.EffectiveDeadlineMs()) * time.Millisecond

	switch backendCfg.Contract {
	case config.RemoteClassifierContractScore:
		scorer, err := newScoringHTTPBackend(external, deadline)
		if err != nil {
			return err
		}
		cbCfg := backendCfg.CircuitBreaker
		if cbCfg != nil && cbCfg.Enabled {
			scorer = newCircuitBreakingBackendScoring(scorer, cbCfg, external.ModelName)
		}
		logging.ComponentEvent("classifier", "complexity_backend_selected", map[string]interface{}{
			"contract": config.RemoteClassifierContractScore,
		})
		b.options = append(b.options, withComplexityScoreBackend(scorer))
	case config.RemoteClassifierContractLabelDistribution:
		var labels SequenceClassifierBackend
		labels, err = newHTTPClassifierInference(
			external,
			newDeclaredLabelMapping(ComplexityVerdictLabels),
			deadline,
		)
		if err != nil {
			return err
		}
		cbCfg := backendCfg.CircuitBreaker
		if cbCfg != nil && cbCfg.Enabled {
			labels = newCircuitBreakingBackend(labels, cbCfg, external.ModelName)
		}
		logging.ComponentEvent("classifier", "complexity_backend_selected", map[string]interface{}{
			"contract": config.RemoteClassifierContractLabelDistribution,
		})
		b.options = append(b.options, withComplexityLabelBackend(labels))
	default:
		// Unreachable: the validator above rejects anything else. Kept so a
		// future contract cannot be silently ignored here.
		return fmt.Errorf("complexity backend contract %q has no reader", backendCfg.Contract)
	}
	return nil
}
