package classification

import (
	"fmt"
	"strings"
	"sync"

	"golang.org/x/sync/errgroup"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (b *classifierOptionBuilder) buildKeywordClassifierOption() (option, error) {
	if len(b.cfg.KeywordRules) == 0 {
		return nil, nil
	}
	keywordClassifier, err := NewKeywordClassifier(b.cfg.KeywordRules)
	if err != nil {
		logging.ComponentErrorEvent("classifier", "keyword_classifier_create_failed", map[string]interface{}{
			"error": err.Error(),
		})
		return nil, err
	}
	return withKeywordClassifier(keywordClassifier), nil
}

func (b *classifierOptionBuilder) buildEmbeddingClassifierOption() (option, error) {
	if len(b.cfg.EmbeddingRules) == 0 {
		return nil, nil
	}
	optConfig := b.cfg.EmbeddingConfig
	if strings.EqualFold(strings.TrimSpace(optConfig.ModelType), "multimodal") {
		if err := b.initMultiModalIfNeeded("embedding_rules with model_type=multimodal"); err != nil {
			return nil, err
		}
	}
	// Eagerly initialize the OpenVINO embedding model so that
	// NewEmbeddingClassifier's preload step can use it. Skip if
	// EMBEDDING_BACKEND_OVERRIDE forces a different backend at runtime.
	backendOverride := embeddingBackendOverride()
	if backendOverride == "" {
		backendOverride = strings.ToLower(strings.TrimSpace(optConfig.Backend))
	}
	if backendOverride == "openvino" {
		modelType := strings.ToLower(strings.TrimSpace(optConfig.ModelType))
		if err := initOpenVINOModel(modelType, b.cfg.MmBertModelPath, b.cfg.Qwen3ModelPath, b.cfg.UseCPU); err != nil {
			logging.ComponentWarnEvent("classifier", "openvino_eager_init_failed", map[string]interface{}{
				"error": err.Error(),
			})
		}
	}
	provider, err := b.embeddingProviderForRules()
	if err != nil {
		return nil, err
	}
	keywordEmbeddingClassifier, err := NewEmbeddingClassifierWithProvider(b.cfg.EmbeddingRules, optConfig, provider)
	if err != nil {
		logging.ComponentErrorEvent("classifier", "embedding_classifier_create_failed", map[string]interface{}{
			"error": err.Error(),
		})
		return nil, err
	}
	return withKeywordEmbeddingClassifier(createEmbeddingInitializer(), keywordEmbeddingClassifier), nil
}

func (b *classifierOptionBuilder) embeddingProviderForRules() (embedding.Provider, error) {
	if b.cfg == nil || !b.cfg.EmbeddingModels.UsesRemoteEmbeddingBackend() {
		return nil, nil
	}
	provider, err := embedding.NewProvider(b.cfg.EmbeddingModels, embedding.ProviderOptions{})
	if err != nil {
		logging.ComponentErrorEvent("classifier", "embedding_provider_create_failed", map[string]interface{}{
			"backend": b.cfg.EmbeddingModels.EmbeddingBackend(),
			"error":   err.Error(),
		})
		return nil, err
	}
	return provider, nil
}

func (b *classifierOptionBuilder) buildContextClassifierOption() (option, error) {
	if len(b.cfg.ContextRules) == 0 {
		return nil, nil
	}
	tokenCounter := NewCalibratedTokenCounter(WithConservativeEstimate())
	return withCalibratedContextClassifier(NewContextClassifier(tokenCounter, b.cfg.ContextRules), tokenCounter), nil
}

func (b *classifierOptionBuilder) buildStructureClassifierOption() (option, error) {
	if len(b.cfg.StructureRules) == 0 {
		return nil, nil
	}
	structureClassifier, err := NewStructureClassifier(b.cfg.StructureRules)
	if err != nil {
		logging.ComponentErrorEvent("classifier", "structure_classifier_create_failed", map[string]interface{}{
			"error": err.Error(),
		})
		return nil, err
	}
	return withStructureClassifier(structureClassifier), nil
}

func (b *classifierOptionBuilder) buildEventClassifierOption() (option, error) {
	if len(b.cfg.EventRules) == 0 {
		return nil, nil
	}
	return withEventClassifier(NewEventClassifier(b.cfg.EventRules)), nil
}

func (b *classifierOptionBuilder) buildReaskClassifierOption() (option, error) {
	if len(b.cfg.ReaskRules) == 0 {
		return nil, nil
	}
	reaskClassifier, err := NewReaskClassifier(b.cfg.ReaskRules, b.defaultEmbeddingModelType())
	if err != nil {
		logging.Errorf("Failed to create reask classifier: %v", err)
		return nil, err
	}
	return withReaskClassifier(reaskClassifier), nil
}

func (b *classifierOptionBuilder) buildComplexityClassifierOption() (option, error) {
	if len(b.cfg.ComplexityRules) == 0 {
		return nil, nil
	}
	modelType := b.defaultEmbeddingModelType()
	if config.HasImageCandidatesInRules(b.cfg.ComplexityRules) {
		if err := b.initMultiModalIfNeeded("complexity image_candidates"); err != nil {
			return nil, err
		}
	}
	if strings.EqualFold(strings.TrimSpace(modelType), "multimodal") {
		if err := b.initMultiModalIfNeeded("complexity model_type=multimodal"); err != nil {
			return nil, err
		}
	}
	complexityClassifier, err := NewComplexityClassifier(
		b.cfg.ComplexityRules,
		modelType,
		b.cfg.ComplexityModel.WithDefaults().PrototypeScoring,
	)
	if err != nil {
		logging.ComponentErrorEvent("classifier", "complexity_classifier_create_failed", map[string]interface{}{
			"model_type": modelType,
			"error":      err.Error(),
		})
		return nil, err
	}
	return withComplexityClassifier(complexityClassifier), nil
}

func (b *classifierOptionBuilder) buildContrastiveJailbreakClassifiersOption() (option, error) {
	contrastiveClassifiers := make(map[string]*ContrastiveJailbreakClassifier)
	defaultModelType := b.cfg.EmbeddingConfig.ModelType
	if strings.EqualFold(strings.TrimSpace(defaultModelType), "multimodal") {
		if err := b.initMultiModalIfNeeded("contrastive jailbreak with model_type=multimodal"); err != nil {
			return nil, err
		}
	}

	var mu sync.Mutex
	var group errgroup.Group
	group.SetLimit(classifierBuildParallelism(len(b.cfg.JailbreakRules)))

	for _, rule := range b.cfg.JailbreakRules {
		if rule.Method != "contrastive" {
			continue
		}
		group.Go(func() error {
			cjc, err := NewContrastiveJailbreakClassifier(rule, defaultModelType)
			if err != nil {
				logging.ComponentErrorEvent("classifier", "contrastive_jailbreak_classifier_create_failed", map[string]interface{}{
					"rule":       rule.Name,
					"model_type": defaultModelType,
					"error":      err.Error(),
				})
				return err
			}
			mu.Lock()
			contrastiveClassifiers[rule.Name] = cjc
			mu.Unlock()
			return nil
		})
	}

	if err := group.Wait(); err != nil {
		return nil, err
	}
	if len(contrastiveClassifiers) == 0 {
		return nil, nil
	}
	logging.ComponentEvent("classifier", "contrastive_jailbreak_classifiers_initialized", map[string]interface{}{
		"count": len(contrastiveClassifiers),
	})
	return withContrastiveJailbreakClassifiers(contrastiveClassifiers), nil
}

func (b *classifierOptionBuilder) buildAuthzClassifierOption() (option, error) {
	roleBindings := b.cfg.GetRoleBindings()
	if len(roleBindings) == 0 {
		return nil, nil
	}
	authzClassifier, err := NewAuthzClassifier(roleBindings)
	if err != nil {
		return nil, fmt.Errorf("failed to create authz classifier: %w", err)
	}
	logging.ComponentEvent("classifier", "authz_classifier_initialized", map[string]interface{}{
		"role_bindings": len(roleBindings),
	})
	return withAuthzClassifier(authzClassifier), nil
}

func (b *classifierOptionBuilder) buildKBClassifiersOption() (option, error) {
	if len(b.cfg.KnowledgeBases) == 0 {
		return nil, nil
	}
	modelType := strings.ToLower(strings.TrimSpace(b.cfg.EmbeddingConfig.ModelType))
	if modelType == "" {
		modelType = "qwen3"
	}
	classifiers := make(map[string]*KnowledgeBaseClassifier, len(b.cfg.KnowledgeBases))

	var mu sync.Mutex
	var group errgroup.Group
	group.SetLimit(classifierBuildParallelism(len(b.cfg.KnowledgeBases)))

	for _, kb := range b.cfg.KnowledgeBases {
		group.Go(func() error {
			classifier, err := NewKnowledgeBaseClassifier(kb, modelType, b.cfg.ConfigBaseDir)
			if err != nil {
				logging.ComponentWarnEvent("classifier", "knowledge_base_classifier_create_failed", map[string]interface{}{
					"knowledge_base":     kb.Name,
					"error":              err.Error(),
					"kb_signals_enabled": false,
				})
				return nil
			}
			mu.Lock()
			classifiers[kb.Name] = classifier
			mu.Unlock()
			logging.ComponentEvent("classifier", "knowledge_base_classifier_initialized", map[string]interface{}{
				"knowledge_base": kb.Name,
				"labels":         classifier.LabelCount(),
			})
			return nil
		})
	}

	if err := group.Wait(); err != nil {
		return nil, err
	}
	if len(classifiers) == 0 {
		return nil, nil
	}
	return withKBClassifiers(classifiers), nil
}
