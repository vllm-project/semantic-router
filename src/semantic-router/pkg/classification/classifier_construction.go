package classification

import (
	"fmt"
	"runtime"
	"strings"
	"sync"

	"golang.org/x/sync/errgroup"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type classifierOptionBuilder struct {
	cfg                *config.RouterConfig
	options            []option
	multiModalInitOnce sync.Once
	multiModalInitErr  error
}

func newClassifierOptionBuilder(cfg *config.RouterConfig, options []option) *classifierOptionBuilder {
	return &classifierOptionBuilder{cfg: cfg, options: options}
}

func (b *classifierOptionBuilder) build(categoryMapping *CategoryMapping) ([]option, error) {
	steps := []func() (option, error){
		b.buildKeywordClassifierOption,
		b.buildEmbeddingClassifierOption,
		b.buildContextClassifierOption,
		b.buildStructureClassifierOption,
		b.buildReaskClassifierOption,
		b.buildComplexityClassifierOption,
		b.buildContrastiveJailbreakClassifiersOption,
		b.buildAuthzClassifierOption,
		b.buildKBClassifiersOption,
	}
	parallelOptions, err := b.buildParallelOptions(steps)
	if err != nil {
		return nil, err
	}
	b.options = append(b.options, parallelOptions...)
	b.addCategoryClassifier(categoryMapping)
	b.addMCPCategoryClassifier()
	return b.options, nil
}

func (b *classifierOptionBuilder) buildParallelOptions(steps []func() (option, error)) ([]option, error) {
	if len(steps) == 0 {
		return nil, nil
	}

	results := make([]option, len(steps))
	var group errgroup.Group
	group.SetLimit(classifierBuildParallelism(len(steps)))

	for i, step := range steps {
		group.Go(func() error {
			opt, err := step()
			if err != nil {
				return err
			}
			results[i] = opt
			return nil
		})
	}

	if err := group.Wait(); err != nil {
		return nil, err
	}

	options := make([]option, 0, len(results))
	for _, opt := range results {
		if opt != nil {
			options = append(options, opt)
		}
	}
	return options, nil
}

func classifierBuildParallelism(stepCount int) int {
	if stepCount <= 1 {
		return 1
	}
	parallelism := runtime.NumCPU()
	if parallelism <= 0 {
		parallelism = 1
	}
	if parallelism > stepCount {
		parallelism = stepCount
	}
	return parallelism
}

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
	keywordEmbeddingClassifier, err := NewEmbeddingClassifier(b.cfg.EmbeddingRules, optConfig)
	if err != nil {
		logging.ComponentErrorEvent("classifier", "embedding_classifier_create_failed", map[string]interface{}{
			"error": err.Error(),
		})
		return nil, err
	}
	return withKeywordEmbeddingClassifier(createEmbeddingInitializer(), keywordEmbeddingClassifier), nil
}

func (b *classifierOptionBuilder) buildContextClassifierOption() (option, error) {
	if len(b.cfg.ContextRules) == 0 {
		return nil, nil
	}
	tokenCounter := &CharacterBasedTokenCounter{}
	return withContextClassifier(NewContextClassifier(tokenCounter, b.cfg.ContextRules)), nil
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

func (b *classifierOptionBuilder) initMultiModalIfNeeded(reason string) error {
	b.multiModalInitOnce.Do(func() {
		mmPath := config.ResolveModelPath(b.cfg.MultiModalModelPath)
		if mmPath == "" {
			b.multiModalInitErr = fmt.Errorf("%s requires embedding_models.multimodal_model_path to be set", reason)
			return
		}
		if err := initMultiModalModel(mmPath, b.cfg.UseCPU); err != nil {
			b.multiModalInitErr = fmt.Errorf("failed to initialize multimodal model for %s: %w", reason, err)
			return
		}
		logging.ComponentEvent("classifier", "multimodal_embedding_initialized", map[string]interface{}{
			"reason":    reason,
			"model_ref": mmPath,
			"use_cpu":   b.cfg.UseCPU,
		})
	})
	return b.multiModalInitErr
}

func (b *classifierOptionBuilder) defaultEmbeddingModelType() string {
	modelType := b.cfg.EmbeddingConfig.ModelType
	if modelType == "" {
		return "qwen3"
	}
	return modelType
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

func (c *Classifier) logHeuristicClassifierInitialization() {
	if c.contextClassifier != nil {
		logging.ComponentEvent("classifier", "context_classifier_initialized", map[string]interface{}{
			"rules": len(c.contextClassifier.rules),
		})
	}
	if c.structureClassifier != nil {
		logging.ComponentEvent("classifier", "structure_classifier_initialized", map[string]interface{}{
			"rules": len(c.structureClassifier.rules),
		})
	}
}
