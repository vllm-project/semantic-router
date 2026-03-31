package classification

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type classifierOptionBuilder struct {
	cfg                   *config.RouterConfig
	options               []option
	multiModalInitialized bool
}

func newClassifierOptionBuilder(cfg *config.RouterConfig, options []option) *classifierOptionBuilder {
	return &classifierOptionBuilder{cfg: cfg, options: options}
}

func (b *classifierOptionBuilder) build(categoryMapping *CategoryMapping) ([]option, error) {
	steps := []func() error{
		b.addKeywordClassifier,
		b.addEmbeddingClassifier,
		b.addContextClassifier,
		b.addStructureClassifier,
		b.addReaskClassifier,
		b.addComplexityClassifier,
		b.addContrastiveJailbreakClassifiers,
		b.addAuthzClassifier,
		b.addKBClassifiers,
	}
	for _, step := range steps {
		if err := step(); err != nil {
			return nil, err
		}
	}
	b.addCategoryClassifier(categoryMapping)
	b.addMCPCategoryClassifier()
	return b.options, nil
}

func (b *classifierOptionBuilder) addKeywordClassifier() error {
	if len(b.cfg.KeywordRules) == 0 {
		return nil
	}
	keywordClassifier, err := NewKeywordClassifier(b.cfg.KeywordRules)
	if err != nil {
		logging.Errorf("Failed to create keyword classifier: %v", err)
		return err
	}
	b.options = append(b.options, withKeywordClassifier(keywordClassifier))
	return nil
}

func (b *classifierOptionBuilder) addEmbeddingClassifier() error {
	if len(b.cfg.EmbeddingRules) == 0 {
		return nil
	}
	optConfig := b.cfg.EmbeddingConfig
	if strings.EqualFold(strings.TrimSpace(optConfig.ModelType), "multimodal") {
		if err := b.initMultiModalIfNeeded("embedding_rules with model_type=multimodal"); err != nil {
			return err
		}
	}
	keywordEmbeddingClassifier, err := NewEmbeddingClassifier(b.cfg.EmbeddingRules, optConfig)
	if err != nil {
		logging.Errorf("Failed to create keyword embedding classifier: %v", err)
		return err
	}
	b.options = append(b.options, withKeywordEmbeddingClassifier(createEmbeddingInitializer(), keywordEmbeddingClassifier))
	return nil
}

func (b *classifierOptionBuilder) addContextClassifier() error {
	if len(b.cfg.ContextRules) == 0 {
		return nil
	}
	tokenCounter := &CharacterBasedTokenCounter{}
	b.options = append(b.options, withContextClassifier(NewContextClassifier(tokenCounter, b.cfg.ContextRules)))
	return nil
}

func (b *classifierOptionBuilder) addStructureClassifier() error {
	if len(b.cfg.StructureRules) == 0 {
		return nil
	}
	structureClassifier, err := NewStructureClassifier(b.cfg.StructureRules)
	if err != nil {
		logging.Errorf("Failed to create structure classifier: %v", err)
		return err
	}
	b.options = append(b.options, withStructureClassifier(structureClassifier))
	return nil
}

func (b *classifierOptionBuilder) addReaskClassifier() error {
	if len(b.cfg.ReaskRules) == 0 {
		return nil
	}
	reaskClassifier, err := NewReaskClassifier(b.cfg.ReaskRules, b.defaultEmbeddingModelType())
	if err != nil {
		logging.Errorf("Failed to create reask classifier: %v", err)
		return err
	}
	b.options = append(b.options, withReaskClassifier(reaskClassifier))
	return nil
}

func (b *classifierOptionBuilder) addComplexityClassifier() error {
	if len(b.cfg.ComplexityRules) == 0 {
		return nil
	}
	modelType := b.defaultEmbeddingModelType()
	if config.HasImageCandidatesInRules(b.cfg.ComplexityRules) {
		if err := b.initMultiModalIfNeeded("complexity image_candidates"); err != nil {
			return err
		}
	}
	if strings.EqualFold(strings.TrimSpace(modelType), "multimodal") {
		if err := b.initMultiModalIfNeeded("complexity model_type=multimodal"); err != nil {
			return err
		}
	}
	complexityClassifier, err := NewComplexityClassifier(b.cfg.ComplexityRules, modelType)
	if err != nil {
		logging.Errorf("Failed to create complexity classifier: %v", err)
		return err
	}
	b.options = append(b.options, withComplexityClassifier(complexityClassifier))
	return nil
}

func (b *classifierOptionBuilder) addContrastiveJailbreakClassifiers() error {
	contrastiveClassifiers := make(map[string]*ContrastiveJailbreakClassifier)
	defaultModelType := b.cfg.EmbeddingConfig.ModelType
	for _, rule := range b.cfg.JailbreakRules {
		if rule.Method != "contrastive" {
			continue
		}
		if strings.EqualFold(strings.TrimSpace(defaultModelType), "multimodal") {
			if err := b.initMultiModalIfNeeded("contrastive jailbreak with model_type=multimodal"); err != nil {
				return err
			}
		}
		cjc, err := NewContrastiveJailbreakClassifier(rule, defaultModelType)
		if err != nil {
			logging.Errorf("Failed to create contrastive jailbreak classifier for rule %q: %v", rule.Name, err)
			return err
		}
		contrastiveClassifiers[rule.Name] = cjc
	}
	if len(contrastiveClassifiers) == 0 {
		return nil
	}
	b.options = append(b.options, withContrastiveJailbreakClassifiers(contrastiveClassifiers))
	logging.Infof("Initialized %d contrastive jailbreak classifiers", len(contrastiveClassifiers))
	return nil
}

func (b *classifierOptionBuilder) addAuthzClassifier() error {
	roleBindings := b.cfg.GetRoleBindings()
	if len(roleBindings) == 0 {
		return nil
	}
	authzClassifier, err := NewAuthzClassifier(roleBindings)
	if err != nil {
		return fmt.Errorf("failed to create authz classifier: %w", err)
	}
	b.options = append(b.options, withAuthzClassifier(authzClassifier))
	logging.Infof("Authz classifier initialized with %d role bindings", len(roleBindings))
	return nil
}

func (b *classifierOptionBuilder) addCategoryClassifier(categoryMapping *CategoryMapping) {
	if b.cfg.CategoryModel.ModelID == "" {
		return
	}
	var categoryInitializer CategoryInitializer
	var categoryInference CategoryInference
	if b.cfg.CategoryModel.UseMmBERT32K {
		logging.Infof("Using mmBERT-32K for intent/category classification (32K context, YaRN RoPE)")
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
	if b.multiModalInitialized {
		return nil
	}
	mmPath := config.ResolveModelPath(b.cfg.MultiModalModelPath)
	if mmPath == "" {
		return fmt.Errorf("%s requires embedding_models.multimodal_model_path to be set", reason)
	}
	if err := initMultiModalModel(mmPath, b.cfg.UseCPU); err != nil {
		return fmt.Errorf("failed to initialize multimodal model for %s: %w", reason, err)
	}
	logging.Infof("Initialized multimodal embedding model for %s: %s", reason, mmPath)
	b.multiModalInitialized = true
	return nil
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
		logging.Infof("Using mmBERT-32K for PII detection (32K context, YaRN RoPE)")
		return createMmBERT32KPIIInitializer(), createMmBERT32KPIIInference()
	}
	return createPIIInitializer(), createPIIInference()
}

func (c *Classifier) initializeConfiguredCategoryRuntime() error {
	if c.IsCategoryEnabled() {
		return c.initializeCategoryClassifier()
	}
	if c.IsMCPCategoryEnabled() {
		return c.initializeMCPCategoryClassifier()
	}
	return nil
}

func (c *Classifier) initializeRequiredRuntimeClassifiers() error {
	steps := []struct {
		enabled bool
		init    func() error
	}{
		{enabled: c.IsJailbreakEnabled(), init: c.initializeJailbreakClassifier},
		{enabled: c.IsPIIEnabled(), init: c.initializePIIClassifier},
		{enabled: c.IsKeywordEmbeddingClassifierEnabled(), init: c.initializeKeywordEmbeddingClassifier},
	}
	for _, step := range steps {
		if !step.enabled {
			continue
		}
		if err := step.init(); err != nil {
			return err
		}
	}
	return nil
}

func (b *classifierOptionBuilder) addKBClassifiers() error {
	if len(b.cfg.KnowledgeBases) == 0 {
		return nil
	}
	modelType := strings.ToLower(strings.TrimSpace(b.cfg.EmbeddingConfig.ModelType))
	if modelType == "" {
		modelType = "qwen3"
	}
	classifiers := make(map[string]*KnowledgeBaseClassifier, len(b.cfg.KnowledgeBases))
	for _, kb := range b.cfg.KnowledgeBases {
		classifier, err := NewKnowledgeBaseClassifier(kb, modelType, b.cfg.ConfigBaseDir)
		if err != nil {
			logging.Warnf("[KnowledgeBase:%s] Failed to create classifier: %v (kb signals for this KB disabled)", kb.Name, err)
			continue
		}
		classifiers[kb.Name] = classifier
		logging.Infof("[KnowledgeBase:%s] Initialized with %d labels", kb.Name, classifier.LabelCount())
	}
	if len(classifiers) == 0 {
		return nil
	}
	b.options = append(b.options, withKBClassifiers(classifiers))
	return nil
}

func (c *Classifier) logHeuristicClassifierInitialization() {
	if c.contextClassifier != nil {
		logging.Infof("Context classifier initialized with %d rules", len(c.contextClassifier.rules))
	}
	if c.structureClassifier != nil {
		logging.Infof("Structure classifier initialized with %d rules", len(c.structureClassifier.rules))
	}
}

func (c *Classifier) initializeBestEffortRuntimeClassifiers() {
	steps := []struct {
		enabled bool
		label   string
		init    func() error
	}{
		{enabled: c.IsFactCheckEnabled(), label: "fact-check classifier", init: c.initializeFactCheckClassifier},
		{enabled: c.IsHallucinationDetectionEnabled(), label: "hallucination detector", init: c.initializeHallucinationDetector},
		{enabled: c.IsFeedbackDetectorEnabled(), label: "feedback detector", init: c.initializeFeedbackDetector},
		{enabled: c.IsPreferenceClassifierEnabled(), label: "preference classifier", init: c.initializePreferenceClassifier},
		{enabled: len(c.Config.LanguageRules) > 0, label: "language classifier", init: c.initializeLanguageClassifier},
	}
	for _, step := range steps {
		if !step.enabled {
			continue
		}
		if err := step.init(); err != nil {
			logging.Warnf("Failed to initialize %s: %v", step.label, err)
		}
	}
}
