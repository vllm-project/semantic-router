package classification

import (
	"fmt"
	"strings"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// Classifier handles text classification, model selection, and jailbreak detection functionality
type Classifier struct {
	// Dependencies - In-tree classifiers
	categoryInitializer         CategoryInitializer
	categoryInference           CategoryInference
	jailbreakInitializer        JailbreakInitializer
	jailbreakInference          JailbreakInference
	piiInitializer              PIIInitializer
	piiInference                PIIInference
	keywordClassifier           *KeywordClassifier
	keywordEmbeddingInitializer EmbeddingClassifierInitializer
	keywordEmbeddingClassifier  *EmbeddingClassifier

	// Dependencies - MCP-based classifiers
	mcpCategoryInitializer MCPCategoryInitializer
	mcpCategoryInference   MCPCategoryInference

	// Hallucination mitigation classifiers
	factCheckClassifier   *FactCheckClassifier
	hallucinationDetector *HallucinationDetector
	feedbackDetector      *FeedbackDetector

	// Preference classifier for route matching via external LLM
	preferenceClassifier *PreferenceClassifier

	// Language classifier
	languageClassifier *LanguageClassifier

	// Context classifier for token count-based routing
	contextClassifier *ContextClassifier

	// Complexity classifier for complexity-based routing using embedding similarity
	complexityClassifier *ComplexityClassifier

	// Contrastive jailbreak classifiers keyed by rule name.
	// Only populated for JailbreakRules with Method == "contrastive".
	contrastiveJailbreakClassifiers map[string]*ContrastiveJailbreakClassifier

	// Authz classifier for user-level authorization signal classification
	authzClassifier *AuthzClassifier

	// Identity header names resolved from authz.identity config (or defaults).
	// Used by EvaluateAllSignalsWithHeaders to read user identity from requests.
	authzUserIDHeader     string
	authzUserGroupsHeader string

	Config           *config.RouterConfig
	CategoryMapping  *CategoryMapping
	PIIMapping       *PIIMapping
	JailbreakMapping *JailbreakMapping

	// Category name mapping layer to support generic categories in config
	// Maps MMLU-Pro category names -> generic category names (as defined in config.Categories)
	MMLUToGeneric map[string]string
	// Maps generic category names -> MMLU-Pro category names
	GenericToMMLU map[string][]string
}

type option func(*Classifier)

func withCategory(categoryMapping *CategoryMapping, categoryInitializer CategoryInitializer, categoryInference CategoryInference) option {
	return func(c *Classifier) {
		c.CategoryMapping = categoryMapping
		c.categoryInitializer = categoryInitializer
		c.categoryInference = categoryInference
	}
}

func withJailbreak(jailbreakMapping *JailbreakMapping, jailbreakInitializer JailbreakInitializer, jailbreakInference JailbreakInference) option {
	return func(c *Classifier) {
		c.JailbreakMapping = jailbreakMapping
		c.jailbreakInitializer = jailbreakInitializer
		c.jailbreakInference = jailbreakInference
	}
}

func withPII(piiMapping *PIIMapping, piiInitializer PIIInitializer, piiInference PIIInference) option {
	return func(c *Classifier) {
		c.PIIMapping = piiMapping
		c.piiInitializer = piiInitializer
		c.piiInference = piiInference
	}
}

func withKeywordClassifier(keywordClassifier *KeywordClassifier) option {
	return func(c *Classifier) {
		c.keywordClassifier = keywordClassifier
	}
}

func withKeywordEmbeddingClassifier(keywordEmbeddingInitializer EmbeddingClassifierInitializer, keywordEmbeddingClassifier *EmbeddingClassifier) option {
	return func(c *Classifier) {
		c.keywordEmbeddingInitializer = keywordEmbeddingInitializer
		c.keywordEmbeddingClassifier = keywordEmbeddingClassifier
	}
}

func withContextClassifier(contextClassifier *ContextClassifier) option {
	return func(c *Classifier) {
		c.contextClassifier = contextClassifier
	}
}

func withComplexityClassifier(complexityClassifier *ComplexityClassifier) option {
	return func(c *Classifier) {
		c.complexityClassifier = complexityClassifier
	}
}

func withContrastiveJailbreakClassifiers(classifiers map[string]*ContrastiveJailbreakClassifier) option {
	return func(c *Classifier) {
		c.contrastiveJailbreakClassifiers = classifiers
	}
}

func withAuthzClassifier(authzClassifier *AuthzClassifier) option {
	return func(c *Classifier) {
		c.authzClassifier = authzClassifier
	}
}

// initModels initializes the models for the classifier
func initModels(classifier *Classifier) (*Classifier, error) {
	// Initialize either in-tree OR MCP-based category classifier
	if classifier.IsCategoryEnabled() {
		if err := classifier.initializeCategoryClassifier(); err != nil {
			return nil, err
		}
	} else if classifier.IsMCPCategoryEnabled() {
		if err := classifier.initializeMCPCategoryClassifier(); err != nil {
			return nil, err
		}
	}

	if classifier.IsJailbreakEnabled() {
		if err := classifier.initializeJailbreakClassifier(); err != nil {
			return nil, err
		}
	}

	if classifier.IsPIIEnabled() {
		if err := classifier.initializePIIClassifier(); err != nil {
			return nil, err
		}
	}

	if classifier.IsKeywordEmbeddingClassifierEnabled() {
		if err := classifier.initializeKeywordEmbeddingClassifier(); err != nil {
			return nil, err
		}
	}

	// Initialize context classifier (no external model init needed, but good to log)
	if classifier.contextClassifier != nil {
		logging.Infof("Context classifier initialized with %d rules", len(classifier.contextClassifier.rules))
	}

	// Initialize hallucination mitigation classifiers
	if classifier.IsFactCheckEnabled() {
		if err := classifier.initializeFactCheckClassifier(); err != nil {
			logging.Warnf("Failed to initialize fact-check classifier: %v", err)
			// Non-fatal - continue without fact-check
		}
	}

	if classifier.IsHallucinationDetectionEnabled() {
		if err := classifier.initializeHallucinationDetector(); err != nil {
			logging.Warnf("Failed to initialize hallucination detector: %v", err)
			// Non-fatal - continue without hallucination detection
		}
	}

	if classifier.IsFeedbackDetectorEnabled() {
		if err := classifier.initializeFeedbackDetector(); err != nil {
			logging.Warnf("Failed to initialize feedback detector: %v", err)
			// Non-fatal - continue without feedback detection
		}
	}

	if classifier.IsPreferenceClassifierEnabled() {
		if err := classifier.initializePreferenceClassifier(); err != nil {
			logging.Warnf("Failed to initialize preference classifier: %v", err)
			// Non-fatal - continue without preference classification
		}
	}

	// Initialize language classifier
	if len(classifier.Config.LanguageRules) > 0 {
		if err := classifier.initializeLanguageClassifier(); err != nil {
			logging.Warnf("Failed to initialize language classifier: %v", err)
			// Non-fatal - continue without language classification
		}
	}

	return classifier, nil
}

// newClassifierWithOptions creates a new classifier with the given options
func newClassifierWithOptions(cfg *config.RouterConfig, options ...option) (*Classifier, error) {
	if cfg == nil {
		return nil, fmt.Errorf("config is nil")
	}

	classifier := &Classifier{Config: cfg}

	// Resolve identity header names from authz.identity config (or defaults).
	classifier.authzUserIDHeader = cfg.Authz.Identity.GetUserIDHeader()
	classifier.authzUserGroupsHeader = cfg.Authz.Identity.GetUserGroupsHeader()

	for _, option := range options {
		option(classifier)
	}

	// Build category name mappings to support generic categories in config
	classifier.buildCategoryNameMappings()

	return initModels(classifier)
}

// NewClassifier creates a new classifier with model selection and jailbreak/PII detection capabilities.
// Both in-tree and MCP classifiers can be configured simultaneously for category classification.
// At runtime, in-tree classifier will be tried first, with MCP as a fallback,
// allowing flexible deployment scenarios such as gradual migration.
func NewClassifier(cfg *config.RouterConfig, categoryMapping *CategoryMapping, piiMapping *PIIMapping, jailbreakMapping *JailbreakMapping) (*Classifier, error) {
	// Create jailbreak inference (vLLM or Candle)
	// Pass full RouterConfig to allow lookup of external models
	jailbreakInference, err := createJailbreakInference(&cfg.PromptGuard, cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create jailbreak inference: %w", err)
	}

	// Create jailbreak initializer (only needed for Candle, nil for vLLM)
	var jailbreakInitializer JailbreakInitializer
	if !cfg.PromptGuard.UseVLLM {
		if cfg.PromptGuard.UseMmBERT32K {
			jailbreakInitializer = createMmBERT32KJailbreakInitializer()
		} else {
			jailbreakInitializer = createJailbreakInitializer()
		}
	}

	// Create PII initializer and inference based on config
	var piiInitializer PIIInitializer
	var piiInference PIIInference
	if cfg.PIIModel.UseMmBERT32K {
		logging.Infof("Using mmBERT-32K for PII detection (32K context, YaRN RoPE)")
		piiInitializer = createMmBERT32KPIIInitializer()
		piiInference = createMmBERT32KPIIInference()
	} else {
		piiInitializer = createPIIInitializer()
		piiInference = createPIIInference()
	}

	options := []option{
		withJailbreak(jailbreakMapping, jailbreakInitializer, jailbreakInference),
		withPII(piiMapping, piiInitializer, piiInference),
	}

	multiModalInitialized := false
	initMultiModalIfNeeded := func(reason string) error {
		if multiModalInitialized {
			return nil
		}
		mmPath := config.ResolveModelPath(cfg.EmbeddingModels.MultiModalModelPath)
		if mmPath == "" {
			return fmt.Errorf("%s requires embedding_models.multimodal_model_path to be set", reason)
		}
		if err := initMultiModalModel(mmPath, cfg.EmbeddingModels.UseCPU); err != nil {
			return fmt.Errorf("failed to initialize multimodal model for %s: %w", reason, err)
		}
		logging.Infof("Initialized multimodal embedding model for %s: %s", reason, mmPath)
		multiModalInitialized = true
		return nil
	}

	// Add keyword classifier if configured
	if len(cfg.KeywordRules) > 0 {
		keywordClassifier, err := NewKeywordClassifier(cfg.KeywordRules)
		if err != nil {
			logging.Errorf("Failed to create keyword classifier: %v", err)
			return nil, err
		}
		options = append(options, withKeywordClassifier(keywordClassifier))
	}

	// Add keyword embedding classifier if configured
	if len(cfg.EmbeddingRules) > 0 {
		// Get optimization config from embedding models configuration
		optConfig := cfg.EmbeddingConfig
		if strings.EqualFold(strings.TrimSpace(optConfig.ModelType), "multimodal") {
			if err := initMultiModalIfNeeded("embedding_rules with model_type=multimodal"); err != nil {
				return nil, err
			}
		}
		keywordEmbeddingClassifier, err := NewEmbeddingClassifier(cfg.EmbeddingRules, optConfig)
		if err != nil {
			logging.Errorf("Failed to create keyword embedding classifier: %v", err)
			return nil, err
		}
		options = append(options, withKeywordEmbeddingClassifier(createEmbeddingInitializer(), keywordEmbeddingClassifier))
	}

	// Add context classifier if configured
	if len(cfg.ContextRules) > 0 {
		// Create token counter (uses character-based heuristic for performance)
		tokenCounter := &CharacterBasedTokenCounter{}
		contextClassifier := NewContextClassifier(tokenCounter, cfg.ContextRules)
		options = append(options, withContextClassifier(contextClassifier))
	}

	// Add complexity classifier if configured
	if len(cfg.ComplexityRules) > 0 {
		// Get model type from embedding models configuration (reuse same model as embedding classifier)
		modelType := cfg.EmbeddingConfig.ModelType
		if modelType == "" {
			modelType = "qwen3" // Default to qwen3
		}

		// Initialize multimodal model if any complexity rule uses image candidates
		if config.HasImageCandidatesInRules(cfg.ComplexityRules) {
			if err := initMultiModalIfNeeded("complexity image_candidates"); err != nil {
				return nil, err
			}
		}

		if strings.EqualFold(strings.TrimSpace(modelType), "multimodal") {
			if err := initMultiModalIfNeeded("complexity model_type=multimodal"); err != nil {
				return nil, err
			}
		}

		complexityClassifier, err := NewComplexityClassifier(cfg.ComplexityRules, modelType)
		if err != nil {
			logging.Errorf("Failed to create complexity classifier: %v", err)
			return nil, err
		}
		options = append(options, withComplexityClassifier(complexityClassifier))
	}

	// Add contrastive jailbreak classifiers for rules with method == "contrastive"
	{
		contrastiveClassifiers := make(map[string]*ContrastiveJailbreakClassifier)
		defaultModelType := cfg.EmbeddingConfig.ModelType
		for _, rule := range cfg.JailbreakRules {
			if rule.Method != "contrastive" {
				continue
			}
			if strings.EqualFold(strings.TrimSpace(defaultModelType), "multimodal") {
				if err := initMultiModalIfNeeded("contrastive jailbreak with model_type=multimodal"); err != nil {
					return nil, err
				}
			}
			cjc, err := NewContrastiveJailbreakClassifier(rule, defaultModelType)
			if err != nil {
				logging.Errorf("Failed to create contrastive jailbreak classifier for rule %q: %v", rule.Name, err)
				return nil, err
			}
			contrastiveClassifiers[rule.Name] = cjc
		}
		if len(contrastiveClassifiers) > 0 {
			options = append(options, withContrastiveJailbreakClassifiers(contrastiveClassifiers))
			logging.Infof("Initialized %d contrastive jailbreak classifiers", len(contrastiveClassifiers))
		}
	}

	// Add authz classifier if authz rules are configured
	roleBindings := cfg.GetRoleBindings()
	if len(roleBindings) > 0 {
		authzClassifier, err := NewAuthzClassifier(roleBindings)
		if err != nil {
			return nil, fmt.Errorf("failed to create authz classifier: %w", err)
		}
		options = append(options, withAuthzClassifier(authzClassifier))
		logging.Infof("Authz classifier initialized with %d role bindings", len(roleBindings))
	}

	// Add in-tree classifier if configured
	if cfg.CategoryModel.ModelID != "" {
		var categoryInitializer CategoryInitializer
		var categoryInference CategoryInference
		if cfg.CategoryModel.UseMmBERT32K {
			logging.Infof("Using mmBERT-32K for intent/category classification (32K context, YaRN RoPE)")
			categoryInitializer = createMmBERT32KCategoryInitializer()
			categoryInference = createMmBERT32KCategoryInference()
		} else {
			categoryInitializer = createCategoryInitializer()
			categoryInference = createCategoryInference()
		}
		options = append(options, withCategory(categoryMapping, categoryInitializer, categoryInference))
	}

	// Add MCP classifier if configured
	// Note: Both in-tree and MCP classifiers can be configured simultaneously.
	// At runtime, in-tree classifier will be tried first, with MCP as a fallback.
	// This allows flexible deployment scenarios (e.g., gradual migration, A/B testing).
	if cfg.MCPCategoryModel.Enabled {
		mcpInit := createMCPCategoryInitializer()
		mcpInf := createMCPCategoryInference(mcpInit)
		options = append(options, withMCPCategory(mcpInit, mcpInf))
	}

	return newClassifierWithOptions(cfg, options...)
}

// IsCategoryEnabled checks if category classification is properly configured
func (c *Classifier) IsCategoryEnabled() bool {
	return c.Config.CategoryModel.ModelID != "" && c.Config.CategoryMappingPath != "" && c.CategoryMapping != nil
}

// initializeCategoryClassifier initializes the category classification model
func (c *Classifier) initializeCategoryClassifier() error {
	if !c.IsCategoryEnabled() || c.categoryInitializer == nil {
		return fmt.Errorf("category classification is not properly configured")
	}

	numClasses := c.CategoryMapping.GetCategoryCount()
	if numClasses < 2 {
		return fmt.Errorf("not enough categories for classification, need at least 2, got %d", numClasses)
	}

	logging.Infof("🔧 Initializing Intent/Category Classifier:")
	logging.Infof("Model: %s", c.Config.CategoryModel.ModelID)
	logging.Infof("Mapping: %s", c.Config.CategoryMappingPath)
	logging.Infof("Classes: %d", numClasses)
	logging.Infof("CPU Mode: %v", c.Config.CategoryModel.UseCPU)

	return c.categoryInitializer.Init(c.Config.CategoryModel.ModelID, c.Config.CategoryModel.UseCPU, numClasses)
}

// IsJailbreakEnabled checks if jailbreak detection is enabled and properly configured
func (c *Classifier) IsJailbreakEnabled() bool {
	if !c.Config.PromptGuard.Enabled || c.JailbreakMapping == nil {
		return false
	}

	// Check configuration based on whether using vLLM or Candle
	if c.Config.PromptGuard.UseVLLM {
		// For vLLM: check if external guardrail model is configured
		externalCfg := c.Config.FindExternalModelByRole(config.ModelRoleGuardrail)
		hasExternalConfig := externalCfg != nil &&
			externalCfg.ModelEndpoint.Address != "" &&
			externalCfg.ModelName != ""

		// Need mapping path and external config
		return c.Config.PromptGuard.JailbreakMappingPath != "" && hasExternalConfig
	}

	// For Candle: need model ID and mapping path
	return c.Config.PromptGuard.ModelID != "" && c.Config.PromptGuard.JailbreakMappingPath != ""
}

// initializeJailbreakClassifier initializes the jailbreak classification model
func (c *Classifier) initializeJailbreakClassifier() error {
	if !c.IsJailbreakEnabled() {
		return fmt.Errorf("jailbreak detection is not properly configured")
	}

	// Skip initialization if using vLLM (no Candle model to initialize)
	if c.Config.PromptGuard.UseVLLM {
		externalCfg := c.Config.FindExternalModelByRole(config.ModelRoleGuardrail)
		logging.Infof("Initializing Jailbreak Detector (vLLM mode):")
		if externalCfg != nil {
			logging.Infof("External Model: %s", externalCfg.ModelName)
			logging.Infof("Endpoint: %s", externalCfg.ModelEndpoint.Address)
		}
		logging.Infof("Mapping: %s", c.Config.PromptGuard.JailbreakMappingPath)
		logging.Infof("Using vLLM for jailbreak detection, skipping Candle initialization")
		return nil
	}

	// For Candle-based inference, need initializer
	if c.jailbreakInitializer == nil {
		return fmt.Errorf("jailbreak initializer is required for Candle-based inference")
	}

	numClasses := c.JailbreakMapping.GetJailbreakTypeCount()
	if numClasses < 2 {
		return fmt.Errorf("not enough jailbreak types for classification, need at least 2, got %d", numClasses)
	}

	logging.Infof("Initializing Jailbreak Detector:")
	logging.Infof("Model: %s", c.Config.PromptGuard.ModelID)
	logging.Infof("Mapping: %s", c.Config.PromptGuard.JailbreakMappingPath)
	logging.Infof("Classes: %d", numClasses)
	logging.Infof("CPU Mode: %v", c.Config.PromptGuard.UseCPU)

	return c.jailbreakInitializer.Init(c.Config.PromptGuard.ModelID, c.Config.PromptGuard.UseCPU, numClasses)
}

// CheckForJailbreak analyzes the given text for jailbreak attempts
func (c *Classifier) CheckForJailbreak(text string) (bool, string, float32, error) {
	return c.CheckForJailbreakWithThreshold(text, c.Config.PromptGuard.Threshold)
}

// CheckForJailbreakWithThreshold analyzes the given text for jailbreak attempts with a custom threshold
func (c *Classifier) CheckForJailbreakWithThreshold(text string, threshold float32) (bool, string, float32, error) {
	if !c.IsJailbreakEnabled() {
		return false, "", 0.0, fmt.Errorf("jailbreak detection is not enabled or properly configured")
	}

	if text == "" {
		return false, "", 0.0, nil
	}

	// Use appropriate jailbreak classifier based on configuration
	var result candle_binding.ClassResult
	var err error

	result, err = c.jailbreakInference.Classify(text)
	if err != nil {
		return false, "", 0.0, fmt.Errorf("jailbreak classification failed: %w", err)
	}
	logging.Infof("Jailbreak classification result: %v", result)

	// Get the jailbreak type name from the class index
	jailbreakType, ok := c.JailbreakMapping.GetJailbreakTypeFromIndex(result.Class)
	if !ok {
		return false, "", 0.0, fmt.Errorf("unknown jailbreak class index: %d", result.Class)
	}

	// Check if confidence meets threshold and indicates jailbreak
	isJailbreak := result.Confidence >= threshold && jailbreakType == "jailbreak"

	if isJailbreak {
		logging.Warnf("JAILBREAK DETECTED: '%s' (confidence: %.3f, threshold: %.3f)",
			jailbreakType, result.Confidence, threshold)
	} else {
		logging.Infof("BENIGN: '%s' (confidence: %.3f, threshold: %.3f)",
			jailbreakType, result.Confidence, threshold)
	}

	return isJailbreak, jailbreakType, result.Confidence, nil
}

// AnalyzeContentForJailbreak analyzes multiple content pieces for jailbreak attempts
func (c *Classifier) AnalyzeContentForJailbreak(contentList []string) (bool, []JailbreakDetection, error) {
	return c.AnalyzeContentForJailbreakWithThreshold(contentList, c.Config.PromptGuard.Threshold)
}

// AnalyzeContentForJailbreakWithThreshold analyzes multiple content pieces for jailbreak attempts with a custom threshold
func (c *Classifier) AnalyzeContentForJailbreakWithThreshold(contentList []string, threshold float32) (bool, []JailbreakDetection, error) {
	if !c.IsJailbreakEnabled() {
		return false, nil, fmt.Errorf("jailbreak detection is not enabled or properly configured")
	}

	var detections []JailbreakDetection
	hasJailbreak := false

	for i, content := range contentList {
		if content == "" {
			continue
		}

		isJailbreak, jailbreakType, confidence, err := c.CheckForJailbreakWithThreshold(content, threshold)
		if err != nil {
			logging.Errorf("Error analyzing content %d: %v", i, err)
			continue
		}

		detection := JailbreakDetection{
			Content:       content,
			IsJailbreak:   isJailbreak,
			JailbreakType: jailbreakType,
			Confidence:    confidence,
			ContentIndex:  i,
		}

		detections = append(detections, detection)

		if isJailbreak {
			hasJailbreak = true
		}
	}

	return hasJailbreak, detections, nil
}

// IsPIIEnabled checks if PII detection is properly configured
func (c *Classifier) IsPIIEnabled() bool {
	return c.Config.PIIModel.ModelID != "" && c.Config.PIIMappingPath != "" && c.PIIMapping != nil
}

// initializePIIClassifier initializes the PII token classification model
func (c *Classifier) initializePIIClassifier() error {
	if !c.IsPIIEnabled() || c.piiInitializer == nil {
		return fmt.Errorf("PII detection is not properly configured")
	}

	numPIIClasses := c.PIIMapping.GetPIITypeCount()
	if numPIIClasses < 2 {
		return fmt.Errorf("not enough PII types for classification, need at least 2, got %d", numPIIClasses)
	}

	logging.Infof("Initializing PII Detector:")
	logging.Infof("Model: %s", c.Config.PIIModel.ModelID)
	logging.Infof("Mapping: %s", c.Config.PIIMappingPath)
	logging.Infof("Classes: %d", numPIIClasses)
	logging.Infof("CPU Mode: %v", c.Config.PIIModel.UseCPU)

	// Pass numClasses to support auto-detection
	return c.piiInitializer.Init(c.Config.PIIModel.ModelID, c.Config.PIIModel.UseCPU, numPIIClasses)
}
