package classification

import (
	"fmt"
	"slices"
	"strings"
	"sync"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

type CategoryInitializer interface {
	Init(modelID string, useCPU bool, numClasses ...int) error
}

type CategoryInitializerImpl struct {
	usedModernBERT bool // Track which init path succeeded for inference routing
}

func (c *CategoryInitializerImpl) Init(modelID string, useCPU bool, numClasses ...int) error {
	// Try auto-detecting Candle BERT init first - checks for lora_config.json
	// This enables LoRA Intent/Category models when available
	success := candle_binding.InitCandleBertClassifier(modelID, numClasses[0], useCPU)
	if success {
		c.usedModernBERT = false
		logging.Infof("Initialized category classifier with auto-detection")
		return nil
	}

	// Fallback to ModernBERT-specific init for backward compatibility
	// This handles models with incomplete configs (missing hidden_act, etc.)
	logging.Infof("Auto-detection failed, falling back to ModernBERT category initializer")
	err := candle_binding.InitModernBertClassifier(modelID, useCPU)
	if err != nil {
		return fmt.Errorf("failed to initialize category classifier (both auto-detect and ModernBERT): %w", err)
	}
	c.usedModernBERT = true
	logging.Infof("Initialized ModernBERT category classifier (fallback mode)")
	return nil
}

// MmBERT32KCategoryInitializerImpl uses mmBERT-32K (YaRN RoPE, 32K context) for intent classification
type MmBERT32KCategoryInitializerImpl struct {
	usedMmBERT32K bool
}

func (c *MmBERT32KCategoryInitializerImpl) Init(modelID string, useCPU bool, numClasses ...int) error {
	logging.Infof("Initializing mmBERT-32K intent classifier from: %s", modelID)
	err := candle_binding.InitMmBert32KIntentClassifier(modelID, useCPU)
	if err != nil {
		return fmt.Errorf("failed to initialize mmBERT-32K intent classifier: %w", err)
	}
	c.usedMmBERT32K = true
	logging.Infof("Initialized mmBERT-32K intent classifier (32K context, YaRN RoPE)")
	return nil
}

// createCategoryInitializer creates the category initializer (auto-detecting)
func createCategoryInitializer() CategoryInitializer {
	return &CategoryInitializerImpl{}
}

// createMmBERT32KCategoryInitializer creates an mmBERT-32K category initializer
func createMmBERT32KCategoryInitializer() CategoryInitializer {
	return &MmBERT32KCategoryInitializerImpl{}
}

type CategoryInference interface {
	Classify(text string) (candle_binding.ClassResult, error)
	ClassifyWithProbabilities(text string) (candle_binding.ClassResultWithProbs, error)
}

type CategoryInferenceImpl struct{}

func (c *CategoryInferenceImpl) Classify(text string) (candle_binding.ClassResult, error) {
	// Try Candle BERT first, fall back to ModernBERT if it fails
	result, err := candle_binding.ClassifyCandleBertText(text)
	if err != nil {
		// Candle BERT not initialized or failed, try ModernBERT
		return candle_binding.ClassifyModernBertText(text)
	}
	return result, nil
}

func (c *CategoryInferenceImpl) ClassifyWithProbabilities(text string) (candle_binding.ClassResultWithProbs, error) {
	// Note: CandleBert doesn't have WithProbabilities yet, fall back to ModernBERT
	// This will work correctly if ModernBERT was initialized as fallback
	return candle_binding.ClassifyModernBertTextWithProbabilities(text)
}

// createCategoryInference creates the category inference (auto-detecting)
func createCategoryInference() CategoryInference {
	return &CategoryInferenceImpl{}
}

// MmBERT32KCategoryInferenceImpl uses mmBERT-32K for intent classification
type MmBERT32KCategoryInferenceImpl struct{}

func (c *MmBERT32KCategoryInferenceImpl) Classify(text string) (candle_binding.ClassResult, error) {
	return candle_binding.ClassifyMmBert32KIntent(text)
}

func (c *MmBERT32KCategoryInferenceImpl) ClassifyWithProbabilities(text string) (candle_binding.ClassResultWithProbs, error) {
	// mmBERT-32K doesn't have WithProbabilities yet, use basic classification
	result, err := candle_binding.ClassifyMmBert32KIntent(text)
	if err != nil {
		return candle_binding.ClassResultWithProbs{}, err
	}
	return candle_binding.ClassResultWithProbs{
		Class:      result.Class,
		Confidence: result.Confidence,
	}, nil
}

// createMmBERT32KCategoryInference creates mmBERT-32K category inference
func createMmBERT32KCategoryInference() CategoryInference {
	return &MmBERT32KCategoryInferenceImpl{}
}

type JailbreakInitializer interface {
	Init(modelID string, useCPU bool, numClasses ...int) error
}

type JailbreakInitializerImpl struct {
	usedModernBERT bool // Track which init path succeeded for inference routing
}

func (c *JailbreakInitializerImpl) Init(modelID string, useCPU bool, numClasses ...int) error {
	// Try auto-detecting jailbreak classifier init first - checks for lora_config.json
	// This enables LoRA Jailbreak models when available
	// Use InitJailbreakClassifier which routes to LORA_JAILBREAK_CLASSIFIER or BERT_JAILBREAK_CLASSIFIER
	err := candle_binding.InitJailbreakClassifier(modelID, numClasses[0], useCPU)
	if err == nil {
		c.usedModernBERT = false
		logging.Infof("Initialized jailbreak classifier with auto-detection")
		return nil
	}

	// Fallback to ModernBERT-specific init for backward compatibility
	// This handles models with incomplete configs (missing hidden_act, etc.)
	logging.Infof("Auto-detection failed, falling back to ModernBERT jailbreak initializer")
	err = candle_binding.InitModernBertJailbreakClassifier(modelID, useCPU)
	if err != nil {
		return fmt.Errorf("failed to initialize jailbreak classifier (both auto-detect and ModernBERT): %w", err)
	}
	c.usedModernBERT = true
	logging.Infof("Initialized ModernBERT jailbreak classifier (fallback mode)")
	return nil
}

// createJailbreakInitializer creates the jailbreak initializer (auto-detecting)
func createJailbreakInitializer() JailbreakInitializer {
	return &JailbreakInitializerImpl{}
}

// MmBERT32KJailbreakInitializerImpl uses mmBERT-32K (YaRN RoPE, 32K context) for jailbreak detection
type MmBERT32KJailbreakInitializerImpl struct {
	usedMmBERT32K bool
}

func (c *MmBERT32KJailbreakInitializerImpl) Init(modelID string, useCPU bool, numClasses ...int) error {
	logging.Infof("Initializing mmBERT-32K jailbreak detector from: %s", modelID)
	err := candle_binding.InitMmBert32KJailbreakClassifier(modelID, useCPU)
	if err != nil {
		return fmt.Errorf("failed to initialize mmBERT-32K jailbreak detector: %w", err)
	}
	c.usedMmBERT32K = true
	logging.Infof("Initialized mmBERT-32K jailbreak detector (32K context, YaRN RoPE)")
	return nil
}

// createMmBERT32KJailbreakInitializer creates an mmBERT-32K jailbreak initializer
func createMmBERT32KJailbreakInitializer() JailbreakInitializer {
	return &MmBERT32KJailbreakInitializerImpl{}
}

type JailbreakInference interface {
	Classify(text string) (candle_binding.ClassResult, error)
}

type JailbreakInferenceImpl struct{}

func (c *JailbreakInferenceImpl) Classify(text string) (candle_binding.ClassResult, error) {
	// Try jailbreak-specific classifier first, fall back to ModernBERT if it fails
	result, err := candle_binding.ClassifyJailbreakText(text)
	if err != nil {
		// Jailbreak classifier not initialized or failed, try ModernBERT
		return candle_binding.ClassifyModernBertJailbreakText(text)
	}
	return result, nil
}

// createJailbreakInferenceCandle creates Candle-based jailbreak inference (auto-detecting)
func createJailbreakInferenceCandle() JailbreakInference {
	return &JailbreakInferenceImpl{}
}

// MmBERT32KJailbreakInferenceImpl uses mmBERT-32K for jailbreak detection
type MmBERT32KJailbreakInferenceImpl struct{}

func (c *MmBERT32KJailbreakInferenceImpl) Classify(text string) (candle_binding.ClassResult, error) {
	return candle_binding.ClassifyMmBert32KJailbreak(text)
}

// createMmBERT32KJailbreakInference creates mmBERT-32K jailbreak inference
func createMmBERT32KJailbreakInference() JailbreakInference {
	return &MmBERT32KJailbreakInferenceImpl{}
}

// createJailbreakInference creates the appropriate jailbreak inference based on configuration
// Checks UseMmBERT32K and UseVLLM flags to decide between mmBERT-32K, vLLM, or Candle implementation
// When UseMmBERT32K is true, uses mmBERT-32K (32K context, YaRN RoPE, multilingual)
// When UseVLLM is true, it will try to find external model config with role="guardrail"
func createJailbreakInference(promptGuardCfg *config.PromptGuardConfig, routerCfg *config.RouterConfig) (JailbreakInference, error) {
	// Check for mmBERT-32K first (takes precedence)
	if promptGuardCfg.UseMmBERT32K {
		logging.Infof("Using mmBERT-32K for jailbreak detection (32K context, YaRN RoPE)")
		return createMmBERT32KJailbreakInference(), nil
	}

	if promptGuardCfg.UseVLLM {
		// Try to find external model configuration with role="guardrail"
		externalCfg := routerCfg.FindExternalModelByRole(config.ModelRoleGuardrail)
		if externalCfg == nil {
			return nil, fmt.Errorf("external model with model_role='%s' is required when use_vllm=true", config.ModelRoleGuardrail)
		}

		// Validate required fields
		if externalCfg.ModelEndpoint.Address == "" {
			return nil, fmt.Errorf("external guardrail model endpoint address is required")
		}
		if externalCfg.ModelName == "" {
			return nil, fmt.Errorf("external guardrail model name is required")
		}

		logging.Infof("Found external guardrail model (provider=%s)", externalCfg.Provider)

		// Use vLLM-based inference with external config
		// Pass default threshold from PromptGuardConfig
		return NewVLLMJailbreakInference(externalCfg, promptGuardCfg.Threshold)
	}
	// Use Candle-based inference
	return createJailbreakInferenceCandle(), nil
}

type PIIInitializer interface {
	Init(modelID string, useCPU bool, numClasses int) error
}

type PIIInitializerImpl struct {
	usedModernBERT bool // Track which init path succeeded for inference routing
}

func (c *PIIInitializerImpl) Init(modelID string, useCPU bool, numClasses int) error {
	// Try auto-detecting Candle BERT init first - checks for lora_config.json
	// This enables LoRA PII models when available
	success := candle_binding.InitCandleBertTokenClassifier(modelID, numClasses, useCPU)
	if success {
		c.usedModernBERT = false
		logging.Infof("Initialized PII token classifier with auto-detection")
		return nil
	}

	// Fallback to ModernBERT-specific init for backward compatibility
	// This handles models with incomplete configs (missing hidden_act, etc.)
	logging.Infof("Auto-detection failed, falling back to ModernBERT PII initializer")
	err := candle_binding.InitModernBertPIITokenClassifier(modelID, useCPU)
	if err != nil {
		return fmt.Errorf("failed to initialize PII token classifier (both auto-detect and ModernBERT): %w", err)
	}
	c.usedModernBERT = true
	logging.Infof("Initialized ModernBERT PII token classifier (fallback mode)")
	return nil
}

// createPIIInitializer creates the PII initializer (auto-detecting)
func createPIIInitializer() PIIInitializer {
	return &PIIInitializerImpl{}
}

// MmBERT32KPIIInitializerImpl uses mmBERT-32K (YaRN RoPE, 32K context) for PII detection
type MmBERT32KPIIInitializerImpl struct {
	usedMmBERT32K bool
}

func (c *MmBERT32KPIIInitializerImpl) Init(modelID string, useCPU bool, numClasses int) error {
	logging.Infof("Initializing mmBERT-32K PII detector from: %s", modelID)
	err := candle_binding.InitMmBert32KPIIClassifier(modelID, useCPU)
	if err != nil {
		return fmt.Errorf("failed to initialize mmBERT-32K PII detector: %w", err)
	}
	c.usedMmBERT32K = true
	logging.Infof("Initialized mmBERT-32K PII detector (32K context, YaRN RoPE)")
	return nil
}

// createMmBERT32KPIIInitializer creates an mmBERT-32K PII initializer
func createMmBERT32KPIIInitializer() PIIInitializer {
	return &MmBERT32KPIIInitializerImpl{}
}

type PIIInference interface {
	ClassifyTokens(text string, configPath string) (candle_binding.TokenClassificationResult, error)
}

type PIIInferenceImpl struct{}

func (c *PIIInferenceImpl) ClassifyTokens(text string, configPath string) (candle_binding.TokenClassificationResult, error) {
	// Auto-detecting inference - uses whichever classifier was initialized (LoRA or Traditional)
	return candle_binding.ClassifyCandleBertTokens(text)
}

// createPIIInference creates the PII inference (auto-detecting)
func createPIIInference() PIIInference {
	return &PIIInferenceImpl{}
}

// MmBERT32KPIIInferenceImpl uses mmBERT-32K for PII token classification
type MmBERT32KPIIInferenceImpl struct{}

func (c *MmBERT32KPIIInferenceImpl) ClassifyTokens(text string, configPath string) (candle_binding.TokenClassificationResult, error) {
	entities, err := candle_binding.ClassifyMmBert32KPII(text, configPath)
	if err != nil {
		return candle_binding.TokenClassificationResult{}, err
	}
	return candle_binding.TokenClassificationResult{Entities: entities}, nil
}

// createMmBERT32KPIIInference creates mmBERT-32K PII inference
func createMmBERT32KPIIInference() PIIInference {
	return &MmBERT32KPIIInferenceImpl{}
}

// JailbreakDetection represents the result of jailbreak analysis for a piece of content
type JailbreakDetection struct {
	Content       string  `json:"content"`
	IsJailbreak   bool    `json:"is_jailbreak"`
	JailbreakType string  `json:"jailbreak_type"`
	Confidence    float32 `json:"confidence"`
	ContentIndex  int     `json:"content_index"`
}

// PIIDetection represents detected PII entities in content
type PIIDetection struct {
	EntityType string  `json:"entity_type"` // Type of PII entity (e.g., "PERSON", "EMAIL", "PHONE")
	Start      int     `json:"start"`       // Start character position in original text
	End        int     `json:"end"`         // End character position in original text
	Text       string  `json:"text"`        // Actual entity text
	Confidence float32 `json:"confidence"`  // Confidence score (0.0 to 1.0)
}

// PIIAnalysisResult represents the result of PII analysis for content
type PIIAnalysisResult struct {
	Content      string         `json:"content"`
	HasPII       bool           `json:"has_pii"`
	Entities     []PIIDetection `json:"entities"`
	ContentIndex int            `json:"content_index"`
}

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
		optConfig := cfg.EmbeddingModels.HNSWConfig
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
		modelType := cfg.EmbeddingModels.HNSWConfig.ModelType
		if modelType == "" {
			modelType = "qwen3" // Default to qwen3
		}
		complexityClassifier, err := NewComplexityClassifier(cfg.ComplexityRules, modelType)
		if err != nil {
			logging.Errorf("Failed to create complexity classifier: %v", err)
			return nil, err
		}
		options = append(options, withComplexityClassifier(complexityClassifier))
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
		logging.Infof("🛡️  Initializing Jailbreak Detector (vLLM mode):")
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

	logging.Infof("🛡️  Initializing Jailbreak Detector:")
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

	logging.Infof("🔒 Initializing PII Detector:")
	logging.Infof("Model: %s", c.Config.PIIModel.ModelID)
	logging.Infof("Mapping: %s", c.Config.PIIMappingPath)
	logging.Infof("Classes: %d", numPIIClasses)
	logging.Infof("CPU Mode: %v", c.Config.PIIModel.UseCPU)

	// Pass numClasses to support auto-detection
	return c.piiInitializer.Init(c.Config.PIIModel.ModelID, c.Config.PIIModel.UseCPU, numPIIClasses)
}

// getUsedSignals analyzes all decisions and returns which signals (type:name) are actually used
// This allows us to skip evaluation of unused signals for performance optimization
// Returns a map with keys in format "type:name" (e.g., "keyword:math_keywords")
func (c *Classifier) getUsedSignals() map[string]bool {
	usedSignals := make(map[string]bool)

	// Analyze all decisions to find which signals are referenced
	for _, decision := range c.Config.Decisions {
		c.analyzeRuleCombination(decision.Rules, usedSignals)
	}

	return usedSignals
}

// getAllSignalTypes returns a map containing all configured signal types
// This is used when forceEvaluateAll is true to evaluate all signals regardless of decision usage
func (c *Classifier) getAllSignalTypes() map[string]bool {
	allSignals := make(map[string]bool)

	// Add all configured keyword rules
	for _, rule := range c.Config.KeywordRules {
		key := strings.ToLower(config.SignalTypeKeyword + ":" + rule.Name)
		allSignals[key] = true
	}

	// Add all configured embedding rules
	for _, rule := range c.Config.EmbeddingRules {
		key := strings.ToLower(config.SignalTypeEmbedding + ":" + rule.Name)
		allSignals[key] = true
	}

	// Add all configured domain categories
	for _, category := range c.Config.Categories {
		key := strings.ToLower(config.SignalTypeDomain + ":" + category.Name)
		allSignals[key] = true
	}

	// Add all configured fact-check rules
	for _, rule := range c.Config.FactCheckRules {
		key := strings.ToLower(config.SignalTypeFactCheck + ":" + rule.Name)
		allSignals[key] = true
	}

	// Add all configured user feedback rules
	for _, rule := range c.Config.UserFeedbackRules {
		key := strings.ToLower(config.SignalTypeUserFeedback + ":" + rule.Name)
		allSignals[key] = true
	}

	// Add all configured preference rules
	for _, rule := range c.Config.PreferenceRules {
		key := strings.ToLower(config.SignalTypePreference + ":" + rule.Name)
		allSignals[key] = true
	}

	// Add all configured language rules
	for _, rule := range c.Config.LanguageRules {
		key := strings.ToLower(config.SignalTypeLanguage + ":" + rule.Name)
		allSignals[key] = true
	}

	// Add all configured context rules
	for _, rule := range c.Config.ContextRules {
		key := strings.ToLower(config.SignalTypeContext + ":" + rule.Name)
		allSignals[key] = true
	}

	// Add all configured complexity rules
	for _, rule := range c.Config.ComplexityRules {
		key := strings.ToLower(config.SignalTypeComplexity + ":" + rule.Name)
		allSignals[key] = true
	}

	// Add all configured modality rules
	for _, rule := range c.Config.ModalityRules {
		key := strings.ToLower(config.SignalTypeModality + ":" + rule.Name)
		allSignals[key] = true
	}

	// Add all configured role bindings (authz signal uses the Role name, not binding name)
	for _, rb := range c.Config.GetRoleBindings() {
		key := strings.ToLower(config.SignalTypeAuthz + ":" + rb.Role)
		allSignals[key] = true
	}

	// Add all configured jailbreak rules
	for _, rule := range c.Config.JailbreakRules {
		key := strings.ToLower(config.SignalTypeJailbreak + ":" + rule.Name)
		allSignals[key] = true
	}

	// Add all configured PII rules
	for _, rule := range c.Config.PIIRules {
		key := strings.ToLower(config.SignalTypePII + ":" + rule.Name)
		allSignals[key] = true
	}

	return allSignals
}

// SignalMetrics contains performance and probability metrics for a single signal
type SignalMetrics struct {
	ExecutionTimeMs float64 `json:"execution_time_ms"` // Execution time in milliseconds
	Confidence      float64 `json:"confidence"`        // Confidence score (0.0-1.0), 0 if not applicable
}

// SignalResults contains all evaluated signal results
type SignalResults struct {
	MatchedKeywordRules      []string
	MatchedKeywords          []string // The actual keywords that matched (not rule names)
	MatchedEmbeddingRules    []string
	MatchedDomainRules       []string
	MatchedFactCheckRules    []string // "needs_fact_check" or "no_fact_check_needed"
	MatchedUserFeedbackRules []string // "satisfied", "need_clarification", "wrong_answer", "want_different"
	MatchedPreferenceRules   []string // Route preference names matched via external LLM
	MatchedLanguageRules     []string // Language codes: "en", "es", "zh", "fr", etc.
	MatchedContextRules      []string // Matched context rule names (e.g. "low_token_count")
	TokenCount               int      // Total token count
	MatchedComplexityRules   []string // Matched complexity rules with difficulty level (e.g. "code_complexity:hard")
	MatchedModalityRules     []string // Matched modality: "AR", "DIFFUSION", or "BOTH"
	MatchedAuthzRules        []string // Matched authz role names for user-level RBAC routing
	MatchedJailbreakRules    []string // Matched jailbreak rule names (confidence >= threshold)
	MatchedPIIRules          []string // Matched PII rule names (denied PII types detected)

	// Jailbreak detection metadata (populated when jailbreak signal is evaluated)
	JailbreakDetected   bool    // Whether any jailbreak was detected (across all rules)
	JailbreakType       string  // Type of the detected jailbreak (from highest-confidence detection)
	JailbreakConfidence float32 // Confidence of the detected jailbreak

	// PII detection metadata (populated when PII signal is evaluated)
	PIIDetected bool     // Whether any PII was detected
	PIIEntities []string // Detected PII entity types (e.g., "EMAIL_ADDRESS", "PERSON")

	SignalConfidences map[string]float64 // Real confidence scores per signal, e.g. "embedding:ai" → 0.88

	// Signal metrics (only populated in eval mode)
	Metrics *SignalMetricsCollection
}

// SignalMetricsCollection contains metrics for all signal types
type SignalMetricsCollection struct {
	Keyword      SignalMetrics `json:"keyword"`
	Embedding    SignalMetrics `json:"embedding"`
	Domain       SignalMetrics `json:"domain"`
	FactCheck    SignalMetrics `json:"fact_check"`
	UserFeedback SignalMetrics `json:"user_feedback"`
	Preference   SignalMetrics `json:"preference"`
	Language     SignalMetrics `json:"language"`
	Context      SignalMetrics `json:"context"`
	Complexity   SignalMetrics `json:"complexity"`
	Modality     SignalMetrics `json:"modality"`
	Authz        SignalMetrics `json:"authz"`
	Jailbreak    SignalMetrics `json:"jailbreak"`
	PII          SignalMetrics `json:"pii"`
}

// analyzeRuleCombination recursively traverses a rule tree to collect all referenced signals.
func (c *Classifier) analyzeRuleCombination(node config.RuleNode, usedSignals map[string]bool) {
	if node.IsLeaf() {
		t := strings.ToLower(strings.TrimSpace(node.Type))
		n := strings.ToLower(strings.TrimSpace(node.Name))
		usedSignals[t+":"+n] = true
		return
	}
	for _, child := range node.Conditions {
		c.analyzeRuleCombination(child, usedSignals)
	}
}

// isSignalTypeUsed checks if any signal of the given type is used in decisions
func isSignalTypeUsed(usedSignals map[string]bool, signalType string) bool {
	// Normalize signal type for comparison (all signals are normalized to lowercase)
	normalizedType := strings.ToLower(strings.TrimSpace(signalType))
	prefix := normalizedType + ":"

	for key := range usedSignals {
		// All signal keys are normalized to lowercase, so use case-insensitive comparison
		if strings.HasPrefix(strings.ToLower(strings.TrimSpace(key)), prefix) {
			return true
		}
	}
	return false
}

// EvaluateAllSignals evaluates all signal types and returns SignalResults
// This is the new method that includes fact_check signals
func (c *Classifier) EvaluateAllSignals(text string) *SignalResults {
	// For backward compatibility, use the same text for both evaluation and context counting
	return c.EvaluateAllSignalsWithContext(text, text, nil, false)
}

// EvaluateAllSignalsWithHeaders evaluates all signal types including the authz signal.
// The authz signal reads user identity and groups from request headers (x-authz-user-id,
// x-authz-user-groups) and evaluates role_bindings. Other signals are evaluated via
// EvaluateAllSignalsWithContext as before.
//
// Returns an error if authz evaluation fails (e.g., missing user identity header when
// role_bindings are configured). Errors are NOT swallowed — the caller must handle them.
// This prevents silent bypass of authz policies.
//
// headers: request headers from ext_proc (includes Authorino-injected authz headers)
func (c *Classifier) EvaluateAllSignalsWithHeaders(text string, contextText string, nonUserMessages []string, headers map[string]string, forceEvaluateAll bool) (*SignalResults, error) {
	results := c.EvaluateAllSignalsWithContext(text, contextText, nonUserMessages, forceEvaluateAll)

	// Evaluate authz signal if role bindings are configured and the signal type is used
	usedSignals := c.getUsedSignals()
	if forceEvaluateAll {
		usedSignals = c.getAllSignalTypes()
	}

	if isSignalTypeUsed(usedSignals, config.SignalTypeAuthz) && c.authzClassifier != nil {
		start := time.Now()
		userID := headers[c.authzUserIDHeader]
		userGroups := ParseUserGroups(headers[c.authzUserGroupsHeader])

		authzResult, err := c.authzClassifier.Classify(userID, userGroups)
		elapsed := time.Since(start)
		latencySeconds := elapsed.Seconds()

		// Record metrics
		results.Metrics.Authz.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
		results.Metrics.Authz.Confidence = 1.0 // Rule-based, always 1.0

		if err != nil {
			// Do NOT swallow authz errors — propagate to caller.
			// A missing user identity header when role_bindings are configured is a hard failure,
			// not a signal that "didn't fire." Silent bypass is not allowed.
			logging.Errorf("[Authz Signal] classification failed: %v", err)
			metrics.RecordSignalExtraction(config.SignalTypeAuthz, "error", latencySeconds)
			return nil, fmt.Errorf("authz signal evaluation failed: %w", err)
		}

		for _, ruleName := range authzResult.MatchedRules {
			metrics.RecordSignalExtraction(config.SignalTypeAuthz, ruleName, latencySeconds)
			metrics.RecordSignalMatch(config.SignalTypeAuthz, ruleName)
		}
		results.MatchedAuthzRules = authzResult.MatchedRules

		logging.Infof("[Signal Computation] Authz signal evaluation completed in %v", elapsed)
	} else if !isSignalTypeUsed(usedSignals, config.SignalTypeAuthz) {
		logging.Infof("[Signal Computation] Authz signal not used in any decision, skipping evaluation")
	}

	return results, nil
}

// EvaluateAllSignalsWithForceOption evaluates signals with option to force evaluate all
// forceEvaluateAll: if true, evaluates all configured signals regardless of decision usage
func (c *Classifier) EvaluateAllSignalsWithForceOption(text string, forceEvaluateAll bool) *SignalResults {
	return c.EvaluateAllSignalsWithContext(text, text, nil, forceEvaluateAll)
}

// EvaluateAllSignalsWithContext evaluates all signal types with separate text for context counting
// text: text to use for signal evaluation (usually latest user message)
// contextText: text to use for context token counting (usually all messages combined)
// nonUserMessages: conversation history (non-user messages) for jailbreak/PII with include_history
// forceEvaluateAll: if true, evaluates all configured signals regardless of decision usage (for eval scenarios)
func (c *Classifier) EvaluateAllSignalsWithContext(text string, contextText string, nonUserMessages []string, forceEvaluateAll bool) *SignalResults {
	// Determine which signals (type:name) should be evaluated
	var usedSignals map[string]bool
	if forceEvaluateAll {
		// Eval mode: evaluate all configured signals
		usedSignals = c.getAllSignalTypes()
		logging.Infof("[Signal Computation] Force evaluate all signals mode enabled")
	} else {
		// Normal mode: only evaluate signals used in decisions
		usedSignals = c.getUsedSignals()
	}

	results := &SignalResults{
		Metrics: &SignalMetricsCollection{}, // Always initialize, no overhead
	}

	var wg sync.WaitGroup
	var mu sync.Mutex

	// Evaluate keyword rules in parallel (only if used in decisions)
	if isSignalTypeUsed(usedSignals, config.SignalTypeKeyword) && c.keywordClassifier != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			start := time.Now()
			category, keywords, err := c.keywordClassifier.ClassifyWithKeywords(text)
			elapsed := time.Since(start)
			latencySeconds := elapsed.Seconds()

			// Record signal extraction metrics
			metrics.RecordSignalExtraction(config.SignalTypeKeyword, category, latencySeconds)

			// Record metrics (use microseconds for better precision)
			results.Metrics.Keyword.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
			results.Metrics.Keyword.Confidence = 1.0 // Rule-based, always 1.0

			logging.Infof("[Signal Computation] Keyword signal evaluation completed in %v", elapsed)
			if err != nil {
				logging.Errorf("keyword rule evaluation failed: %v", err)
			} else if category != "" {
				// Record signal match
				metrics.RecordSignalMatch(config.SignalTypeKeyword, category)

				mu.Lock()
				results.MatchedKeywordRules = append(results.MatchedKeywordRules, category)
				results.MatchedKeywords = append(results.MatchedKeywords, keywords...)
				mu.Unlock()
			}
		}()
	} else if !isSignalTypeUsed(usedSignals, config.SignalTypeKeyword) {
		logging.Infof("[Signal Computation] Keyword signal not used in any decision, skipping evaluation")
	}

	// Evaluate embedding rules in parallel (only if used in decisions)
	// Uses ClassifyAll() to return ALL matched rules (not just the single best),
	// enabling AND conditions in the Decision Engine (e.g., embedding:"ai" AND embedding:"programming").
	// Also stores real similarity scores in SignalConfidences for quality-based routing.
	if isSignalTypeUsed(usedSignals, config.SignalTypeEmbedding) && c.keywordEmbeddingClassifier != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			start := time.Now()
			matchedRules, err := c.keywordEmbeddingClassifier.ClassifyAll(text)
			elapsed := time.Since(start)

			// Record metrics
			results.Metrics.Embedding.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0

			logging.Infof("[Signal Computation] Embedding signal evaluation completed in %v", elapsed)
			if err != nil {
				logging.Errorf("embedding rule evaluation failed: %v", err)
			} else if len(matchedRules) > 0 {
				// Record the highest confidence for metrics display
				var bestConfidence float64
				for _, mr := range matchedRules {
					if mr.Score > bestConfidence {
						bestConfidence = mr.Score
					}
				}
				results.Metrics.Embedding.Confidence = bestConfidence

				mu.Lock()
				// Add ALL matched rules (not just the best one)
				for _, mr := range matchedRules {
					// Record signal extraction and match metrics for each matched rule
					metrics.RecordSignalExtraction(config.SignalTypeEmbedding, mr.RuleName, elapsed.Seconds())
					metrics.RecordSignalMatch(config.SignalTypeEmbedding, mr.RuleName)

					// Append rule name to the matched list
					results.MatchedEmbeddingRules = append(results.MatchedEmbeddingRules, mr.RuleName)

					// Store real similarity score for this rule
					// The Decision Engine will use this instead of hardcoded 1.0
					if results.SignalConfidences == nil {
						results.SignalConfidences = make(map[string]float64)
					}
					results.SignalConfidences["embedding:"+mr.RuleName] = mr.Score

					logging.Infof("[Signal Computation] Embedding match: rule=%q, score=%.4f, method=%s",
						mr.RuleName, mr.Score, mr.Method)
				}
				mu.Unlock()
			}
		}()
	} else if !isSignalTypeUsed(usedSignals, config.SignalTypeEmbedding) {
		logging.Infof("[Signal Computation] Embedding signal not used in any decision, skipping evaluation")
	}

	// Evaluate domain rules (category classification) in parallel (only if used in decisions)
	if isSignalTypeUsed(usedSignals, config.SignalTypeDomain) && c.IsCategoryEnabled() && c.categoryInference != nil && c.CategoryMapping != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			start := time.Now()
			result, err := c.categoryInference.Classify(text)
			elapsed := time.Since(start)
			latencySeconds := elapsed.Seconds()

			var categoryName string
			if err == nil {
				// Map class index to category name
				if name, ok := c.CategoryMapping.GetCategoryFromIndex(result.Class); ok {
					categoryName = name
				}
			}

			// Record signal extraction metrics
			metrics.RecordSignalExtraction(config.SignalTypeDomain, categoryName, latencySeconds)

			// Record metrics (use microseconds for better precision)
			results.Metrics.Domain.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
			if categoryName != "" && err == nil {
				results.Metrics.Domain.Confidence = float64(result.Confidence)
			}

			logging.Infof("[Signal Computation] Domain signal evaluation completed in %v", elapsed)
			if err != nil {
				logging.Errorf("domain rule evaluation failed: %v", err)
			} else if result.Confidence >= c.Config.CategoryModel.Threshold {
				// Only add domain if confidence meets threshold
				// Without this check, low-confidence misclassifications can still match decisions,
				// causing incorrect routing for typo-laden text
				if categoryName != "" {
					// Record signal match
					metrics.RecordSignalMatch(config.SignalTypeDomain, categoryName)

					mu.Lock()
					results.MatchedDomainRules = append(results.MatchedDomainRules, categoryName)
					mu.Unlock()
				}
			}
		}()
	} else if !isSignalTypeUsed(usedSignals, config.SignalTypeDomain) {
		logging.Infof("[Signal Computation] Domain signal not used in any decision, skipping evaluation")
	}

	// Evaluate fact-check rules in parallel (only if used in decisions)
	// Only evaluate if fact_check_rules are configured and fact-check classifier is enabled
	if isSignalTypeUsed(usedSignals, config.SignalTypeFactCheck) && len(c.Config.FactCheckRules) > 0 && c.IsFactCheckEnabled() {
		wg.Add(1)
		go func() {
			defer wg.Done()
			start := time.Now()
			factCheckResult, err := c.ClassifyFactCheck(text)
			elapsed := time.Since(start)
			latencySeconds := elapsed.Seconds()

			// Determine which signal to output based on classification result
			signalName := "no_fact_check_needed"
			if err == nil && factCheckResult != nil && factCheckResult.NeedsFactCheck {
				signalName = "needs_fact_check"
			}

			// Record signal extraction metrics
			metrics.RecordSignalExtraction(config.SignalTypeFactCheck, signalName, latencySeconds)

			// Record metrics (use microseconds for better precision)
			results.Metrics.FactCheck.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
			if signalName != "" && err == nil && factCheckResult != nil {
				results.Metrics.FactCheck.Confidence = float64(factCheckResult.Confidence)
			}

			logging.Infof("[Signal Computation] Fact-check signal evaluation completed in %v", elapsed)
			if err != nil {
				logging.Errorf("fact-check rule evaluation failed: %v", err)
			} else if factCheckResult != nil {
				// Check if this signal is defined in fact_check_rules
				for _, rule := range c.Config.FactCheckRules {
					if rule.Name == signalName {
						// Record signal match
						metrics.RecordSignalMatch(config.SignalTypeFactCheck, rule.Name)

						mu.Lock()
						results.MatchedFactCheckRules = append(results.MatchedFactCheckRules, rule.Name)
						mu.Unlock()
						break
					}
				}
			}
		}()
	} else if !isSignalTypeUsed(usedSignals, config.SignalTypeFactCheck) {
		logging.Infof("[Signal Computation] Fact-check signal not used in any decision, skipping evaluation")
	}

	// Evaluate user feedback rules in parallel (only if used in decisions)
	// Only evaluate if user_feedback_rules are configured and feedback detector is enabled
	if isSignalTypeUsed(usedSignals, config.SignalTypeUserFeedback) && len(c.Config.UserFeedbackRules) > 0 && c.IsFeedbackDetectorEnabled() {
		wg.Add(1)
		go func() {
			defer wg.Done()
			start := time.Now()
			feedbackResult, err := c.ClassifyFeedback(text)
			elapsed := time.Since(start)
			latencySeconds := elapsed.Seconds()

			// Use the feedback type directly as the signal name
			signalName := ""
			if err == nil && feedbackResult != nil {
				signalName = feedbackResult.FeedbackType
			}

			// Record signal extraction metrics
			metrics.RecordSignalExtraction(config.SignalTypeUserFeedback, signalName, latencySeconds)

			// Record metrics (use microseconds for better precision)
			results.Metrics.UserFeedback.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
			if signalName != "" && err == nil && feedbackResult != nil {
				results.Metrics.UserFeedback.Confidence = float64(feedbackResult.Confidence)
			}

			logging.Infof("[Signal Computation] User feedback signal evaluation completed in %v", elapsed)
			if err != nil {
				logging.Errorf("user feedback rule evaluation failed: %v", err)
			} else if feedbackResult != nil {
				// Check if this signal is defined in user_feedback_rules
				for _, rule := range c.Config.UserFeedbackRules {
					if rule.Name == signalName {
						// Record signal match
						metrics.RecordSignalMatch(config.SignalTypeUserFeedback, rule.Name)

						mu.Lock()
						results.MatchedUserFeedbackRules = append(results.MatchedUserFeedbackRules, rule.Name)
						mu.Unlock()
						break
					}
				}
			}
		}()
	} else if !isSignalTypeUsed(usedSignals, config.SignalTypeUserFeedback) {
		logging.Infof("[Signal Computation] User feedback signal not used in any decision, skipping evaluation")
	}

	// Evaluate preference rules in parallel (only if used in decisions)
	// Only evaluate if preference_rules are configured and preference classifier is enabled
	if isSignalTypeUsed(usedSignals, config.SignalTypePreference) && len(c.Config.PreferenceRules) > 0 && c.IsPreferenceClassifierEnabled() {
		wg.Add(1)
		go func() {
			defer wg.Done()
			start := time.Now()
			// Build conversation JSON from text (simple single-turn format)
			conversationJSON := fmt.Sprintf(`[{"role":"user","content":"%s"}]`, text)

			preferenceResult, err := c.preferenceClassifier.Classify(conversationJSON)
			elapsed := time.Since(start)
			latencySeconds := elapsed.Seconds()

			// Use the preference name directly as the signal name
			preferenceName := ""
			if err == nil && preferenceResult != nil {
				preferenceName = preferenceResult.Preference
			}

			// Record signal extraction metrics
			metrics.RecordSignalExtraction(config.SignalTypePreference, preferenceName, latencySeconds)

			// Record metrics (use microseconds for better precision)
			results.Metrics.Preference.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
			if preferenceName != "" && err == nil && preferenceResult != nil && preferenceResult.Confidence > 0 {
				results.Metrics.Preference.Confidence = float64(preferenceResult.Confidence)
			}

			logging.Infof("[Signal Computation] Preference signal evaluation completed in %v", elapsed)
			if err != nil {
				logging.Errorf("preference rule evaluation failed: %v", err)
			} else if preferenceResult != nil {
				// Check if this preference is defined in preference_rules
				for _, rule := range c.Config.PreferenceRules {
					if rule.Name == preferenceName {
						// Record signal match
						metrics.RecordSignalMatch(config.SignalTypePreference, rule.Name)

						mu.Lock()
						results.MatchedPreferenceRules = append(results.MatchedPreferenceRules, rule.Name)
						mu.Unlock()
						logging.Infof("Preference rule matched: %s", rule.Name)
						break
					}
				}
			}
		}()
	} else if !isSignalTypeUsed(usedSignals, config.SignalTypePreference) {
		logging.Infof("[Signal Computation] Preference signal not used in any decision, skipping evaluation")
	}

	// Evaluate language rules in parallel (only if used in decisions)
	// Only evaluate if language_rules are configured and language classifier is enabled
	if isSignalTypeUsed(usedSignals, config.SignalTypeLanguage) && len(c.Config.LanguageRules) > 0 && c.IsLanguageEnabled() {
		wg.Add(1)
		go func() {
			defer wg.Done()
			start := time.Now()
			languageResult, err := c.languageClassifier.Classify(text)
			elapsed := time.Since(start)
			latencySeconds := elapsed.Seconds()

			// Use the language code directly as the signal name
			languageCode := ""
			if err == nil && languageResult != nil {
				languageCode = languageResult.LanguageCode
			}

			// Record signal extraction metrics
			metrics.RecordSignalExtraction(config.SignalTypeLanguage, languageCode, latencySeconds)

			// Record metrics (use microseconds for better precision)
			results.Metrics.Language.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
			if languageCode != "" && err == nil && languageResult != nil {
				results.Metrics.Language.Confidence = languageResult.Confidence
			}

			logging.Infof("[Signal Computation] Language signal evaluation completed in %v", elapsed)
			if err != nil {
				logging.Errorf("language rule evaluation failed: %v", err)
			} else if languageResult != nil {
				// Check if this language code is defined in language_rules
				for _, rule := range c.Config.LanguageRules {
					if rule.Name == languageCode {
						// Record signal match
						metrics.RecordSignalMatch(config.SignalTypeLanguage, rule.Name)

						mu.Lock()
						results.MatchedLanguageRules = append(results.MatchedLanguageRules, rule.Name)
						mu.Unlock()
						break
					}
				}
			}
		}()
	} else if !isSignalTypeUsed(usedSignals, config.SignalTypeLanguage) {
		logging.Infof("[Signal Computation] Language signal not used in any decision, skipping evaluation")
	}

	// Evaluate context rules in parallel (only if used in decisions)
	// Use contextText for token counting to include all messages in multi-turn conversations
	if isSignalTypeUsed(usedSignals, config.SignalTypeContext) && c.contextClassifier != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			start := time.Now()
			matchedRules, count, err := c.contextClassifier.Classify(contextText)
			elapsed := time.Since(start)

			// Record metrics (use microseconds for better precision)
			results.Metrics.Context.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
			results.Metrics.Context.Confidence = 1.0 // Rule-based, always 1.0

			logging.Infof("[Signal Computation] Context signal evaluation completed in %v (count=%d)", elapsed, count)
			if err != nil {
				logging.Errorf("context rule evaluation failed: %v", err)
			} else {
				mu.Lock()
				results.MatchedContextRules = matchedRules
				results.TokenCount = count
				mu.Unlock()
			}
		}()
	} else if !isSignalTypeUsed(usedSignals, config.SignalTypeContext) {
		logging.Infof("[Signal Computation] Context signal not used in any decision, skipping evaluation")
	}

	// Evaluate complexity rules in parallel (only if used in decisions)
	if isSignalTypeUsed(usedSignals, config.SignalTypeComplexity) && c.complexityClassifier != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			start := time.Now()
			matchedRules, err := c.complexityClassifier.Classify(text)
			elapsed := time.Since(start)
			latencySeconds := elapsed.Seconds()

			// Record signal extraction metrics for each matched rule
			for _, ruleName := range matchedRules {
				metrics.RecordSignalExtraction(config.SignalTypeComplexity, ruleName, latencySeconds)
				metrics.RecordSignalMatch(config.SignalTypeComplexity, ruleName)
			}

			// Record metrics (use microseconds for better precision)
			results.Metrics.Complexity.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
			results.Metrics.Complexity.Confidence = 1.0 // Rule-based, always 1.0

			logging.Infof("[Signal Computation] Complexity signal evaluation completed in %v", elapsed)
			if err != nil {
				logging.Errorf("complexity rule evaluation failed: %v", err)
			} else {
				mu.Lock()
				results.MatchedComplexityRules = matchedRules
				mu.Unlock()
			}
		}()
	} else if !isSignalTypeUsed(usedSignals, config.SignalTypeComplexity) {
		logging.Infof("[Signal Computation] Complexity signal not used in any decision, skipping evaluation")
	}

	// Evaluate modality rules in parallel (only if used in decisions)
	// Uses modality_detector config for classifier/keyword/hybrid detection
	if isSignalTypeUsed(usedSignals, config.SignalTypeModality) && len(c.Config.ModalityRules) > 0 && c.Config.ModalityDetector.Enabled {
		wg.Add(1)
		go func() {
			defer wg.Done()
			start := time.Now()
			modalityResult := c.classifyModality(text, &c.Config.ModalityDetector.ModalityDetectionConfig)
			elapsed := time.Since(start)
			latencySeconds := elapsed.Seconds()

			signalName := modalityResult.Modality

			// Record signal extraction metrics
			metrics.RecordSignalExtraction(config.SignalTypeModality, signalName, latencySeconds)

			// Record metrics
			results.Metrics.Modality.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
			results.Metrics.Modality.Confidence = float64(modalityResult.Confidence)

			logging.Infof("[Signal Computation] Modality signal evaluation completed in %v: %s (confidence=%.3f, method=%s)",
				elapsed, signalName, modalityResult.Confidence, modalityResult.Method)

			// Check if this signal name is defined in modality_rules
			for _, rule := range c.Config.ModalityRules {
				if strings.EqualFold(rule.Name, signalName) {
					metrics.RecordSignalMatch(config.SignalTypeModality, rule.Name)
					mu.Lock()
					results.MatchedModalityRules = append(results.MatchedModalityRules, rule.Name)
					mu.Unlock()
					break
				}
			}
		}()
	} else if !isSignalTypeUsed(usedSignals, config.SignalTypeModality) {
		logging.Infof("[Signal Computation] Modality signal not used in any decision, skipping evaluation")
	}

	// Evaluate jailbreak rules in parallel (only if used in decisions and jailbreak inference is enabled)
	if isSignalTypeUsed(usedSignals, config.SignalTypeJailbreak) && len(c.Config.JailbreakRules) > 0 && c.IsJailbreakEnabled() {
		wg.Add(1)
		go func() {
			defer wg.Done()
			start := time.Now()

			// For each jailbreak rule, build content list and check with the rule's threshold
			for _, rule := range c.Config.JailbreakRules {
				contentToAnalyze := []string{}
				if text != "" {
					contentToAnalyze = append(contentToAnalyze, text)
				}
				if rule.IncludeHistory && len(nonUserMessages) > 0 {
					contentToAnalyze = append(contentToAnalyze, nonUserMessages...)
				}
				if len(contentToAnalyze) == 0 {
					continue
				}

				hasJailbreak, detections, err := c.AnalyzeContentForJailbreakWithThreshold(contentToAnalyze, rule.Threshold)
				if err != nil {
					logging.Errorf("[Signal Computation] Jailbreak rule %q evaluation failed: %v", rule.Name, err)
					continue
				}

				if hasJailbreak {
					metrics.RecordSignalExtraction(config.SignalTypeJailbreak, rule.Name, time.Since(start).Seconds())
					metrics.RecordSignalMatch(config.SignalTypeJailbreak, rule.Name)

					// Find highest-confidence detection for metadata
					var bestType string
					var bestConf float32
					for _, d := range detections {
						if d.IsJailbreak && d.Confidence > bestConf {
							bestConf = d.Confidence
							bestType = d.JailbreakType
						}
					}

					mu.Lock()
					results.MatchedJailbreakRules = append(results.MatchedJailbreakRules, rule.Name)
					// Keep highest-confidence metadata across all rules
					if bestConf > results.JailbreakConfidence {
						results.JailbreakDetected = true
						results.JailbreakType = bestType
						results.JailbreakConfidence = bestConf
					}
					if results.SignalConfidences == nil {
						results.SignalConfidences = make(map[string]float64)
					}
					results.SignalConfidences["jailbreak:"+rule.Name] = float64(bestConf)
					mu.Unlock()
				}
			}

			elapsed := time.Since(start)
			results.Metrics.Jailbreak.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
			if results.JailbreakConfidence > 0 {
				results.Metrics.Jailbreak.Confidence = float64(results.JailbreakConfidence)
			}
			logging.Infof("[Signal Computation] Jailbreak signal evaluation completed in %v", elapsed)
		}()
	} else if !isSignalTypeUsed(usedSignals, config.SignalTypeJailbreak) {
		logging.Infof("[Signal Computation] Jailbreak signal not used in any decision, skipping evaluation")
	}

	// Evaluate PII rules in parallel (only if used in decisions and PII inference is enabled)
	if isSignalTypeUsed(usedSignals, config.SignalTypePII) && len(c.Config.PIIRules) > 0 && c.IsPIIEnabled() {
		wg.Add(1)
		go func() {
			defer wg.Done()
			start := time.Now()

			for _, rule := range c.Config.PIIRules {
				contentToAnalyze := []string{}
				if text != "" {
					contentToAnalyze = append(contentToAnalyze, text)
				}
				if rule.IncludeHistory && len(nonUserMessages) > 0 {
					contentToAnalyze = append(contentToAnalyze, nonUserMessages...)
				}
				if len(contentToAnalyze) == 0 {
					continue
				}

				hasPII, analysisResults, err := c.AnalyzeContentForPIIWithThreshold(contentToAnalyze, rule.Threshold)
				if err != nil {
					logging.Errorf("[Signal Computation] PII rule %q evaluation failed: %v", rule.Name, err)
					continue
				}

				if hasPII {
					// Collect all detected entity types
					entityTypes := make(map[string]bool)
					for _, ar := range analysisResults {
						for _, entity := range ar.Entities {
							entityTypes[entity.EntityType] = true
						}
					}

					// Build allow-list set for fast lookup
					allowSet := make(map[string]bool, len(rule.PIITypesAllowed))
					for _, allowed := range rule.PIITypesAllowed {
						allowSet[strings.ToUpper(allowed)] = true
					}

					// Check if any detected entity type is NOT in the allow-list
					var deniedEntities []string
					for entityType := range entityTypes {
						if !allowSet[strings.ToUpper(entityType)] {
							deniedEntities = append(deniedEntities, entityType)
						}
					}

					if len(deniedEntities) > 0 {
						// Signal fires: denied PII types detected
						metrics.RecordSignalExtraction(config.SignalTypePII, rule.Name, time.Since(start).Seconds())
						metrics.RecordSignalMatch(config.SignalTypePII, rule.Name)

						mu.Lock()
						results.MatchedPIIRules = append(results.MatchedPIIRules, rule.Name)
						results.PIIDetected = true
						// Merge entity types across rules (deduplicate)
						for _, e := range deniedEntities {
							if !slices.Contains(results.PIIEntities, e) {
								results.PIIEntities = append(results.PIIEntities, e)
							}
						}
						mu.Unlock()
					}
				}
			}

			elapsed := time.Since(start)
			results.Metrics.PII.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
			if results.PIIDetected {
				results.Metrics.PII.Confidence = 1.0 // Binary: PII found or not
			}
			logging.Infof("[Signal Computation] PII signal evaluation completed in %v", elapsed)
		}()
	} else if !isSignalTypeUsed(usedSignals, config.SignalTypePII) {
		logging.Infof("[Signal Computation] PII signal not used in any decision, skipping evaluation")
	}

	// Wait for all signal evaluations to complete
	wg.Wait()

	// Phase 2: Apply signal composers (handle signal dependencies)
	// This phase filters signals based on other signals' results
	results = c.applySignalComposers(results)

	return results
}

// EvaluateDecisionWithEngine evaluates all decisions using pre-computed signals
// Accepts SignalResults to avoid duplicate signal computation
func (c *Classifier) EvaluateDecisionWithEngine(signals *SignalResults) (*decision.DecisionResult, error) {
	// Check if decisions are configured
	if len(c.Config.Decisions) == 0 {
		return nil, fmt.Errorf("no decisions configured")
	}

	logging.Infof("Signal evaluation results: keyword=%v, embedding=%v, domain=%v, fact_check=%v, user_feedback=%v, preference=%v, language=%v, context=%v, complexity=%v, modality=%v, authz=%v, jailbreak=%v, pii=%v",
		signals.MatchedKeywordRules, signals.MatchedEmbeddingRules, signals.MatchedDomainRules,
		signals.MatchedFactCheckRules, signals.MatchedUserFeedbackRules, signals.MatchedPreferenceRules,
		signals.MatchedLanguageRules, signals.MatchedContextRules,
		signals.MatchedComplexityRules, signals.MatchedModalityRules, signals.MatchedAuthzRules,
		signals.MatchedJailbreakRules, signals.MatchedPIIRules)
	// Create decision engine
	engine := decision.NewDecisionEngine(
		c.Config.KeywordRules,
		c.Config.EmbeddingRules,
		c.Config.Categories,
		c.Config.Decisions,
		c.Config.Strategy,
	)

	// Evaluate decisions with all signals
	result, err := engine.EvaluateDecisionsWithSignals(&decision.SignalMatches{
		KeywordRules:      signals.MatchedKeywordRules,
		EmbeddingRules:    signals.MatchedEmbeddingRules,
		DomainRules:       signals.MatchedDomainRules,
		FactCheckRules:    signals.MatchedFactCheckRules,
		UserFeedbackRules: signals.MatchedUserFeedbackRules,
		PreferenceRules:   signals.MatchedPreferenceRules,
		LanguageRules:     signals.MatchedLanguageRules,
		ContextRules:      signals.MatchedContextRules,
		ComplexityRules:   signals.MatchedComplexityRules,
		ModalityRules:     signals.MatchedModalityRules,
		SignalConfidences: signals.SignalConfidences,
		AuthzRules:        signals.MatchedAuthzRules,
		JailbreakRules:    signals.MatchedJailbreakRules,
		PIIRules:          signals.MatchedPIIRules,
	})
	if err != nil {
		return nil, fmt.Errorf("decision evaluation failed: %w", err)
	}
	if result == nil {
		return nil, nil
	}

	// Populate matched keywords from signal evaluation
	result.MatchedKeywords = signals.MatchedKeywords

	logging.Infof("Decision evaluation result: decision=%s, confidence=%.3f, matched_rules=%v, matched_keywords=%v",
		result.Decision.Name, result.Confidence, result.MatchedRules, result.MatchedKeywords)

	return result, nil
}

// ModalityClassificationResult holds the result of modality signal classification
type ModalityClassificationResult struct {
	Modality   string  // "AR", "DIFFUSION", or "BOTH"
	Confidence float32 // Confidence score (0.0-1.0)
	Method     string  // Detection method used: "classifier", "keyword", or "hybrid"
}

// classifyModality determines the response modality for a text prompt.
// It supports three configurable methods via ModalityDetectionConfig:
//   - "classifier": ML-based (mmBERT-32K) — errors if model not loaded
//   - "keyword":    Configurable keyword matching — requires keywords in config
//   - "hybrid":     Classifier when available + keyword confirmation/fallback (default)
func (c *Classifier) classifyModality(text string, detectionConfig *config.ModalityDetectionConfig) ModalityClassificationResult {
	if text == "" {
		return ModalityClassificationResult{Modality: "AR", Confidence: 1.0, Method: "default"}
	}

	method := detectionConfig.GetMethod()

	switch method {
	case config.ModalityDetectionClassifier:
		return c.classifyModalityByClassifier(text, detectionConfig)
	case config.ModalityDetectionKeyword:
		return c.classifyModalityByKeyword(text, detectionConfig)
	case config.ModalityDetectionHybrid:
		return c.classifyModalityHybrid(text, detectionConfig)
	default:
		logging.Errorf("[ModalitySignal] BUG: unknown detection method %q — defaulting to AR", method)
		return ModalityClassificationResult{Modality: "AR", Confidence: 0.0, Method: "error/unknown-method"}
	}
}

// classifyModalityByClassifier uses the mmBERT-32K ML classifier exclusively.
func (c *Classifier) classifyModalityByClassifier(text string, cfg *config.ModalityDetectionConfig) ModalityClassificationResult {
	result, err := candle_binding.ClassifyMmBert32KModality(text)
	if err == nil {
		logging.Infof("[ModalitySignal] Classifier: %s (confidence=%.3f) for prompt: %.80s",
			result.Modality, result.Confidence, text)
		return ModalityClassificationResult{
			Modality:   result.Modality,
			Confidence: result.Confidence,
			Method:     "classifier",
		}
	}

	logging.Errorf("[ModalitySignal] Classifier unavailable: %v — defaulting to AR", err)
	return ModalityClassificationResult{Modality: "AR", Confidence: 0.0, Method: "classifier/error"}
}

// classifyModalityByKeyword uses keyword patterns from config to detect modality.
func (c *Classifier) classifyModalityByKeyword(text string, cfg *config.ModalityDetectionConfig) ModalityClassificationResult {
	if cfg == nil || len(cfg.Keywords) == 0 {
		logging.Warnf("[ModalitySignal] Keyword detection requested but no keywords configured — defaulting to AR")
		return ModalityClassificationResult{Modality: "AR", Confidence: 0.5, Method: "keyword/no-config"}
	}

	lowerContent := strings.ToLower(text)

	// Check if any configured image keyword matches
	hasImageIntent := false
	for _, kw := range cfg.Keywords {
		if strings.Contains(lowerContent, strings.ToLower(kw)) {
			hasImageIntent = true
			break
		}
	}

	if !hasImageIntent {
		return ModalityClassificationResult{Modality: "AR", Confidence: 0.8, Method: "keyword"}
	}

	// Image intent detected — check if it's BOTH using both_keywords from config
	if len(cfg.BothKeywords) > 0 {
		for _, kw := range cfg.BothKeywords {
			if strings.Contains(lowerContent, strings.ToLower(kw)) {
				logging.Infof("[ModalitySignal] Keyword: BOTH detected (image + both_keyword %q) for: %.80s", kw, text)
				return ModalityClassificationResult{Modality: "BOTH", Confidence: 0.75, Method: "keyword"}
			}
		}
	}

	logging.Infof("[ModalitySignal] Keyword: DIFFUSION detected for: %.80s", text)
	return ModalityClassificationResult{Modality: "DIFFUSION", Confidence: 0.8, Method: "keyword"}
}

// classifyModalityHybrid uses the ML classifier as primary, with keyword matching as
// fallback (when classifier is unavailable) or confirmation (when classifier confidence is low).
func (c *Classifier) classifyModalityHybrid(text string, cfg *config.ModalityDetectionConfig) ModalityClassificationResult {
	confThreshold := cfg.GetConfidenceThreshold()

	// Try classifier first
	classifierResult, err := candle_binding.ClassifyMmBert32KModality(text)
	if err == nil && classifierResult.Confidence >= confThreshold {
		logging.Infof("[ModalitySignal] Hybrid(classifier): %s (confidence=%.3f, threshold=%.2f) for: %.80s",
			classifierResult.Modality, classifierResult.Confidence, confThreshold, text)
		return ModalityClassificationResult{
			Modality:   classifierResult.Modality,
			Confidence: classifierResult.Confidence,
			Method:     "hybrid/classifier",
		}
	}

	if err == nil {
		// Classifier available but low confidence - use keyword to confirm/override
		keywordResult := c.classifyModalityByKeyword(text, cfg)

		if classifierResult.Modality == keywordResult.Modality {
			logging.Infof("[ModalitySignal] Hybrid(agree): %s (classifier=%.3f, keyword=%.3f) for: %.80s",
				classifierResult.Modality, classifierResult.Confidence, keywordResult.Confidence, text)
			return ModalityClassificationResult{
				Modality:   classifierResult.Modality,
				Confidence: (classifierResult.Confidence + keywordResult.Confidence) / 2,
				Method:     "hybrid/agree",
			}
		}

		lowerThreshold := confThreshold * cfg.GetLowerThresholdRatio()
		if classifierResult.Confidence >= lowerThreshold {
			logging.Infof("[ModalitySignal] Hybrid(classifier-preferred): %s (classifier=%.3f vs keyword=%s) for: %.80s",
				classifierResult.Modality, classifierResult.Confidence, keywordResult.Modality, text)
			return ModalityClassificationResult{
				Modality:   classifierResult.Modality,
				Confidence: classifierResult.Confidence,
				Method:     "hybrid/classifier-preferred",
			}
		}

		logging.Infof("[ModalitySignal] Hybrid(keyword-override): %s (classifier=%s@%.3f too low) for: %.80s",
			keywordResult.Modality, classifierResult.Modality, classifierResult.Confidence, text)
		return ModalityClassificationResult{
			Modality:   keywordResult.Modality,
			Confidence: keywordResult.Confidence,
			Method:     "hybrid/keyword-override",
		}
	}

	// Classifier unavailable - fall back to keyword detection
	logging.Debugf("[ModalitySignal] Hybrid: classifier unavailable (%v), using keyword detection", err)
	return c.classifyModalityByKeyword(text, cfg)
}

// ClassifyCategoryWithEntropy performs category classification with entropy-based reasoning decision
func (c *Classifier) ClassifyCategoryWithEntropy(text string) (string, float64, entropy.ReasoningDecision, error) {
	// Try keyword classifier first
	if c.keywordClassifier != nil {
		category, confidence, err := c.keywordClassifier.Classify(text)
		if err != nil {
			return "", 0.0, entropy.ReasoningDecision{}, err
		}
		if category != "" {
			// Keyword matched - determine reasoning mode from category configuration
			reasoningDecision := c.makeReasoningDecisionForKeywordCategory(category)
			return category, confidence, reasoningDecision, nil
		}
	}

	// Try embedding based similarity classification if properly configured
	if c.keywordEmbeddingClassifier != nil {
		category, confidence, err := c.keywordEmbeddingClassifier.Classify(text)
		if err != nil {
			return "", 0.0, entropy.ReasoningDecision{}, err
		}
		if category != "" {
			// Keyword embedding matched - determine reasoning mode from category configuration
			reasoningDecision := c.makeReasoningDecisionForKeywordCategory(category)
			return category, confidence, reasoningDecision, nil
		}
	}

	// Try in-tree first if properly configured
	if c.IsCategoryEnabled() && c.categoryInference != nil {
		return c.classifyCategoryWithEntropyInTree(text)
	}

	// If in-tree classifier was initialized but config is now invalid, return specific error
	if c.categoryInference != nil && !c.IsCategoryEnabled() {
		return "", 0.0, entropy.ReasoningDecision{}, fmt.Errorf("category classification is not properly configured")
	}

	// Fall back to MCP
	if c.IsMCPCategoryEnabled() && c.mcpCategoryInference != nil {
		return c.classifyCategoryWithEntropyMCP(text)
	}

	return "", 0.0, entropy.ReasoningDecision{}, fmt.Errorf("no category classification method available")
}

// makeReasoningDecisionForKeywordCategory creates a reasoning decision for keyword-matched categories
func (c *Classifier) makeReasoningDecisionForKeywordCategory(category string) entropy.ReasoningDecision {
	// Find the decision configuration
	normalizedCategory := strings.ToLower(strings.TrimSpace(category))
	useReasoning := false

	for _, decision := range c.Config.Decisions {
		if strings.ToLower(decision.Name) == normalizedCategory {
			// Check if the decision has reasoning enabled in its best model
			if len(decision.ModelRefs) > 0 && decision.ModelRefs[0].UseReasoning != nil {
				useReasoning = *decision.ModelRefs[0].UseReasoning
			}
			break
		}
	}

	return entropy.ReasoningDecision{
		UseReasoning:     useReasoning,
		Confidence:       1.0, // Keyword matches have 100% confidence
		DecisionReason:   "keyword_match_category_config",
		FallbackStrategy: "keyword_based_classification",
		TopCategories: []entropy.CategoryProbability{
			{
				Category:    category,
				Probability: 1.0,
			},
		},
	}
}

// classifyCategoryWithEntropyInTree performs category classification with entropy using in-tree model
func (c *Classifier) classifyCategoryWithEntropyInTree(text string) (string, float64, entropy.ReasoningDecision, error) {
	if !c.IsCategoryEnabled() {
		return "", 0.0, entropy.ReasoningDecision{}, fmt.Errorf("category classification is not properly configured")
	}

	// Get full probability distribution
	var result candle_binding.ClassResultWithProbs
	var err error

	result, err = c.categoryInference.ClassifyWithProbabilities(text)
	if err != nil {
		return "", 0.0, entropy.ReasoningDecision{}, fmt.Errorf("classification error: %w", err)
	}

	logging.Infof("Classification result: class=%d, confidence=%.4f, entropy_available=%t",
		result.Class, result.Confidence, len(result.Probabilities) > 0)

	// Get category names for all classes and translate to generic names when configured
	categoryNames := make([]string, len(result.Probabilities))
	for i := range result.Probabilities {
		if name, ok := c.CategoryMapping.GetCategoryFromIndex(i); ok {
			categoryNames[i] = c.translateMMLUToGeneric(name)
		} else {
			categoryNames[i] = fmt.Sprintf("unknown_%d", i)
		}
	}

	// Build decision reasoning map from configuration
	// Use the best model's reasoning capability for each decision
	categoryReasoningMap := make(map[string]bool)
	for _, decision := range c.Config.Decisions {
		useReasoning := false
		if len(decision.ModelRefs) > 0 && decision.ModelRefs[0].UseReasoning != nil {
			// Use the first (best) model's reasoning capability
			useReasoning = *decision.ModelRefs[0].UseReasoning
		}
		categoryReasoningMap[strings.ToLower(decision.Name)] = useReasoning
	}

	// Make entropy-based reasoning decision
	entropyStart := time.Now()
	reasoningDecision := entropy.MakeEntropyBasedReasoningDecision(
		result.Probabilities,
		categoryNames,
		categoryReasoningMap,
		float64(c.Config.CategoryModel.Threshold),
	)
	entropyLatency := time.Since(entropyStart).Seconds()

	// Calculate entropy value for metrics
	entropyValue := entropy.CalculateEntropy(result.Probabilities)

	// Determine top category for metrics
	topCategory := "none"
	if len(reasoningDecision.TopCategories) > 0 {
		topCategory = reasoningDecision.TopCategories[0].Category
	}

	// Validate probability distribution quality
	probSum := float32(0.0)
	for _, prob := range result.Probabilities {
		probSum += prob
	}

	// Record probability distribution quality checks
	if probSum >= 0.99 && probSum <= 1.01 {
		metrics.RecordProbabilityDistributionQuality("sum_check", "valid")
	} else {
		metrics.RecordProbabilityDistributionQuality("sum_check", "invalid")
		logging.Warnf("Probability distribution sum is %.3f (should be ~1.0)", probSum)
	}

	// Check for negative probabilities
	hasNegative := false
	for _, prob := range result.Probabilities {
		if prob < 0 {
			hasNegative = true
			break
		}
	}

	if hasNegative {
		metrics.RecordProbabilityDistributionQuality("negative_check", "invalid")
	} else {
		metrics.RecordProbabilityDistributionQuality("negative_check", "valid")
	}

	// Calculate uncertainty level from entropy value
	entropyResult := entropy.AnalyzeEntropy(result.Probabilities)
	uncertaintyLevel := entropyResult.UncertaintyLevel

	// Record comprehensive entropy classification metrics
	metrics.RecordEntropyClassificationMetrics(
		topCategory,
		uncertaintyLevel,
		entropyValue,
		reasoningDecision.Confidence,
		reasoningDecision.UseReasoning,
		reasoningDecision.DecisionReason,
		topCategory,
		entropyLatency,
	)

	// Check confidence threshold for category determination
	if result.Confidence < c.Config.CategoryModel.Threshold {
		// Determine fallback category (default to "other" if not configured)
		fallbackCategory := c.Config.FallbackCategory
		if fallbackCategory == "" {
			fallbackCategory = "other"
		}

		logging.Infof("Classification confidence (%.4f) below threshold (%.4f), falling back to category: %s",
			result.Confidence, c.Config.CategoryModel.Threshold, fallbackCategory)

		// Record the fallback category as a signal match
		metrics.RecordSignalMatch(config.SignalTypeKeyword, fallbackCategory)

		// Return fallback category instead of empty string to enable proper decision routing
		return fallbackCategory, float64(result.Confidence), reasoningDecision, nil
	}

	// Convert class index to category name and translate to generic
	categoryName, ok := c.CategoryMapping.GetCategoryFromIndex(result.Class)
	if !ok {
		// Determine fallback category (default to "other" if not configured)
		fallbackCategory := c.Config.FallbackCategory
		if fallbackCategory == "" {
			fallbackCategory = "other"
		}

		logging.Warnf("Class index %d not found in category mapping, falling back to: %s", result.Class, fallbackCategory)
		metrics.RecordSignalMatch(config.SignalTypeKeyword, fallbackCategory)
		return fallbackCategory, float64(result.Confidence), reasoningDecision, nil
	}
	genericCategory := c.translateMMLUToGeneric(categoryName)

	// Record the category as a signal match
	metrics.RecordSignalMatch(config.SignalTypeKeyword, genericCategory)

	logging.Infof("Classified as category: %s (mmlu=%s), reasoning_decision: use=%t, confidence=%.3f, reason=%s",
		genericCategory, categoryName, reasoningDecision.UseReasoning, reasoningDecision.Confidence, reasoningDecision.DecisionReason)

	return genericCategory, float64(result.Confidence), reasoningDecision, nil
}

// ClassifyPII performs PII token classification on the given text and returns detected PII types
func (c *Classifier) ClassifyPII(text string) ([]string, error) {
	return c.ClassifyPIIWithThreshold(text, c.Config.PIIModel.Threshold)
}

// ClassifyPIIWithThreshold performs PII token classification with a custom threshold
func (c *Classifier) ClassifyPIIWithThreshold(text string, threshold float32) ([]string, error) {
	if !c.IsPIIEnabled() {
		return []string{}, fmt.Errorf("PII detection is not properly configured")
	}

	if text == "" {
		return []string{}, nil
	}

	// Use ModernBERT PII token classifier for entity detection
	configPath := fmt.Sprintf("%s/config.json", c.Config.PIIModel.ModelID)
	tokenResult, err := c.piiInference.ClassifyTokens(text, configPath)
	if err != nil {
		return nil, fmt.Errorf("PII token classification error: %w", err)
	}

	if len(tokenResult.Entities) > 0 {
		logging.Infof("PII token classification found %d entities", len(tokenResult.Entities))
	}

	// Extract unique PII types from detected entities
	// Translate class_X format to named types using PII mapping
	piiTypes := make(map[string]bool)
	for _, entity := range tokenResult.Entities {
		if entity.Confidence >= threshold {
			// Translate entity type from class_X format to named type (e.g., class_6 → DATE_TIME)
			translatedType := c.PIIMapping.TranslatePIIType(entity.EntityType)
			piiTypes[translatedType] = true
			logging.Infof("Detected PII entity: %s → %s ('%s') at [%d-%d] with confidence %.3f",
				entity.EntityType, translatedType, entity.Text, entity.Start, entity.End, entity.Confidence)
		}
	}

	// Convert to slice
	var result []string
	for piiType := range piiTypes {
		result = append(result, piiType)
	}

	if len(result) > 0 {
		logging.Infof("Detected PII types: %v", result)
	}

	return result, nil
}

// ClassifyPIIWithDetails performs PII token classification and returns full entity details including confidence scores
func (c *Classifier) ClassifyPIIWithDetails(text string) ([]PIIDetection, error) {
	return c.ClassifyPIIWithDetailsAndThreshold(text, c.Config.PIIModel.Threshold)
}

// ClassifyPIIWithDetailsAndThreshold performs PII token classification with a custom threshold and returns full entity details
func (c *Classifier) ClassifyPIIWithDetailsAndThreshold(text string, threshold float32) ([]PIIDetection, error) {
	if !c.IsPIIEnabled() {
		return []PIIDetection{}, fmt.Errorf("PII detection is not properly configured")
	}

	if text == "" {
		return []PIIDetection{}, nil
	}

	// Use PII token classifier for entity detection
	configPath := fmt.Sprintf("%s/config.json", c.Config.PIIModel.ModelID)
	tokenResult, err := c.piiInference.ClassifyTokens(text, configPath)
	if err != nil {
		return nil, fmt.Errorf("PII token classification error: %w", err)
	}

	if len(tokenResult.Entities) > 0 {
		logging.Infof("PII token classification found %d entities", len(tokenResult.Entities))
	}

	// Convert token entities to PII detections, filtering by threshold
	// Translate class_X format to named types using PII mapping
	var detections []PIIDetection
	for _, entity := range tokenResult.Entities {
		if entity.Confidence >= threshold {
			// Translate entity type from class_X format to named type (e.g., class_6 → DATE_TIME)
			translatedType := c.PIIMapping.TranslatePIIType(entity.EntityType)
			detection := PIIDetection{
				EntityType: translatedType,
				Start:      entity.Start,
				End:        entity.End,
				Text:       entity.Text,
				Confidence: entity.Confidence,
			}
			detections = append(detections, detection)
			logging.Infof("Detected PII entity: %s → %s ('%s') at [%d-%d] with confidence %.3f",
				entity.EntityType, translatedType, entity.Text, entity.Start, entity.End, entity.Confidence)
		}
	}

	if len(detections) > 0 {
		// Log unique PII types for compatibility with existing logs
		uniqueTypes := make(map[string]bool)
		for _, d := range detections {
			uniqueTypes[d.EntityType] = true
		}
		types := make([]string, 0, len(uniqueTypes))
		for t := range uniqueTypes {
			types = append(types, t)
		}
		logging.Infof("Detected PII types: %v", types)
	}

	return detections, nil
}

// DetectPIIInContent performs PII classification on all provided content
func (c *Classifier) DetectPIIInContent(allContent []string) []string {
	var detectedPII []string
	seenPII := make(map[string]bool)

	for _, content := range allContent {
		if content != "" {
			// TODO: classifier may not handle the entire content, so we need to split the content into smaller chunks
			piiTypes, err := c.ClassifyPII(content)
			if err != nil {
				logging.Errorf("PII classification error: %v", err)
				// Continue without PII enforcement on error
			} else {
				// Add all detected PII types, avoiding duplicates
				for _, piiType := range piiTypes {
					if !seenPII[piiType] {
						detectedPII = append(detectedPII, piiType)
						seenPII[piiType] = true
						logging.Infof("Detected PII type '%s' in content", piiType)
					}
				}
			}
		}
	}

	return detectedPII
}

// AnalyzeContentForPII performs detailed PII analysis on multiple content pieces
func (c *Classifier) AnalyzeContentForPII(contentList []string) (bool, []PIIAnalysisResult, error) {
	return c.AnalyzeContentForPIIWithThreshold(contentList, c.Config.PIIModel.Threshold)
}

// AnalyzeContentForPIIWithThreshold performs detailed PII analysis with a custom threshold
func (c *Classifier) AnalyzeContentForPIIWithThreshold(contentList []string, threshold float32) (bool, []PIIAnalysisResult, error) {
	if !c.IsPIIEnabled() {
		return false, nil, fmt.Errorf("PII detection is not properly configured")
	}

	var analysisResults []PIIAnalysisResult
	hasPII := false

	for i, content := range contentList {
		if content == "" {
			continue
		}

		var result PIIAnalysisResult
		result.Content = content
		result.ContentIndex = i

		// Use ModernBERT PII token classifier for detailed analysis
		configPath := fmt.Sprintf("%s/config.json", c.Config.PIIModel.ModelID)
		tokenResult, err := c.piiInference.ClassifyTokens(content, configPath)
		if err != nil {
			logging.Errorf("Error analyzing content %d: %v", i, err)
			continue
		}

		// Convert token entities to PII detections
		for _, entity := range tokenResult.Entities {
			if entity.Confidence >= threshold {
				detection := PIIDetection{
					EntityType: entity.EntityType,
					Start:      entity.Start,
					End:        entity.End,
					Text:       entity.Text,
					Confidence: entity.Confidence,
				}
				result.Entities = append(result.Entities, detection)
				result.HasPII = true
				hasPII = true
			}
		}

		analysisResults = append(analysisResults, result)
	}

	return hasPII, analysisResults, nil
}

// SelectBestModelForCategory selects the best model from a decision based on score and TTFT
func (c *Classifier) SelectBestModelForCategory(categoryName string) string {
	decision := c.findDecision(categoryName)
	if decision == nil {
		logging.Warnf("Could not find matching decision %s in config, using default model", categoryName)
		return c.Config.DefaultModel
	}

	bestModel, bestScore := c.selectBestModelInternalForDecision(decision, nil)

	if bestModel == "" {
		logging.Warnf("No models found for decision %s, using default model", categoryName)
		return c.Config.DefaultModel
	}

	logging.Infof("Selected model %s for decision %s with score %.4f", bestModel, categoryName, bestScore)
	return bestModel
}

// findDecision finds the decision configuration by name (case-insensitive)
func (c *Classifier) findDecision(decisionName string) *config.Decision {
	for i, decision := range c.Config.Decisions {
		if strings.EqualFold(decision.Name, decisionName) {
			return &c.Config.Decisions[i]
		}
	}
	return nil
}

// GetDecisionByName returns the decision configuration by name (case-insensitive)
func (c *Classifier) GetDecisionByName(decisionName string) *config.Decision {
	return c.findDecision(decisionName)
}

// GetCategorySystemPrompt returns the system prompt for a specific category if available.
// This is useful when the MCP server provides category-specific system prompts that should
// be injected when processing queries in that category.
// Returns empty string and false if no system prompt is available for the category.
func (c *Classifier) GetCategorySystemPrompt(category string) (string, bool) {
	if c.CategoryMapping == nil {
		return "", false
	}
	return c.CategoryMapping.GetCategorySystemPrompt(category)
}

// GetCategoryDescription returns the description for a given category if available.
// This is useful for logging, debugging, or providing context to downstream systems.
// Returns empty string and false if the category has no description.
func (c *Classifier) GetCategoryDescription(category string) (string, bool) {
	if c.CategoryMapping == nil {
		return "", false
	}
	return c.CategoryMapping.GetCategoryDescription(category)
}

// buildCategoryNameMappings builds translation maps between MMLU-Pro and generic categories
func (c *Classifier) buildCategoryNameMappings() {
	c.MMLUToGeneric = make(map[string]string)
	c.GenericToMMLU = make(map[string][]string)

	// Build set of known MMLU-Pro categories from the model mapping (if available)
	knownMMLU := make(map[string]bool)
	if c.CategoryMapping != nil {
		for _, label := range c.CategoryMapping.IdxToCategory {
			knownMMLU[strings.ToLower(label)] = true
		}
	}

	for _, cat := range c.Config.Categories {
		if len(cat.MMLUCategories) > 0 {
			for _, mmlu := range cat.MMLUCategories {
				key := strings.ToLower(mmlu)
				c.MMLUToGeneric[key] = cat.Name
				c.GenericToMMLU[cat.Name] = append(c.GenericToMMLU[cat.Name], mmlu)
			}
		} else {
			// Fallback: identity mapping when the generic name matches an MMLU category
			nameLower := strings.ToLower(cat.Name)
			if knownMMLU[nameLower] {
				c.MMLUToGeneric[nameLower] = cat.Name
				c.GenericToMMLU[cat.Name] = append(c.GenericToMMLU[cat.Name], cat.Name)
			}
		}
	}
}

// translateMMLUToGeneric translates an MMLU-Pro category to a generic category if mapping exists
func (c *Classifier) translateMMLUToGeneric(mmluCategory string) string {
	if mmluCategory == "" {
		return ""
	}
	if c.MMLUToGeneric == nil {
		return mmluCategory
	}
	if generic, ok := c.MMLUToGeneric[strings.ToLower(mmluCategory)]; ok {
		return generic
	}
	return mmluCategory
}

// selectBestModelInternalForDecision performs the core model selection logic for decisions
//
// modelFilter is optional - if provided, only models passing the filter will be considered
func (c *Classifier) selectBestModelInternalForDecision(decision *config.Decision, modelFilter func(string) bool) (string, float64) {
	bestModel := ""

	// With new architecture, we only support one model per decision (first ModelRef)
	if len(decision.ModelRefs) > 0 {
		modelRef := decision.ModelRefs[0]
		model := modelRef.Model

		if modelFilter == nil || modelFilter(model) {
			// Use LoRA name if specified, otherwise use the base model name
			finalModelName := model
			if modelRef.LoRAName != "" {
				finalModelName = modelRef.LoRAName
				logging.Debugf("Using LoRA adapter '%s' for base model '%s'", finalModelName, model)
			}
			bestModel = finalModelName
		}
	}

	return bestModel, 1.0 // Return score 1.0 since we don't have scores anymore
}

// SelectBestModelFromList selects the best model from a list of candidate models for a given decision
func (c *Classifier) SelectBestModelFromList(candidateModels []string, categoryName string) string {
	if len(candidateModels) == 0 {
		return c.Config.DefaultModel
	}

	decision := c.findDecision(categoryName)
	if decision == nil {
		// Return first candidate if decision not found
		return candidateModels[0]
	}

	bestModel, bestScore := c.selectBestModelInternalForDecision(decision,
		func(model string) bool {
			return slices.Contains(candidateModels, model)
		})

	if bestModel == "" {
		logging.Warnf("No suitable model found from candidates for decision %s, using first candidate", categoryName)
		return candidateModels[0]
	}

	logging.Infof("Selected best model %s for decision %s with score %.4f", bestModel, categoryName, bestScore)
	return bestModel
}

// GetModelsForCategory returns all models that are configured for the given decision
// If a ModelRef has a LoRAName specified, the LoRA name is returned instead of the base model name
func (c *Classifier) GetModelsForCategory(categoryName string) []string {
	var models []string

	for _, decision := range c.Config.Decisions {
		if strings.EqualFold(decision.Name, categoryName) {
			for _, modelRef := range decision.ModelRefs {
				// Use LoRA name if specified, otherwise use the base model name
				if modelRef.LoRAName != "" {
					models = append(models, modelRef.LoRAName)
				} else {
					models = append(models, modelRef.Model)
				}
			}
			break
		}
	}

	return models
}

// updateBestModel updates the best model, score if the new score is better.
func (c *Classifier) updateBestModel(score float64, model string, bestScore *float64, bestModel *string) {
	if score > *bestScore {
		*bestScore = score
		*bestModel = model
	}
}

// IsFactCheckEnabled checks if fact-check classification is enabled and properly configured
func (c *Classifier) IsFactCheckEnabled() bool {
	return c.Config.IsFactCheckClassifierEnabled()
}

// IsHallucinationDetectionEnabled checks if hallucination detection is enabled and properly configured
func (c *Classifier) IsHallucinationDetectionEnabled() bool {
	return c.Config.IsHallucinationModelEnabled()
}

// IsFeedbackDetectorEnabled checks if feedback detection is enabled and properly configured
func (c *Classifier) IsFeedbackDetectorEnabled() bool {
	return c.Config.IsFeedbackDetectorEnabled()
}

// initializeFactCheckClassifier initializes the fact-check classification model
func (c *Classifier) initializeFactCheckClassifier() error {
	if !c.IsFactCheckEnabled() {
		return nil
	}

	classifier, err := NewFactCheckClassifier(&c.Config.HallucinationMitigation.FactCheckModel)
	if err != nil {
		return fmt.Errorf("failed to create fact-check classifier: %w", err)
	}

	if err := classifier.Initialize(); err != nil {
		return fmt.Errorf("failed to initialize fact-check classifier: %w", err)
	}

	c.factCheckClassifier = classifier
	logging.Infof("Fact-check classifier initialized successfully")
	return nil
}

// initializeHallucinationDetector initializes the hallucination detection model
func (c *Classifier) initializeHallucinationDetector() error {
	if !c.IsHallucinationDetectionEnabled() {
		return nil
	}

	detector, err := NewHallucinationDetector(&c.Config.HallucinationMitigation.HallucinationModel)
	if err != nil {
		return fmt.Errorf("failed to create hallucination detector: %w", err)
	}

	if err := detector.Initialize(); err != nil {
		return fmt.Errorf("failed to initialize hallucination detector: %w", err)
	}

	// Initialize NLI model if configured
	if c.Config.HallucinationMitigation.NLIModel.ModelID != "" {
		detector.SetNLIConfig(&c.Config.HallucinationMitigation.NLIModel)
		if err := detector.InitializeNLI(); err != nil {
			// NLI is optional - log warning but don't fail
			logging.Warnf("Failed to initialize NLI model: %v (NLI-enhanced detection will be unavailable)", err)
		} else {
			logging.Infof("NLI model initialized successfully for enhanced hallucination detection")
		}
	}

	c.hallucinationDetector = detector
	logging.Infof("Hallucination detector initialized successfully")
	return nil
}

// initializeFeedbackDetector initializes the feedback detection model
func (c *Classifier) initializeFeedbackDetector() error {
	if !c.IsFeedbackDetectorEnabled() {
		return nil
	}

	detector, err := NewFeedbackDetector(&c.Config.FeedbackDetector)
	if err != nil {
		return fmt.Errorf("failed to create feedback detector: %w", err)
	}

	if err := detector.Initialize(); err != nil {
		return fmt.Errorf("failed to initialize feedback detector: %w", err)
	}

	c.feedbackDetector = detector
	logging.Infof("Feedback detector initialized successfully")
	return nil
}

// IsLanguageEnabled checks if language classification is enabled
func (c *Classifier) IsLanguageEnabled() bool {
	return len(c.Config.LanguageRules) > 0 && c.languageClassifier != nil
}

// IsPreferenceClassifierEnabled checks if preference classification is enabled and properly configured
func (c *Classifier) IsPreferenceClassifierEnabled() bool {
	// Need preference rules configured and either a local Candle model or an external model
	if len(c.Config.PreferenceRules) == 0 {
		return false
	}

	if c.Config.Classifier.PreferenceModel.ModelID != "" {
		return true
	}

	externalCfg := c.Config.FindExternalModelByRole(config.ModelRolePreference)
	return externalCfg != nil &&
		externalCfg.ModelEndpoint.Address != "" &&
		externalCfg.ModelName != ""
}

// initializePreferenceClassifier initializes the preference classifier with external LLM
func (c *Classifier) initializePreferenceClassifier() error {
	if !c.IsPreferenceClassifierEnabled() {
		return nil
	}

	externalCfg := c.Config.FindExternalModelByRole(config.ModelRolePreference)
	classifier, err := NewPreferenceClassifier(externalCfg, c.Config.PreferenceRules, &c.Config.Classifier.PreferenceModel)
	if err != nil {
		return fmt.Errorf("failed to create preference classifier: %w", err)
	}

	c.preferenceClassifier = classifier
	logging.Infof("Preference classifier initialized successfully with %d routes", len(c.Config.PreferenceRules))
	return nil
}

// initializeLanguageClassifier initializes the language classifier
func (c *Classifier) initializeLanguageClassifier() error {
	if len(c.Config.LanguageRules) == 0 {
		return nil
	}

	classifier, err := NewLanguageClassifier(c.Config.LanguageRules)
	if err != nil {
		return fmt.Errorf("failed to create language classifier: %w", err)
	}

	c.languageClassifier = classifier
	logging.Infof("Language classifier initialized")
	return nil
}

// ClassifyFactCheck performs fact-check classification on the given text
// Returns the classification result indicating if the prompt needs fact-checking
func (c *Classifier) ClassifyFactCheck(text string) (*FactCheckResult, error) {
	if c.factCheckClassifier == nil || !c.factCheckClassifier.IsInitialized() {
		return nil, fmt.Errorf("fact-check classifier is not initialized")
	}

	result, err := c.factCheckClassifier.Classify(text)
	if err != nil {
		return nil, fmt.Errorf("fact-check classification failed: %w", err)
	}

	if result != nil {
		logging.Infof("Fact-check classification: needs_fact_check=%v, confidence=%.3f, label=%s",
			result.NeedsFactCheck, result.Confidence, result.Label)
	}

	return result, nil
}

// DetectHallucination checks if an answer contains hallucinations given the context
// context: The tool results or RAG context that should ground the answer
// question: The original user question
// answer: The LLM-generated answer to verify
func (c *Classifier) DetectHallucination(context, question, answer string) (*HallucinationResult, error) {
	if c.hallucinationDetector == nil || !c.hallucinationDetector.IsInitialized() {
		return nil, fmt.Errorf("hallucination detector is not initialized")
	}

	result, err := c.hallucinationDetector.Detect(context, question, answer)
	if err != nil {
		return nil, fmt.Errorf("hallucination detection failed: %w", err)
	}

	if result != nil {
		logging.Infof("Hallucination detection: detected=%v, confidence=%.3f, unsupported_spans=%d",
			result.HallucinationDetected, result.Confidence, len(result.UnsupportedSpans))
	}

	return result, nil
}

// DetectHallucinationWithNLI checks if an answer contains hallucinations with NLI explanations
// context: The tool results or RAG context that should ground the answer
// question: The original user question
// answer: The LLM-generated answer to verify
// Returns enhanced result with detailed NLI analysis for each hallucinated span
func (c *Classifier) DetectHallucinationWithNLI(context, question, answer string) (*EnhancedHallucinationResult, error) {
	if c.hallucinationDetector == nil || !c.hallucinationDetector.IsInitialized() {
		return nil, fmt.Errorf("hallucination detector is not initialized")
	}

	// Check if NLI is initialized
	if !c.hallucinationDetector.IsNLIInitialized() {
		logging.Warnf("NLI model not initialized, falling back to basic hallucination detection")
		// Fall back to basic detection and convert to enhanced format
		basicResult, err := c.hallucinationDetector.Detect(context, question, answer)
		if err != nil {
			return nil, fmt.Errorf("hallucination detection failed: %w", err)
		}
		// Convert basic result to enhanced format
		enhancedResult := &EnhancedHallucinationResult{
			HallucinationDetected: basicResult.HallucinationDetected,
			Confidence:            basicResult.Confidence,
			Spans:                 []EnhancedHallucinationSpan{},
		}
		for _, span := range basicResult.UnsupportedSpans {
			enhancedResult.Spans = append(enhancedResult.Spans, EnhancedHallucinationSpan{
				Text:                    span,
				HallucinationConfidence: basicResult.Confidence,
				NLILabel:                0, // Unknown
				NLILabelStr:             "UNKNOWN",
				NLIConfidence:           0,
				Severity:                2, // Medium
				Explanation:             fmt.Sprintf("Unsupported claim detected (confidence: %.1f%%)", basicResult.Confidence*100),
			})
		}
		return enhancedResult, nil
	}

	result, err := c.hallucinationDetector.DetectWithNLI(context, question, answer)
	if err != nil {
		return nil, fmt.Errorf("hallucination detection with NLI failed: %w", err)
	}

	if result != nil {
		logging.Infof("Hallucination detection (NLI): detected=%v, confidence=%.3f, spans=%d",
			result.HallucinationDetected, result.Confidence, len(result.Spans))
	}

	return result, nil
}

// ClassifyFeedback performs user feedback classification on the given text
// Returns the classification result indicating the type of user feedback
func (c *Classifier) ClassifyFeedback(text string) (*FeedbackResult, error) {
	if c.feedbackDetector == nil || !c.feedbackDetector.IsInitialized() {
		return nil, fmt.Errorf("feedback detector is not initialized")
	}

	result, err := c.feedbackDetector.Classify(text)
	if err != nil {
		return nil, fmt.Errorf("feedback classification failed: %w", err)
	}

	if result != nil {
		logging.Infof("Feedback classification: feedback_type=%s, confidence=%.3f",
			result.FeedbackType, result.Confidence)
	}

	return result, nil
}

// GetFactCheckClassifier returns the fact-check classifier instance
func (c *Classifier) GetFactCheckClassifier() *FactCheckClassifier {
	return c.factCheckClassifier
}

// GetHallucinationDetector returns the hallucination detector instance
func (c *Classifier) GetHallucinationDetector() *HallucinationDetector {
	return c.hallucinationDetector
}

// GetFeedbackDetector returns the feedback detector instance
func (c *Classifier) GetFeedbackDetector() *FeedbackDetector {
	return c.feedbackDetector
}

// GetLanguageClassifier returns the language classifier instance
func (c *Classifier) GetLanguageClassifier() *LanguageClassifier {
	return c.languageClassifier
}

// applySignalComposers applies composer filters to signals that depend on other signals
// This is executed after all signals are computed in parallel
func (c *Classifier) applySignalComposers(results *SignalResults) *SignalResults {
	// Filter complexity signals by composer conditions
	if len(results.MatchedComplexityRules) > 0 && len(c.Config.ComplexityRules) > 0 {
		results.MatchedComplexityRules = c.filterComplexityByComposer(
			results.MatchedComplexityRules,
			results,
		)
	}

	// Future: Add other signals' composer filtering here
	// if len(results.MatchedXxxRules) > 0 { ... }

	return results
}

// filterComplexityByComposer filters complexity rules based on their composer conditions
func (c *Classifier) filterComplexityByComposer(
	matchedRules []string,
	allSignals *SignalResults,
) []string {
	filtered := []string{}

	for _, matched := range matchedRules {
		// Parse rule name (e.g., "code_complexity:hard" -> "code_complexity")
		parts := strings.Split(matched, ":")
		if len(parts) != 2 {
			logging.Warnf("Invalid complexity rule format: %s", matched)
			continue
		}
		ruleName := parts[0]

		// Find the corresponding rule config
		var rule *config.ComplexityRule
		for i := range c.Config.ComplexityRules {
			if c.Config.ComplexityRules[i].Name == ruleName {
				rule = &c.Config.ComplexityRules[i]
				break
			}
		}

		if rule == nil {
			logging.Warnf("Complexity rule config not found: %s", ruleName)
			continue
		}

		// If no composer, keep the result (no filtering)
		if rule.Composer == nil {
			filtered = append(filtered, matched)
			logging.Debugf("Complexity rule '%s' has no composer, keeping result", matched)
			continue
		}

		// Evaluate composer conditions
		if c.evaluateComposer(rule.Composer, allSignals) {
			filtered = append(filtered, matched)
			logging.Infof("Complexity rule '%s' passed composer filter", matched)
		} else {
			logging.Infof("Complexity rule '%s' filtered out by composer", matched)
		}
	}

	return filtered
}

// evaluateComposer evaluates a composer rule tree against signal results.
// Returns true when the tree matches (allowing the complexity rule through the filter).
// A nil composer always returns true (no filter applied).
func (c *Classifier) evaluateComposer(
	composer *config.RuleNode,
	signals *SignalResults,
) bool {
	if composer == nil {
		return true
	}
	return c.evalComposerNode(*composer, signals)
}

// evalComposerNode recursively evaluates a RuleNode against signal results.
func (c *Classifier) evalComposerNode(
	node config.RuleNode,
	signals *SignalResults,
) bool {
	if node.IsLeaf() {
		return c.evalComposerLeaf(node.Type, node.Name, signals)
	}

	switch strings.ToUpper(node.Operator) {
	case "OR":
		for _, child := range node.Conditions {
			if c.evalComposerNode(child, signals) {
				return true
			}
		}
		return false
	case "NOT":
		// Strictly unary: negate the single child's result.
		if len(node.Conditions) != 1 {
			logging.Warnf("Composer NOT operator requires exactly 1 child, got %d — treating as false", len(node.Conditions))
			return false
		}
		return !c.evalComposerNode(node.Conditions[0], signals)
	default: // AND
		for _, child := range node.Conditions {
			if !c.evalComposerNode(child, signals) {
				return false
			}
		}
		return true
	}
}

// evalComposerLeaf evaluates a single signal reference against signal results.
func (c *Classifier) evalComposerLeaf(
	typ, name string,
	signals *SignalResults,
) bool {
	switch typ {
	case "keyword":
		return slices.Contains(signals.MatchedKeywordRules, name)
	case "embedding":
		return slices.Contains(signals.MatchedEmbeddingRules, name)
	case "domain":
		return slices.Contains(signals.MatchedDomainRules, name)
	case "fact_check":
		return slices.Contains(signals.MatchedFactCheckRules, name)
	case "user_feedback":
		return slices.Contains(signals.MatchedUserFeedbackRules, name)
	case "preference":
		return slices.Contains(signals.MatchedPreferenceRules, name)
	case "language":
		return slices.Contains(signals.MatchedLanguageRules, name)
	case "context":
		return slices.Contains(signals.MatchedContextRules, name)
	case "modality":
		return slices.Contains(signals.MatchedModalityRules, name)
	default:
		logging.Warnf("Unknown composer condition type: %s", typ)
		return false
	}
}

// GetQueryEmbedding returns the embedding vector for a query text as float64
// This is used by model selection algorithms for similarity-based selection
// Returns float64 for compatibility with numerical operations
func (c *Classifier) GetQueryEmbedding(text string) []float64 {
	if text == "" {
		return nil
	}

	// Use the candle binding to get the embedding
	// GetEmbedding returns ([]float32, error) with auto-detected dimension
	embedding32, err := candle_binding.GetEmbedding(text, 0)
	if err != nil {
		logging.Debugf("Failed to get query embedding: %v", err)
		return nil
	}

	// Convert float32 to float64 for numerical operations
	embedding64 := make([]float64, len(embedding32))
	for i, v := range embedding32 {
		embedding64[i] = float64(v)
	}

	return embedding64
}
