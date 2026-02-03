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

	// Latency classifier
	latencyClassifier *LatencyClassifier

	// Context classifier for token count-based routing
	contextClassifier *ContextClassifier

	// Complexity classifier for complexity-based routing using embedding similarity
	complexityClassifier *ComplexityClassifier

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

	// Initialize latency classifier
	if len(classifier.Config.LatencyRules) > 0 {
		if err := classifier.initializeLatencyClassifier(); err != nil {
			logging.Warnf("Failed to initialize latency classifier: %v", err)
			// Non-fatal - continue without latency classification
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
		optConfig := cfg.HNSWConfig
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
		modelType := cfg.HNSWConfig.ModelType
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
	MatchedLatencyRules      []string // Latency rule names that matched based on model TPOT
	MatchedContextRules      []string // Matched context rule names (e.g. "low_token_count")
	TokenCount               int      // Total token count
	MatchedComplexityRules   []string // Matched complexity rules with difficulty level (e.g. "code_complexity:hard")
}

// analyzeRuleCombination recursively analyzes rule combinations to find used signals
func (c *Classifier) analyzeRuleCombination(rules config.RuleCombination, usedSignals map[string]bool) {
	for _, condition := range rules.Conditions {
		// Normalize condition type and name (trim whitespace, lowercase) for consistent matching
		t := strings.ToLower(strings.TrimSpace(condition.Type))
		n := strings.ToLower(strings.TrimSpace(condition.Name))
		signalKey := t + ":" + n
		usedSignals[signalKey] = true
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
	return c.EvaluateAllSignalsWithContext(text, text)
}

// EvaluateAllSignalsWithContext evaluates all signal types with separate text for context counting
// text: text to use for signal evaluation (usually latest user message)
// contextText: text to use for context token counting (usually all messages combined)
func (c *Classifier) EvaluateAllSignalsWithContext(text string, contextText string) *SignalResults {
	// Determine which signals (type:name) are actually used in decisions
	usedSignals := c.getUsedSignals()

	results := &SignalResults{}
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
	if isSignalTypeUsed(usedSignals, config.SignalTypeEmbedding) && c.keywordEmbeddingClassifier != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			start := time.Now()
			category, _, err := c.keywordEmbeddingClassifier.Classify(text)
			elapsed := time.Since(start)
			latencySeconds := elapsed.Seconds()

			// Record signal extraction metrics
			metrics.RecordSignalExtraction(config.SignalTypeEmbedding, category, latencySeconds)

			logging.Infof("[Signal Computation] Embedding signal evaluation completed in %v", elapsed)
			if err != nil {
				logging.Errorf("embedding rule evaluation failed: %v", err)
			} else if category != "" {
				// Record signal match
				metrics.RecordSignalMatch(config.SignalTypeEmbedding, category)

				mu.Lock()
				results.MatchedEmbeddingRules = append(results.MatchedEmbeddingRules, category)
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

	// Evaluate latency rules in parallel (only if used in decisions)
	// Latency evaluation is model-aware, so we need to collect models from decisions that use latency signals
	if isSignalTypeUsed(usedSignals, config.SignalTypeLatency) && len(c.Config.LatencyRules) > 0 && c.IsLatencyEnabled() {
		wg.Add(1)
		go func() {
			defer wg.Done()
			start := time.Now()

			// Collect all models from decisions that use latency signals
			availableModels := c.collectModelsForLatencySignals(usedSignals)

			if len(availableModels) > 0 {
				latencyResult, err := c.latencyClassifier.Classify(availableModels)
				elapsed := time.Since(start)
				latencySeconds := elapsed.Seconds()

				// Record signal extraction metrics for each matched latency rule
				if err == nil && latencyResult != nil {
					for _, ruleName := range latencyResult.MatchedRules {
						metrics.RecordSignalExtraction(config.SignalTypeLatency, ruleName, latencySeconds)
						metrics.RecordSignalMatch(config.SignalTypeLatency, ruleName)
					}
				} else {
					// Record extraction even if no match
					metrics.RecordSignalExtraction(config.SignalTypeLatency, "", latencySeconds)
				}

				logging.Infof("[Signal Computation] Latency signal evaluation completed in %v", elapsed)
				if err != nil {
					logging.Errorf("latency rule evaluation failed: %v", err)
				} else if latencyResult != nil {
					mu.Lock()
					results.MatchedLatencyRules = latencyResult.MatchedRules
					mu.Unlock()
				}
			} else if isSignalTypeUsed(usedSignals, config.SignalTypeLatency) {
				// Diagnostic: latency signals are used but no models found
				// This can happen if decisions reference latency signals but have no ModelRefs
				latencySignals := []string{}
				for key := range usedSignals {
					if strings.HasPrefix(strings.ToLower(key), config.SignalTypeLatency+":") {
						latencySignals = append(latencySignals, key)
					}
				}
				logging.Warnf("[Signal Computation] Latency signals are used (%v) but no models found in decisions. Latency routing will be skipped.", latencySignals)
			}
		}()
	} else if !isSignalTypeUsed(usedSignals, config.SignalTypeLatency) {
		logging.Infof("[Signal Computation] Latency signal not used in any decision, skipping evaluation")
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

	logging.Infof("Signal evaluation results: keyword=%v, embedding=%v, domain=%v, fact_check=%v, user_feedback=%v, preference=%v, language=%v, latency=%v, context=%v, complexity=%v",
		signals.MatchedKeywordRules, signals.MatchedEmbeddingRules, signals.MatchedDomainRules,
		signals.MatchedFactCheckRules, signals.MatchedUserFeedbackRules, signals.MatchedPreferenceRules,
		signals.MatchedLanguageRules, signals.MatchedLatencyRules, signals.MatchedContextRules,
		signals.MatchedComplexityRules)
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
		LatencyRules:      signals.MatchedLatencyRules,
		ContextRules:      signals.MatchedContextRules,
		ComplexityRules:   signals.MatchedComplexityRules,
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

// collectModelsForLatencySignals collects all models from decisions that use latency signals
func (c *Classifier) collectModelsForLatencySignals(usedSignals map[string]bool) []string {
	modelSet := make(map[string]bool)

	for i := range c.Config.Decisions {
		decision := &c.Config.Decisions[i]
		// Check if this decision uses latency signals
		usesLatency := false
		// usedSignals keys are already normalized to lowercase by analyzeRuleCombination
		latencyPrefix := config.SignalTypeLatency + ":"
		for key := range usedSignals {
			if strings.HasPrefix(key, latencyPrefix) {
				// Check if this decision's rules reference this latency signal
				// decisionUsesLatencySignal normalizes condition.Type/Name from config
				if c.decisionUsesLatencySignal(decision, key) {
					usesLatency = true
					break
				}
			}
		}

		if usesLatency {
			// Collect models from this decision
			for _, modelRef := range decision.ModelRefs {
				modelName := ""
				if modelRef.LoRAName != "" {
					modelName = modelRef.LoRAName
				} else {
					modelName = modelRef.Model
				}
				// Skip empty model names
				if modelName != "" {
					modelSet[modelName] = true
				}
			}
		}
	}

	// Convert set to slice and sort deterministically
	var models []string
	for model := range modelSet {
		models = append(models, model)
	}
	// Sort deterministically to ensure consistent ordering
	slices.Sort(models)

	return models
}

// decisionUsesLatencySignal checks if a decision uses a latency signal key
// condition.Type and condition.Name come from config, so we normalize them for comparison
func (c *Classifier) decisionUsesLatencySignal(decision *config.Decision, normalizedSignalKey string) bool {
	for _, condition := range decision.Rules.Conditions {
		// Normalize condition from config for comparison (all signals are normalized to lowercase)
		normalizedType := strings.ToLower(strings.TrimSpace(condition.Type))
		if normalizedType == config.SignalTypeLatency {
			currentKey := normalizedType + ":" + strings.ToLower(strings.TrimSpace(condition.Name))
			if currentKey == normalizedSignalKey {
				return true
			}
		}
	}
	return false
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
	// Need preference rules configured and external model with role="preference"
	if len(c.Config.PreferenceRules) == 0 {
		return false
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
	if externalCfg == nil {
		return fmt.Errorf("external model with role='preference' not found")
	}

	classifier, err := NewPreferenceClassifier(externalCfg, c.Config.PreferenceRules)
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

// initializeLatencyClassifier initializes the latency classifier
func (c *Classifier) initializeLatencyClassifier() error {
	if len(c.Config.LatencyRules) == 0 {
		return nil
	}

	classifier, err := NewLatencyClassifier(c.Config.LatencyRules)
	if err != nil {
		return fmt.Errorf("failed to create latency classifier: %w", err)
	}

	c.latencyClassifier = classifier
	logging.Infof("Latency classifier initialized")
	return nil
}

// IsLatencyEnabled checks if latency classification is enabled
func (c *Classifier) IsLatencyEnabled() bool {
	return len(c.Config.LatencyRules) > 0 && c.latencyClassifier != nil
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

// evaluateComposer evaluates a composer's conditions against signal results
func (c *Classifier) evaluateComposer(
	composer *config.RuleCombination,
	signals *SignalResults,
) bool {
	if composer == nil {
		return true
	}

	// Evaluate each condition
	conditionResults := make([]bool, len(composer.Conditions))
	for i, condition := range composer.Conditions {
		conditionResults[i] = c.evaluateComposerCondition(&condition, signals)
	}

	// Apply operator (AND/OR)
	if composer.Operator == "OR" {
		for _, result := range conditionResults {
			if result {
				return true
			}
		}
		return false
	} else { // Default to AND
		for _, result := range conditionResults {
			if !result {
				return false
			}
		}
		return true
	}
}

// evaluateComposerCondition evaluates a single condition against signal results
func (c *Classifier) evaluateComposerCondition(
	condition *config.RuleCondition,
	signals *SignalResults,
) bool {
	switch condition.Type {
	case "keyword":
		return slices.Contains(signals.MatchedKeywordRules, condition.Name)
	case "embedding":
		return slices.Contains(signals.MatchedEmbeddingRules, condition.Name)
	case "domain":
		return slices.Contains(signals.MatchedDomainRules, condition.Name)
	case "fact_check":
		return slices.Contains(signals.MatchedFactCheckRules, condition.Name)
	case "user_feedback":
		return slices.Contains(signals.MatchedUserFeedbackRules, condition.Name)
	case "preference":
		return slices.Contains(signals.MatchedPreferenceRules, condition.Name)
	case "language":
		return slices.Contains(signals.MatchedLanguageRules, condition.Name)
	case "latency":
		return slices.Contains(signals.MatchedLatencyRules, condition.Name)
	case "context":
		return slices.Contains(signals.MatchedContextRules, condition.Name)
	default:
		logging.Warnf("Unknown composer condition type: %s", condition.Type)
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
