package classification

import (
	"fmt"
	"slices"
	"strings"
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

// createCategoryInitializer creates the category initializer (auto-detecting)
func createCategoryInitializer() CategoryInitializer {
	return &CategoryInitializerImpl{}
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

type JailbreakInitializer interface {
	Init(modelID string, useCPU bool, numClasses ...int) error
}

type LinearJailbreakInitializer struct{}

func (c *LinearJailbreakInitializer) Init(modelID string, useCPU bool, numClasses ...int) error {
	err := candle_binding.InitJailbreakClassifier(modelID, numClasses[0], useCPU)
	if err != nil {
		return err
	}
	logging.Infof("Initialized linear jailbreak classifier with %d classes", numClasses[0])
	return nil
}

type ModernBertJailbreakInitializer struct{}

func (c *ModernBertJailbreakInitializer) Init(modelID string, useCPU bool, numClasses ...int) error {
	err := candle_binding.InitModernBertJailbreakClassifier(modelID, useCPU)
	if err != nil {
		return err
	}
	logging.Infof("Initialized ModernBERT jailbreak classifier (classes auto-detected from model)")
	return nil
}

// createJailbreakInitializer creates the appropriate jailbreak initializer based on configuration
func createJailbreakInitializer(useModernBERT bool) JailbreakInitializer {
	if useModernBERT {
		return &ModernBertJailbreakInitializer{}
	}
	return &LinearJailbreakInitializer{}
}

type JailbreakInference interface {
	Classify(text string) (candle_binding.ClassResult, error)
}

type LinearJailbreakInference struct{}

func (c *LinearJailbreakInference) Classify(text string) (candle_binding.ClassResult, error) {
	return candle_binding.ClassifyJailbreakText(text)
}

type ModernBertJailbreakInference struct{}

func (c *ModernBertJailbreakInference) Classify(text string) (candle_binding.ClassResult, error) {
	return candle_binding.ClassifyModernBertJailbreakText(text)
}

// createJailbreakInferenceCandle creates Candle-based jailbreak inference
func createJailbreakInferenceCandle(useModernBERT bool) JailbreakInference {
	if useModernBERT {
		return &ModernBertJailbreakInference{}
	}
	return &LinearJailbreakInference{}
}

// createJailbreakInference creates the appropriate jailbreak inference based on configuration
// Checks UseVLLM flag to decide between vLLM or Candle implementation
func createJailbreakInference(cfg *config.PromptGuardConfig) (JailbreakInference, error) {
	if cfg.UseVLLM {
		// Use vLLM-based inference
		return NewVLLMJailbreakInference(cfg)
	}
	// Use Candle-based inference
	return createJailbreakInferenceCandle(cfg.UseModernBERT), nil
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
	jailbreakInference, err := createJailbreakInference(&cfg.PromptGuard)
	if err != nil {
		return nil, fmt.Errorf("failed to create jailbreak inference: %w", err)
	}

	// Create jailbreak initializer (only needed for Candle, nil for vLLM)
	var jailbreakInitializer JailbreakInitializer
	if !cfg.PromptGuard.UseVLLM {
		jailbreakInitializer = createJailbreakInitializer(cfg.PromptGuard.UseModernBERT)
	}

	options := []option{
		withJailbreak(jailbreakMapping, jailbreakInitializer, jailbreakInference),
		withPII(piiMapping, createPIIInitializer(), createPIIInference()),
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
		keywordEmbeddingClassifier, err := NewEmbeddingClassifier(cfg.EmbeddingRules)
		if err != nil {
			logging.Errorf("Failed to create keyword embedding classifier: %v", err)
			return nil, err
		}
		options = append(options, withKeywordEmbeddingClassifier(createEmbeddingInitializer(), keywordEmbeddingClassifier))
	}

	// Add in-tree classifier if configured
	if cfg.CategoryModel.ModelID != "" {
		options = append(options, withCategory(categoryMapping, createCategoryInitializer(), createCategoryInference()))
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
	return c.Config.CategoryModel.ModelID != "" && c.Config.CategoryModel.CategoryMappingPath != "" && c.CategoryMapping != nil
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

	return c.categoryInitializer.Init(c.Config.CategoryModel.ModelID, c.Config.CategoryModel.UseCPU, numClasses)
}

// IsJailbreakEnabled checks if jailbreak detection is enabled and properly configured
func (c *Classifier) IsJailbreakEnabled() bool {
	if !c.Config.PromptGuard.Enabled || c.JailbreakMapping == nil {
		return false
	}

	// Check configuration based on whether using vLLM or Candle
	if c.Config.PromptGuard.UseVLLM {
		// For vLLM: need endpoint, model name, and mapping path
		return c.Config.PromptGuard.ClassifierVLLMEndpoint.Address != "" &&
			c.Config.PromptGuard.VLLMModelName != "" &&
			c.Config.PromptGuard.JailbreakMappingPath != ""
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

	start := time.Now()
	result, err = c.jailbreakInference.Classify(text)
	metrics.RecordClassifierLatency("jailbreak", time.Since(start).Seconds())

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

	// Pass numClasses to support auto-detection
	return c.piiInitializer.Init(c.Config.PIIModel.ModelID, c.Config.PIIModel.UseCPU, numPIIClasses)
}

// SignalResults contains all evaluated signal results
type SignalResults struct {
	MatchedKeywordRules   []string
	MatchedEmbeddingRules []string
	MatchedDomainRules    []string
	MatchedFactCheckRules []string // "needs_fact_check" or "no_fact_check_needed"
}

// EvaluateAllRules evaluates all rule types and returns matched rule names
// Returns (matchedKeywordRules, matchedEmbeddingRules, matchedDomainRules)
// matchedKeywordRules: list of matched keyword rule category names
// matchedEmbeddingRules: list of matched embedding rule category names
// matchedDomainRules: list of matched domain rule category names (from category classification)
func (c *Classifier) EvaluateAllRules(text string) ([]string, []string, []string, error) {
	results := c.EvaluateAllSignals(text)
	return results.MatchedKeywordRules, results.MatchedEmbeddingRules, results.MatchedDomainRules, nil
}

// EvaluateAllSignals evaluates all signal types and returns SignalResults
// This is the new method that includes fact_check signals
func (c *Classifier) EvaluateAllSignals(text string) *SignalResults {
	results := &SignalResults{}

	// Evaluate keyword rules - check each rule individually
	if c.keywordClassifier != nil {
		category, _, err := c.keywordClassifier.Classify(text)
		if err != nil {
			logging.Errorf("keyword rule evaluation failed: %v", err)
		} else if category != "" {
			results.MatchedKeywordRules = append(results.MatchedKeywordRules, category)
		}
	}

	// Evaluate embedding rules - check each rule individually
	if c.keywordEmbeddingClassifier != nil {
		category, _, err := c.keywordEmbeddingClassifier.Classify(text)
		if err != nil {
			logging.Errorf("embedding rule evaluation failed: %v", err)
		} else if category != "" {
			results.MatchedEmbeddingRules = append(results.MatchedEmbeddingRules, category)
		}
	}

	// Evaluate domain rules (category classification)
	if c.IsCategoryEnabled() && c.categoryInference != nil && c.CategoryMapping != nil {
		result, err := c.categoryInference.Classify(text)
		if err != nil {
			logging.Errorf("domain rule evaluation failed: %v", err)
		} else {
			// Map class index to category name
			if categoryName, ok := c.CategoryMapping.GetCategoryFromIndex(result.Class); ok {
				if categoryName != "" {
					results.MatchedDomainRules = append(results.MatchedDomainRules, categoryName)
				}
			}
		}
	}

	// Evaluate fact-check rules
	// Only evaluate if fact_check_rules are configured and fact-check classifier is enabled
	if len(c.Config.FactCheckRules) > 0 && c.IsFactCheckEnabled() {
		factCheckResult, err := c.ClassifyFactCheck(text)
		if err != nil {
			logging.Errorf("fact-check rule evaluation failed: %v", err)
		} else if factCheckResult != nil {
			// Determine which signal to output based on classification result
			// Threshold is already applied in ClassifyFactCheck using fact_check_model.threshold
			signalName := "no_fact_check_needed"
			if factCheckResult.NeedsFactCheck {
				signalName = "needs_fact_check"
			}

			// Check if this signal is defined in fact_check_rules
			for _, rule := range c.Config.FactCheckRules {
				if rule.Name == signalName {
					results.MatchedFactCheckRules = append(results.MatchedFactCheckRules, rule.Name)
					break
				}
			}
		}
	}

	return results
}

// EvaluateDecisionWithEngine evaluates all decisions using the DecisionEngine
// Returns the best matching decision based on the configured strategy
func (c *Classifier) EvaluateDecisionWithEngine(text string) (*decision.DecisionResult, error) {
	// Check if decisions are configured
	if len(c.Config.Decisions) == 0 {
		return nil, fmt.Errorf("no decisions configured")
	}

	// Evaluate all signals (includes fact_check)
	signals := c.EvaluateAllSignals(text)

	logging.Infof("Signal evaluation results: keyword=%v, embedding=%v, domain=%v, fact_check=%v",
		signals.MatchedKeywordRules, signals.MatchedEmbeddingRules, signals.MatchedDomainRules, signals.MatchedFactCheckRules)

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
		KeywordRules:   signals.MatchedKeywordRules,
		EmbeddingRules: signals.MatchedEmbeddingRules,
		DomainRules:    signals.MatchedDomainRules,
		FactCheckRules: signals.MatchedFactCheckRules,
	})
	if err != nil {
		return nil, fmt.Errorf("decision evaluation failed: %w", err)
	}

	logging.Infof("Decision evaluation result: decision=%s, confidence=%.3f, matched_rules=%v",
		result.Decision.Name, result.Confidence, result.MatchedRules)

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

	start := time.Now()
	result, err = c.categoryInference.ClassifyWithProbabilities(text)
	metrics.RecordClassifierLatency("category", time.Since(start).Seconds())

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
		fallbackCategory := c.Config.CategoryModel.FallbackCategory
		if fallbackCategory == "" {
			fallbackCategory = "other"
		}

		logging.Infof("Classification confidence (%.4f) below threshold (%.4f), falling back to category: %s",
			result.Confidence, c.Config.CategoryModel.Threshold, fallbackCategory)

		// Record the fallback category classification metric
		metrics.RecordCategoryClassification(fallbackCategory)

		// Return fallback category instead of empty string to enable proper decision routing
		return fallbackCategory, float64(result.Confidence), reasoningDecision, nil
	}

	// Convert class index to category name and translate to generic
	categoryName, ok := c.CategoryMapping.GetCategoryFromIndex(result.Class)
	if !ok {
		// Determine fallback category (default to "other" if not configured)
		fallbackCategory := c.Config.CategoryModel.FallbackCategory
		if fallbackCategory == "" {
			fallbackCategory = "other"
		}

		logging.Warnf("Class index %d not found in category mapping, falling back to: %s", result.Class, fallbackCategory)
		metrics.RecordCategoryClassification(fallbackCategory)
		return fallbackCategory, float64(result.Confidence), reasoningDecision, nil
	}
	genericCategory := c.translateMMLUToGeneric(categoryName)

	// Record the category classification metric
	metrics.RecordCategoryClassification(genericCategory)

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
	start := time.Now()
	tokenResult, err := c.piiInference.ClassifyTokens(text, configPath)
	metrics.RecordClassifierLatency("pii", time.Since(start).Seconds())
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
	start := time.Now()
	tokenResult, err := c.piiInference.ClassifyTokens(text, configPath)
	metrics.RecordClassifierLatency("pii", time.Since(start).Seconds())
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
		start := time.Now()
		tokenResult, err := c.piiInference.ClassifyTokens(content, configPath)
		metrics.RecordClassifierLatency("pii", time.Since(start).Seconds())
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

// GetFactCheckClassifier returns the fact-check classifier instance
func (c *Classifier) GetFactCheckClassifier() *FactCheckClassifier {
	return c.factCheckClassifier
}

// GetHallucinationDetector returns the hallucination detector instance
func (c *Classifier) GetHallucinationDetector() *HallucinationDetector {
	return c.hallucinationDetector
}
