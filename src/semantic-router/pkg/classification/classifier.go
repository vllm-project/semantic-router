package classification

import (
	"fmt"
	"slices"
	"strings"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

type CategoryInitializer interface {
	Init(modelID string, useCPU bool, numClasses ...int) error
}

type LinearCategoryInitializer struct{}

func (c *LinearCategoryInitializer) Init(modelID string, useCPU bool, numClasses ...int) error {
	err := candle_binding.InitClassifier(modelID, numClasses[0], useCPU)
	if err != nil {
		return err
	}
	logging.Infof("Initialized linear category classifier with %d classes", numClasses[0])
	return nil
}

type ModernBertCategoryInitializer struct{}

func (c *ModernBertCategoryInitializer) Init(modelID string, useCPU bool, numClasses ...int) error {
	err := candle_binding.InitModernBertClassifier(modelID, useCPU)
	if err != nil {
		return err
	}
	logging.Infof("Initialized ModernBERT category classifier (classes auto-detected from model)")
	return nil
}

// createCategoryInitializer creates the appropriate category initializer based on configuration
func createCategoryInitializer(useModernBERT bool) CategoryInitializer {
	if useModernBERT {
		return &ModernBertCategoryInitializer{}
	}
	return &LinearCategoryInitializer{}
}

type CategoryInference interface {
	Classify(text string) (candle_binding.ClassResult, error)
	ClassifyWithProbabilities(text string) (candle_binding.ClassResultWithProbs, error)
}

type LinearCategoryInference struct{}

func (c *LinearCategoryInference) Classify(text string) (candle_binding.ClassResult, error) {
	return candle_binding.ClassifyText(text)
}

func (c *LinearCategoryInference) ClassifyWithProbabilities(text string) (candle_binding.ClassResultWithProbs, error) {
	return candle_binding.ClassifyTextWithProbabilities(text)
}

type ModernBertCategoryInference struct{}

func (c *ModernBertCategoryInference) Classify(text string) (candle_binding.ClassResult, error) {
	return candle_binding.ClassifyModernBertText(text)
}

func (c *ModernBertCategoryInference) ClassifyWithProbabilities(text string) (candle_binding.ClassResultWithProbs, error) {
	return candle_binding.ClassifyModernBertTextWithProbabilities(text)
}

// createCategoryInference creates the appropriate category inference based on configuration
func createCategoryInference(useModernBERT bool) CategoryInference {
	if useModernBERT {
		return &ModernBertCategoryInference{}
	}
	return &LinearCategoryInference{}
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

// createJailbreakInference creates the appropriate jailbreak inference based on configuration
func createJailbreakInference(useModernBERT bool) JailbreakInference {
	if useModernBERT {
		return &ModernBertJailbreakInference{}
	}
	return &LinearJailbreakInference{}
}

type PIIInitializer interface {
	Init(modelID string, useCPU bool) error
}

type ModernBertPIIInitializer struct{}

func (c *ModernBertPIIInitializer) Init(modelID string, useCPU bool) error {
	err := candle_binding.InitModernBertPIITokenClassifier(modelID, useCPU)
	if err != nil {
		return err
	}
	logging.Infof("Initialized ModernBERT PII token classifier for entity detection")
	return nil
}

// createPIIInitializer creates the appropriate PII initializer (currently only ModernBERT)
func createPIIInitializer() PIIInitializer { return &ModernBertPIIInitializer{} }

type PIIInference interface {
	ClassifyTokens(text string, configPath string) (candle_binding.TokenClassificationResult, error)
}

type ModernBertPIIInference struct{}

func (c *ModernBertPIIInference) ClassifyTokens(text string, configPath string) (candle_binding.TokenClassificationResult, error) {
	return candle_binding.ClassifyModernBertPIITokens(text, configPath)
}

// createPIIInference creates the appropriate PII inference (currently only ModernBERT)
func createPIIInference() PIIInference { return &ModernBertPIIInference{} }

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

	// NEW: Unified classifier for LoRA models (preferred when available)
	UnifiedClassifier *UnifiedClassifier

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
	options := []option{
		withJailbreak(jailbreakMapping, createJailbreakInitializer(cfg.PromptGuard.UseModernBERT), createJailbreakInference(cfg.PromptGuard.UseModernBERT)),
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
		options = append(options, withCategory(categoryMapping, createCategoryInitializer(cfg.UseModernBERT), createCategoryInference(cfg.UseModernBERT)))
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

	return c.categoryInitializer.Init(c.Config.CategoryModel.ModelID, c.Config.CategoryModel.UseCPU, numClasses)
}

// IsJailbreakEnabled checks if jailbreak detection is enabled and properly configured
func (c *Classifier) IsJailbreakEnabled() bool {
	return c.Config.PromptGuard.Enabled && c.Config.PromptGuard.ModelID != "" && c.Config.PromptGuard.JailbreakMappingPath != "" && c.JailbreakMapping != nil
}

// initializeJailbreakClassifier initializes the jailbreak classification model
func (c *Classifier) initializeJailbreakClassifier() error {
	if !c.IsJailbreakEnabled() || c.jailbreakInitializer == nil {
		return fmt.Errorf("jailbreak detection is not properly configured")
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

	return c.piiInitializer.Init(c.Config.PIIModel.ModelID, c.Config.PIIModel.UseCPU)
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

	// Try UnifiedClassifier (LoRA models) first - highest accuracy
	if c.UnifiedClassifier != nil {
		return c.classifyWithUnifiedClassifier(text)
	}

	// Try in-tree classifier if properly configured
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
	// Find the category configuration
	normalizedCategory := strings.ToLower(strings.TrimSpace(category))
	useReasoning := false

	for _, cat := range c.Config.Categories {
		if strings.ToLower(cat.Name) == normalizedCategory {
			// Check if the category has reasoning enabled in its best model
			if len(cat.ModelScores) > 0 && cat.ModelScores[0].UseReasoning != nil {
				useReasoning = *cat.ModelScores[0].UseReasoning
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

// classifyWithUnifiedClassifier uses UnifiedClassifier (LoRA models) for classification
func (c *Classifier) classifyWithUnifiedClassifier(text string) (string, float64, entropy.ReasoningDecision, error) {
	// Use batch classification with single item
	results, err := c.UnifiedClassifier.ClassifyBatch([]string{text})
	if err != nil {
		return "", 0.0, entropy.ReasoningDecision{}, fmt.Errorf("unified classifier error: %w", err)
	}

	if len(results.IntentResults) == 0 {
		return "", 0.0, entropy.ReasoningDecision{}, fmt.Errorf("no classification results from unified classifier")
	}

	intentResult := results.IntentResults[0]
	category := intentResult.Category
	confidence := float64(intentResult.Confidence)

	// Build reasoning decision based on category configuration
	reasoningDecision := c.makeReasoningDecisionForCategory(category, confidence)

	return category, confidence, reasoningDecision, nil
}

// makeReasoningDecisionForCategory creates reasoning decision based on category config
func (c *Classifier) makeReasoningDecisionForCategory(category string, confidence float64) entropy.ReasoningDecision {
	normalizedCategory := strings.ToLower(strings.TrimSpace(category))
	useReasoning := false

	for _, cat := range c.Config.Categories {
		if strings.ToLower(cat.Name) == normalizedCategory {
			if len(cat.ModelScores) > 0 && cat.ModelScores[0].UseReasoning != nil {
				useReasoning = *cat.ModelScores[0].UseReasoning
			}
			break
		}
	}

	return entropy.ReasoningDecision{
		UseReasoning:     useReasoning,
		Confidence:       confidence,
		DecisionReason:   "unified_lora_classification",
		FallbackStrategy: "lora_based_classification",
		TopCategories: []entropy.CategoryProbability{
			{
				Category:    category,
				Probability: float32(confidence),
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

	// Build category reasoning map from configuration
	// Use the best model's reasoning capability for each category
	categoryReasoningMap := make(map[string]bool)
	for _, category := range c.Config.Categories {
		useReasoning := false
		if len(category.ModelScores) > 0 && category.ModelScores[0].UseReasoning != nil {
			// Use the first (best) model's reasoning capability
			useReasoning = *category.ModelScores[0].UseReasoning
		}
		categoryReasoningMap[strings.ToLower(category.Name)] = useReasoning
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
		logging.Infof("Classification confidence (%.4f) below threshold (%.4f), but entropy analysis available",
			result.Confidence, c.Config.CategoryModel.Threshold)

		// Still return reasoning decision based on entropy even if confidence is low
		return "", float64(result.Confidence), reasoningDecision, nil
	}

	// Convert class index to category name and translate to generic
	categoryName, ok := c.CategoryMapping.GetCategoryFromIndex(result.Class)
	if !ok {
		logging.Warnf("Class index %d not found in category mapping", result.Class)
		return "", float64(result.Confidence), reasoningDecision, nil
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
	piiTypes := make(map[string]bool)
	for _, entity := range tokenResult.Entities {
		if entity.Confidence >= threshold {
			piiTypes[entity.EntityType] = true
			logging.Infof("Detected PII entity: %s ('%s') at [%d-%d] with confidence %.3f",
				entity.EntityType, entity.Text, entity.Start, entity.End, entity.Confidence)
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

// SelectBestModelForCategory selects the best model from a category based on score and TTFT
func (c *Classifier) SelectBestModelForCategory(categoryName string) string {
	cat := c.findCategory(categoryName)
	if cat == nil {
		logging.Warnf("Could not find matching category %s in config, using default model", categoryName)
		return c.Config.DefaultModel
	}

	bestModel, bestScore := c.selectBestModelInternal(cat, nil)

	if bestModel == "" {
		logging.Warnf("No models found for category %s, using default model", categoryName)
		return c.Config.DefaultModel
	}

	logging.Infof("Selected model %s for category %s with score %.4f", bestModel, categoryName, bestScore)
	return bestModel
}

// findCategory finds the category configuration by name (case-insensitive)
func (c *Classifier) findCategory(categoryName string) *config.Category {
	for i, category := range c.Config.Categories {
		if strings.EqualFold(category.Name, categoryName) {
			return &c.Config.Categories[i]
		}
	}
	return nil
}

// GetCategoryByName returns the category configuration by name (case-insensitive)
// This is a public method that can be used by other packages to get category information
func (c *Classifier) GetCategoryByName(categoryName string) *config.Category {
	return c.findCategory(categoryName)
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

// selectBestModelInternal performs the core model selection logic
//
// modelFilter is optional - if provided, only models passing the filter will be considered
func (c *Classifier) selectBestModelInternal(cat *config.Category, modelFilter func(string) bool) (string, float64) {
	bestModel := ""
	bestScore := -1.0

	c.forEachModelScore(cat, func(modelScore config.ModelScore) {
		model := modelScore.Model
		if modelFilter != nil && !modelFilter(model) {
			return
		}
		// Use LoRA name if specified, otherwise use the base model name
		// This enables intent-aware LoRA routing where the final model name
		// in the request becomes the LoRA adapter name
		finalModelName := model
		if modelScore.LoRAName != "" {
			finalModelName = modelScore.LoRAName
			logging.Debugf("Using LoRA adapter '%s' for base model '%s'", finalModelName, model)
		}
		c.updateBestModel(modelScore.Score, finalModelName, &bestScore, &bestModel)
	})

	return bestModel, bestScore
}

// forEachModelScore traverses the ModelScores document of the category and executes the callback for each element.
func (c *Classifier) forEachModelScore(cat *config.Category, fn func(modelScore config.ModelScore)) {
	for _, modelScore := range cat.ModelScores {
		fn(modelScore)
	}
}

// SelectBestModelFromList selects the best model from a list of candidate models for a given category
func (c *Classifier) SelectBestModelFromList(candidateModels []string, categoryName string) string {
	if len(candidateModels) == 0 {
		return c.Config.DefaultModel
	}

	cat := c.findCategory(categoryName)
	if cat == nil {
		// Return first candidate if category not found
		return candidateModels[0]
	}

	bestModel, bestScore := c.selectBestModelInternal(cat,
		func(model string) bool {
			return slices.Contains(candidateModels, model)
		})

	if bestModel == "" {
		logging.Warnf("No suitable model found from candidates for category %s, using first candidate", categoryName)
		return candidateModels[0]
	}

	logging.Infof("Selected best model %s for category %s with score %.4f", bestModel, categoryName, bestScore)
	return bestModel
}

// GetModelsForCategory returns all models that are configured for the given category
// If a ModelScore has a LoRAName specified, the LoRA name is returned instead of the base model name
func (c *Classifier) GetModelsForCategory(categoryName string) []string {
	var models []string

	for _, category := range c.Config.Categories {
		if strings.EqualFold(category.Name, categoryName) {
			for _, modelScore := range category.ModelScores {
				// Use LoRA name if specified, otherwise use the base model name
				if modelScore.LoRAName != "" {
					models = append(models, modelScore.LoRAName)
				} else {
					models = append(models, modelScore.Model)
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
