package classification

import (
	"fmt"
	"log"
	"slices"
	"strings"
	"sync"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/semantic-router/pkg/metrics"
)

type CategoryInference interface {
	Classify(text string) (candle_binding.ClassResult, error)
}

type LinearCategoryInference struct{}

func (c *LinearCategoryInference) Classify(text string) (candle_binding.ClassResult, error) {
	return candle_binding.ClassifyText(text)
}

type ModernBertCategoryInference struct{}

func (c *ModernBertCategoryInference) Classify(text string) (candle_binding.ClassResult, error) {
	return candle_binding.ClassifyModernBertText(text)
}

// createCategoryInference creates the appropriate category inference based on configuration
func createCategoryInference(useModernBERT bool) CategoryInference {
	if useModernBERT {
		return &ModernBertCategoryInference{}
	}
	return &LinearCategoryInference{}
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

type PIIInference interface {
	ClassifyTokens(text string, configPath string) (candle_binding.TokenClassificationResult, error)
}

type ModernBertPIIInference struct{}

func (c *ModernBertPIIInference) ClassifyTokens(text string, configPath string) (candle_binding.TokenClassificationResult, error) {
	return candle_binding.ClassifyModernBertPIITokens(text, configPath)
}

// createPIIInference creates the appropriate PII inference (currently only ModernBERT)
func createPIIInference() PIIInference {
	return &ModernBertPIIInference{}
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
	// Dependencies
	categoryInference  CategoryInference
	jailbreakInference JailbreakInference
	piiInference       PIIInference

	Config           *config.RouterConfig
	CategoryMapping  *CategoryMapping
	PIIMapping       *PIIMapping
	JailbreakMapping *JailbreakMapping
	// Model selection fields
	ModelLoad     map[string]int
	ModelLoadLock sync.Mutex
	ModelTTFT     map[string]float64
	// Jailbreak detection state
	JailbreakInitialized bool
}

// NewClassifier creates a new classifier with model selection and jailbreak detection capabilities
func NewClassifier(cfg *config.RouterConfig, categoryMapping *CategoryMapping, piiMapping *PIIMapping, jailbreakMapping *JailbreakMapping, modelTTFT map[string]float64) *Classifier {
	return &Classifier{
		categoryInference:  createCategoryInference(cfg.Classifier.CategoryModel.UseModernBERT),
		jailbreakInference: createJailbreakInference(cfg.PromptGuard.UseModernBERT),
		piiInference:       createPIIInference(),

		Config:               cfg,
		CategoryMapping:      categoryMapping,
		PIIMapping:           piiMapping,
		JailbreakMapping:     jailbreakMapping,
		ModelLoad:            make(map[string]int),
		ModelTTFT:            modelTTFT,
		JailbreakInitialized: false,
	}
}

// InitializeJailbreakClassifier initializes the jailbreak classification model
func (c *Classifier) InitializeJailbreakClassifier() error {
	if !c.IsJailbreakEnabled() {
		return nil
	}

	numClasses := c.JailbreakMapping.GetJailbreakTypeCount()
	if numClasses < 2 {
		return fmt.Errorf("not enough jailbreak types for classification, need at least 2, got %d", numClasses)
	}

	var err error
	if c.Config.PromptGuard.UseModernBERT {
		// Initialize ModernBERT jailbreak classifier
		err = candle_binding.InitModernBertJailbreakClassifier(c.Config.PromptGuard.ModelID, c.Config.PromptGuard.UseCPU)
		if err != nil {
			return fmt.Errorf("failed to initialize ModernBERT jailbreak classifier: %w", err)
		}
		log.Printf("Initialized ModernBERT jailbreak classifier (classes auto-detected from model)")
	} else {
		// Initialize linear jailbreak classifier
		err = candle_binding.InitJailbreakClassifier(c.Config.PromptGuard.ModelID, numClasses, c.Config.PromptGuard.UseCPU)
		if err != nil {
			return fmt.Errorf("failed to initialize jailbreak classifier: %w", err)
		}
		log.Printf("Initialized linear jailbreak classifier with %d classes", numClasses)
	}

	c.JailbreakInitialized = true
	return nil
}

// IsJailbreakEnabled checks if jailbreak detection is enabled and properly configured
func (c *Classifier) IsJailbreakEnabled() bool {
	return c.Config.PromptGuard.Enabled && c.Config.PromptGuard.ModelID != "" && c.Config.PromptGuard.JailbreakMappingPath != "" && c.JailbreakMapping != nil
}

// CheckForJailbreak analyzes the given text for jailbreak attempts
func (c *Classifier) CheckForJailbreak(text string) (bool, string, float32, error) {
	if !c.IsJailbreakEnabled() || !c.JailbreakInitialized {
		return false, "", 0.0, nil
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
	log.Printf("Jailbreak classification result: %v", result)

	// Get the jailbreak type name from the class index
	jailbreakType, ok := c.JailbreakMapping.GetJailbreakTypeFromIndex(result.Class)
	if !ok {
		return false, "", 0.0, fmt.Errorf("unknown jailbreak class index: %d", result.Class)
	}

	// Check if confidence meets threshold and indicates jailbreak
	isJailbreak := result.Confidence >= c.Config.PromptGuard.Threshold && jailbreakType == "jailbreak"

	if isJailbreak {
		log.Printf("JAILBREAK DETECTED: '%s' (confidence: %.3f, threshold: %.3f)",
			jailbreakType, result.Confidence, c.Config.PromptGuard.Threshold)
	} else {
		log.Printf("BENIGN: '%s' (confidence: %.3f, threshold: %.3f)",
			jailbreakType, result.Confidence, c.Config.PromptGuard.Threshold)
	}

	return isJailbreak, jailbreakType, result.Confidence, nil
}

// AnalyzeContentForJailbreak analyzes multiple content pieces for jailbreak attempts
func (c *Classifier) AnalyzeContentForJailbreak(contentList []string) (bool, []JailbreakDetection, error) {
	if !c.IsJailbreakEnabled() || !c.JailbreakInitialized {
		return false, nil, nil
	}

	var detections []JailbreakDetection
	hasJailbreak := false

	for i, content := range contentList {
		if content == "" {
			continue
		}

		isJailbreak, jailbreakType, confidence, err := c.CheckForJailbreak(content)
		if err != nil {
			log.Printf("Error analyzing content %d: %v", i, err)
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

// ClassifyCategory performs category classification on the given text
func (c *Classifier) ClassifyCategory(text string) (string, float64, error) {
	if c.CategoryMapping == nil {
		return "", 0.0, fmt.Errorf("category mapping not initialized")
	}

	// Use appropriate classifier based on configuration
	var result candle_binding.ClassResult
	var err error

	start := time.Now()
	result, err = c.categoryInference.Classify(text)
	metrics.RecordClassifierLatency("category", time.Since(start).Seconds())

	if err != nil {
		return "", 0.0, fmt.Errorf("classification error: %w", err)
	}

	log.Printf("Classification result: class=%d, confidence=%.4f", result.Class, result.Confidence)

	// Check confidence threshold
	if result.Confidence < c.Config.Classifier.CategoryModel.Threshold {
		log.Printf("Classification confidence (%.4f) below threshold (%.4f)",
			result.Confidence, c.Config.Classifier.CategoryModel.Threshold)
		return "", float64(result.Confidence), nil
	}

	// Convert class index to category name
	categoryName, ok := c.CategoryMapping.GetCategoryFromIndex(result.Class)
	if !ok {
		log.Printf("Class index %d not found in category mapping", result.Class)
		return "", float64(result.Confidence), nil
	}

	// Record the category classification metric
	metrics.RecordCategoryClassification(categoryName)

	log.Printf("Classified as category: %s", categoryName)
	return categoryName, float64(result.Confidence), nil
}

// ClassifyPII performs PII token classification on the given text and returns detected PII types
func (c *Classifier) ClassifyPII(text string) ([]string, error) {
	if c.PIIMapping == nil {
		return []string{}, nil // No PII classifier enabled
	}

	if text == "" {
		return []string{}, nil
	}

	// Use ModernBERT PII token classifier for entity detection
	configPath := fmt.Sprintf("%s/config.json", c.Config.Classifier.PIIModel.ModelID)
	start := time.Now()
	tokenResult, err := c.piiInference.ClassifyTokens(text, configPath)
	metrics.RecordClassifierLatency("pii", time.Since(start).Seconds())
	if err != nil {
		return nil, fmt.Errorf("PII token classification error: %w", err)
	}

	if len(tokenResult.Entities) > 0 {
		log.Printf("PII token classification found %d entities", len(tokenResult.Entities))
	}

	// Extract unique PII types from detected entities
	piiTypes := make(map[string]bool)
	for _, entity := range tokenResult.Entities {
		if entity.Confidence >= c.Config.Classifier.PIIModel.Threshold {
			piiTypes[entity.EntityType] = true
			log.Printf("Detected PII entity: %s ('%s') at [%d-%d] with confidence %.3f",
				entity.EntityType, entity.Text, entity.Start, entity.End, entity.Confidence)
		}
	}

	// Convert to slice
	var result []string
	for piiType := range piiTypes {
		result = append(result, piiType)
	}

	if len(result) > 0 {
		log.Printf("Detected PII types: %v", result)
	}

	return result, nil
}

// DetectPIIInContent performs PII classification on all provided content
func (c *Classifier) DetectPIIInContent(allContent []string) []string {
	var detectedPII []string
	seenPII := make(map[string]bool)

	for _, content := range allContent {
		if content != "" {
			//TODO: classifier may not handle the entire content, so we need to split the content into smaller chunks
			piiTypes, err := c.ClassifyPII(content)
			if err != nil {
				log.Printf("PII classification error: %v", err)
				// Continue without PII enforcement on error
			} else {
				// Add all detected PII types, avoiding duplicates
				for _, piiType := range piiTypes {
					if !seenPII[piiType] {
						detectedPII = append(detectedPII, piiType)
						seenPII[piiType] = true
						log.Printf("Detected PII type '%s' in content", piiType)
					}
				}
			}
		}
	}

	return detectedPII
}

// AnalyzeContentForPII performs detailed PII analysis on multiple content pieces
func (c *Classifier) AnalyzeContentForPII(contentList []string) (bool, []PIIAnalysisResult, error) {
	if c.PIIMapping == nil {
		return false, nil, nil // No PII classifier enabled
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
		configPath := fmt.Sprintf("%s/config.json", c.Config.Classifier.PIIModel.ModelID)
		start := time.Now()
		tokenResult, err := c.piiInference.ClassifyTokens(content, configPath)
		metrics.RecordClassifierLatency("pii", time.Since(start).Seconds())
		if err != nil {
			log.Printf("Error analyzing content %d: %v", i, err)
			continue
		}

		// Convert token entities to PII detections
		for _, entity := range tokenResult.Entities {
			if entity.Confidence >= c.Config.Classifier.PIIModel.Threshold {
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

// ClassifyAndSelectBestModel performs classification and selects the best model for the query
func (c *Classifier) ClassifyAndSelectBestModel(query string) string {
	// If no categories defined, return default model
	if len(c.Config.Categories) == 0 {
		return c.Config.DefaultModel
	}

	// First, classify the text to determine the category
	categoryName, confidence, err := c.ClassifyCategory(query)
	if err != nil {
		log.Printf("Classification error: %v, falling back to default model", err)
		return c.Config.DefaultModel
	}

	if categoryName == "" {
		log.Printf("Classification confidence (%.4f) below threshold, using default model", confidence)
		return c.Config.DefaultModel
	}

	// Then select the best model from the determined category based on score and TTFT
	return c.SelectBestModelForCategory(categoryName)
}

// SelectBestModelForCategory selects the best model from a category based on score and TTFT
func (c *Classifier) SelectBestModelForCategory(categoryName string) string {
	var cat *config.Category
	for i, category := range c.Config.Categories {
		if strings.EqualFold(category.Name, categoryName) {
			cat = &c.Config.Categories[i]
			break
		}
	}

	if cat == nil {
		log.Printf("Could not find matching category %s in config, using default model", categoryName)
		return c.Config.DefaultModel
	}

	c.ModelLoadLock.Lock()
	defer c.ModelLoadLock.Unlock()

	bestModel := ""
	bestScore := -1.0
	bestQuality := 0.0

	if c.Config.Classifier.LoadAware {
		c.forEachModelScore(cat, func(modelScore config.ModelScore) {
			quality := modelScore.Score
			model := modelScore.Model
			baseTTFT := c.ModelTTFT[model]
			load := c.ModelLoad[model]
			estTTFT := baseTTFT * (1 + float64(load))
			if estTTFT == 0 {
				estTTFT = 1
			}
			score := quality / estTTFT
			c.updateBestModel(score, quality, model, &bestScore, &bestQuality, &bestModel)
		})
	} else {
		c.forEachModelScore(cat, func(modelScore config.ModelScore) {
			quality := modelScore.Score
			model := modelScore.Model
			c.updateBestModel(quality, quality, model, &bestScore, &bestQuality, &bestModel)
		})
	}

	if bestModel == "" {
		log.Printf("No models found for category %s, using default model", categoryName)
		return c.Config.DefaultModel
	}

	log.Printf("Selected model %s for category %s with quality %.4f and combined score %.4e",
		bestModel, categoryName, bestQuality, bestScore)
	return bestModel
}

// forEachModelScore traverses the ModelScores document of the category and executes the callback for each element。
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

	// Find the category configuration
	var cat *config.Category
	for i, category := range c.Config.Categories {
		if strings.EqualFold(category.Name, categoryName) {
			cat = &c.Config.Categories[i]
			break
		}
	}

	if cat == nil {
		// Return first candidate if category not found
		return candidateModels[0]
	}

	c.ModelLoadLock.Lock()
	defer c.ModelLoadLock.Unlock()

	bestModel := ""
	bestScore := -1.0
	bestQuality := 0.0

	filteredFn := func(modelScore config.ModelScore) {
		model := modelScore.Model
		if !slices.Contains(candidateModels, model) {
			return
		}
		quality := modelScore.Score
		if c.Config.Classifier.LoadAware {
			baseTTFT := c.ModelTTFT[model]
			load := c.ModelLoad[model]
			estTTFT := baseTTFT * (1 + float64(load))
			if estTTFT == 0 {
				estTTFT = 1 // avoid div by zero
			}
			score := quality / estTTFT
			c.updateBestModel(score, quality, model, &bestScore, &bestQuality, &bestModel)
		} else {
			c.updateBestModel(quality, quality, model, &bestScore, &bestQuality, &bestModel)
		}
	}

	c.forEachModelScore(cat, filteredFn)

	if bestModel == "" {
		log.Printf("No suitable model found from candidates for category %s, using first candidate", categoryName)
		return candidateModels[0]
	}

	log.Printf("Selected best model %s for category %s with quality %.4f and combined score %.4e",
		bestModel, categoryName, bestQuality, bestScore)
	return bestModel
}

// GetModelsForCategory returns all models that are configured for the given category
func (c *Classifier) GetModelsForCategory(categoryName string) []string {
	var models []string

	for _, category := range c.Config.Categories {
		if strings.EqualFold(category.Name, categoryName) {
			for _, modelScore := range category.ModelScores {
				models = append(models, modelScore.Model)
			}
			break
		}
	}

	return models
}

// IncrementModelLoad increments the load counter for a model
func (c *Classifier) IncrementModelLoad(model string) {
	c.ModelLoadLock.Lock()
	defer c.ModelLoadLock.Unlock()
	c.ModelLoad[model]++
}

// DecrementModelLoad decrements the load counter for a model
func (c *Classifier) DecrementModelLoad(model string) {
	c.ModelLoadLock.Lock()
	defer c.ModelLoadLock.Unlock()
	if c.ModelLoad[model] > 0 {
		c.ModelLoad[model]--
	}
}

// updateBestModel updates the best model, score, and quality if the new score is better.
func (c *Classifier) updateBestModel(score, quality float64, model string, bestScore *float64, bestQuality *float64, bestModel *string) {
	if score > *bestScore {
		*bestScore = score
		*bestModel = model
		*bestQuality = quality
	}
}
