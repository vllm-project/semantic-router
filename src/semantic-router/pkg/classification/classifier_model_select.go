package classification

import (
	"fmt"
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

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

	logging.Debugf("Selected best model %s for decision %s from candidates (score=%.4f)", bestModel, categoryName, bestScore)
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
			logging.Infof("NLI model initialized for enhanced hallucination detection")
		}
	}

	c.hallucinationDetector = detector
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
	return nil
}

// IsLanguageEnabled checks if language classification is enabled
func (c *Classifier) IsLanguageEnabled() bool {
	return len(c.Config.LanguageRules) > 0 && c.languageClassifier != nil
}

// IsPreferenceClassifierEnabled checks if preference classification is enabled and properly configured
func (c *Classifier) IsPreferenceClassifierEnabled() bool {
	// Need preference rules configured and either contrastive mode or an external model
	if len(c.Config.PreferenceRules) == 0 {
		return false
	}

	if c.Config.PreferenceModel.ContrastiveEnabled() {
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
	preferenceCfg := c.Config.PreferenceModel.WithDefaults()
	classifier, err := NewPreferenceClassifier(externalCfg, c.Config.PreferenceRules, &preferenceCfg)
	if err != nil {
		return fmt.Errorf("failed to create preference classifier: %w", err)
	}

	c.preferenceClassifier = classifier
	logging.Infof("Preference classifier initialized: %d routes", len(c.Config.PreferenceRules))
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
