package classification

import (
	"fmt"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// IsCategoryEnabled checks if category classification is properly configured.
func (c *Classifier) IsCategoryEnabled() bool {
	return c.Config.CategoryModel.ModelID != "" && c.Config.CategoryMappingPath != "" && c.CategoryMapping != nil
}

// initializeCategoryClassifier initializes the category classification model.
func (c *Classifier) initializeCategoryClassifier() error {
	if !c.IsCategoryEnabled() || c.categoryInitializer == nil {
		return fmt.Errorf("category classification is not properly configured")
	}

	numClasses := c.CategoryMapping.GetCategoryCount()
	if numClasses < 2 {
		return fmt.Errorf("not enough categories for classification, need at least 2, got %d", numClasses)
	}

	logging.ComponentEvent("classifier", "category_classifier_init_started", map[string]interface{}{
		"model_ref": c.Config.CategoryModel.ModelID,
		"classes":   numClasses,
		"use_cpu":   c.Config.CategoryModel.UseCPU,
	})

	return c.categoryInitializer.Init(c.Config.CategoryModel.ModelID, c.Config.CategoryModel.UseCPU, numClasses)
}

// IsJailbreakEnabled checks if jailbreak detection is enabled and properly configured.
func (c *Classifier) IsJailbreakEnabled() bool {
	if !c.Config.PromptGuard.Enabled || c.JailbreakMapping == nil {
		return false
	}

	if c.Config.PromptGuard.UseVLLM {
		externalCfg := c.Config.FindExternalModelByRole(config.ModelRoleGuardrail)
		hasExternalConfig := externalCfg != nil &&
			externalCfg.ModelEndpoint.Address != "" &&
			externalCfg.ModelName != ""

		return c.Config.PromptGuard.JailbreakMappingPath != "" && hasExternalConfig
	}

	return c.Config.PromptGuard.ModelID != "" && c.Config.PromptGuard.JailbreakMappingPath != ""
}

// initializeJailbreakClassifier initializes the jailbreak classification model.
func (c *Classifier) initializeJailbreakClassifier() error {
	if !c.IsJailbreakEnabled() {
		return fmt.Errorf("jailbreak detection is not properly configured")
	}

	if c.Config.PromptGuard.UseVLLM {
		externalCfg := c.Config.FindExternalModelByRole(config.ModelRoleGuardrail)
		logging.ComponentEvent("classifier", "jailbreak_detector_init_started", map[string]interface{}{
			"mode":      "vllm",
			"model_ref": externalCfg.ModelName,
		})
		return nil
	}

	if c.jailbreakInitializer == nil {
		return fmt.Errorf("jailbreak initializer is required for Candle-based inference")
	}

	numClasses := c.JailbreakMapping.GetJailbreakTypeCount()
	if numClasses < 2 {
		return fmt.Errorf("not enough jailbreak types for classification, need at least 2, got %d", numClasses)
	}

	logging.ComponentEvent("classifier", "jailbreak_detector_init_started", map[string]interface{}{
		"mode":      "candle",
		"model_ref": c.Config.PromptGuard.ModelID,
		"classes":   numClasses,
		"use_cpu":   c.Config.PromptGuard.UseCPU,
	})

	return c.jailbreakInitializer.Init(c.Config.PromptGuard.ModelID, c.Config.PromptGuard.UseCPU, numClasses)
}

// CheckForJailbreak analyzes the given text for jailbreak attempts.
func (c *Classifier) CheckForJailbreak(text string) (bool, string, float32, error) {
	return c.CheckForJailbreakWithThreshold(text, c.Config.PromptGuard.Threshold)
}

// CheckForJailbreakWithThreshold analyzes the given text for jailbreak attempts with a custom threshold.
func (c *Classifier) CheckForJailbreakWithThreshold(text string, threshold float32) (bool, string, float32, error) {
	if !c.IsJailbreakEnabled() {
		return false, "", 0.0, fmt.Errorf("jailbreak detection is not enabled or properly configured")
	}

	if text == "" {
		return false, "", 0.0, nil
	}

	var result candle_binding.ClassResult
	var err error

	result, err = c.jailbreakInference.Classify(text)
	if err != nil {
		return false, "", 0.0, fmt.Errorf("jailbreak classification failed: %w", err)
	}
	logging.Debugf("Jailbreak classification result: %v", result)

	jailbreakType, ok := c.JailbreakMapping.GetJailbreakTypeFromIndex(result.Class)
	if !ok {
		return false, "", 0.0, fmt.Errorf("unknown jailbreak class index: %d", result.Class)
	}

	isJailbreak := result.Confidence >= threshold && jailbreakType == "jailbreak"
	if isJailbreak {
		logging.Warnf("JAILBREAK DETECTED: '%s' (confidence: %.3f, threshold: %.3f)",
			jailbreakType, result.Confidence, threshold)
	}

	return isJailbreak, jailbreakType, result.Confidence, nil
}

// AnalyzeContentForJailbreak analyzes multiple content pieces for jailbreak attempts.
func (c *Classifier) AnalyzeContentForJailbreak(contentList []string) (bool, []JailbreakDetection, error) {
	return c.AnalyzeContentForJailbreakWithThreshold(contentList, c.Config.PromptGuard.Threshold)
}

// AnalyzeContentForJailbreakWithThreshold analyzes multiple content pieces for jailbreak attempts with a custom threshold.
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

// IsPIIEnabled checks if PII detection is properly configured.
func (c *Classifier) IsPIIEnabled() bool {
	return c.Config.PIIModel.ModelID != "" && c.Config.PIIMappingPath != "" && c.PIIMapping != nil
}

// initializePIIClassifier initializes the PII token classification model.
func (c *Classifier) initializePIIClassifier() error {
	if !c.IsPIIEnabled() || c.piiInitializer == nil {
		return fmt.Errorf("PII detection is not properly configured")
	}

	numPIIClasses := c.PIIMapping.GetPIITypeCount()
	if numPIIClasses < 2 {
		return fmt.Errorf("not enough PII types for classification, need at least 2, got %d", numPIIClasses)
	}

	logging.ComponentEvent("classifier", "pii_detector_init_started", map[string]interface{}{
		"model_ref": c.Config.PIIModel.ModelID,
		"classes":   numPIIClasses,
		"use_cpu":   c.Config.PIIModel.UseCPU,
	})

	return c.piiInitializer.Init(c.Config.PIIModel.ModelID, c.Config.PIIModel.UseCPU, numPIIClasses)
}
