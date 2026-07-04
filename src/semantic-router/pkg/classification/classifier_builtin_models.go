package classification

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// IsCategoryEnabled checks if category classification is properly configured.
func (c *Classifier) IsCategoryEnabled() bool {
	modelConfigured := c.Config.CategoryModel.ModelID != "" || c.Config.CategoryModel.Backend != nil
	return c.Config.CategoryModel.Active() && modelConfigured && c.Config.CategoryMappingPath != "" && c.CategoryMapping != nil
}

// initializeCategoryClassifier initializes the category classification model.
func (c *Classifier) initializeCategoryClassifier() error {
	if c.Config.CategoryModel.Backend != nil {
		// Remote inference is fully constructed during classifier assembly and has
		// no local model lifecycle to execute.
		return nil
	}
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

	if c.Config.PromptGuard.Protocol != "" {
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

	if err := validateJailbreakPositiveLabels(c.Config.PromptGuard.PositiveLabels, c.JailbreakMapping); err != nil {
		return err
	}

	if c.Config.PromptGuard.Protocol != "" {
		externalCfg := c.Config.FindExternalModelByRole(config.ModelRoleGuardrail)
		logging.ComponentEvent("classifier", "jailbreak_detector_init_started", map[string]interface{}{
			"mode":      c.Config.PromptGuard.Protocol,
			"model_ref": externalCfg.ModelName,
		})
		c.jailbreakModelReady = true
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

	if err := c.jailbreakInitializer.Init(c.Config.PromptGuard.ModelID, c.Config.PromptGuard.UseCPU, numClasses); err != nil {
		return err
	}
	c.jailbreakModelReady = true
	return nil
}

// CheckForJailbreak analyzes the given text for jailbreak attempts.
func (c *Classifier) CheckForJailbreak(ctx context.Context, text string) (bool, string, float32, error) {
	return c.CheckForJailbreakWithThreshold(ctx, text, c.Config.PromptGuard.Threshold)
}

// CheckForJailbreakWithThreshold analyzes the given text for jailbreak attempts with a custom threshold.
func (c *Classifier) CheckForJailbreakWithThreshold(ctx context.Context, text string, threshold float32) (bool, string, float32, error) {
	if !c.IsJailbreakEnabled() {
		return false, "", 0.0, fmt.Errorf("jailbreak detection is not enabled or properly configured")
	}

	if text == "" {
		return false, "", 0.0, nil
	}

	// Scans in chunks so a long text is not judged on its first 512 tokens.
	// The verdict stays argmax-based here: this call reports the predicted
	// class and that class's confidence. A caller that has to threshold
	// P(jailbreak) independently of argmax wants CheckForJailbreakRiskWithThreshold.
	result, scanned, lastErr := c.scanJailbreakChunks(ctx, text)
	if !scanned {
		if lastErr != nil {
			return false, "", 0.0, fmt.Errorf("jailbreak classification failed: %w", lastErr)
		}
		return false, "", 0.0, nil
	}
	logging.Debugf("Jailbreak classification result: %v", result)

	class, confidence := deriveArgmax(result.Probabilities)
	jailbreakType, ok := c.JailbreakMapping.GetJailbreakTypeFromIndex(class)
	if !ok {
		return false, "", 0.0, fmt.Errorf("unknown jailbreak class index: %d", class)
	}

	isJailbreak := confidence >= threshold && isPositiveJailbreakLabel(c.Config.PromptGuard.PositiveLabels, jailbreakType)
	if !isJailbreak && lastErr != nil {
		// A clean verdict needs every chunk; one that was never scored leaves
		// the text unresolved, as CheckForJailbreakRiskWithThreshold does.
		return false, "", 0.0, fmt.Errorf("jailbreak classification failed on part of the text: %w", lastErr)
	}
	if isJailbreak {
		logging.Warnf("JAILBREAK DETECTED: '%s' (confidence: %.3f, threshold: %.3f)",
			jailbreakType, confidence, threshold)
	}

	return isJailbreak, jailbreakType, confidence, nil
}

// AnalyzeContentForJailbreak analyzes multiple content pieces for jailbreak attempts.
func (c *Classifier) AnalyzeContentForJailbreak(ctx context.Context, contentList []string) (bool, []JailbreakDetection, error) {
	return c.AnalyzeContentForJailbreakWithThreshold(ctx, contentList, c.Config.PromptGuard.Threshold)
}

// AnalyzeContentForJailbreakWithThreshold analyzes multiple content pieces for jailbreak attempts with a custom threshold.
func (c *Classifier) AnalyzeContentForJailbreakWithThreshold(ctx context.Context, contentList []string, threshold float32) (bool, []JailbreakDetection, error) {
	if !c.IsJailbreakEnabled() {
		return false, nil, fmt.Errorf("jailbreak detection is not enabled or properly configured")
	}

	var detections []JailbreakDetection
	hasJailbreak := false
	failedCount := 0
	var lastErr error

	for i, content := range contentList {
		if content == "" {
			continue
		}

		isJailbreak, jailbreakType, confidence, err := c.CheckForJailbreakWithThreshold(ctx, content, threshold)
		if err != nil {
			logging.Errorf("Error analyzing content %d: %v", i, err)
			failedCount++
			lastErr = err
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

	// Fail closed: individual inference failures are tolerated as long as some
	// content was actually classified, but if nothing could be classified the
	// caller must not receive a benign "no jailbreak" verdict it cannot
	// distinguish from a clean scan.
	if failedCount > 0 && len(detections) == 0 {
		return false, nil, fmt.Errorf("jailbreak classification failed for all %d content item(s): %w", failedCount, lastErr)
	}

	return hasJailbreak, detections, nil
}

// IsPIIEnabled checks if PII detection is properly configured.
func (c *Classifier) IsPIIEnabled() bool {
	return c.Config.PIIModel.Active() && c.Config.PIIModel.ModelID != "" && c.Config.PIIMappingPath != "" && c.PIIMapping != nil
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

	if err := c.piiInitializer.Init(c.Config.PIIModel.ModelID, c.Config.PIIModel.UseCPU, numPIIClasses); err != nil {
		return err
	}
	c.piiModelReady = true
	return nil
}
