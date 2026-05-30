package classification

import (
	"strings"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// ModalityClassificationResult holds the result of modality signal classification.
type ModalityClassificationResult struct {
	Modality   string  // "AR", "DIFFUSION", or "BOTH"
	Confidence float32 // Confidence score (0.0-1.0)
	Method     string  // Detection method used: "classifier", "keyword", or "hybrid"
}

// classifyModality determines the response modality for a text prompt.
// It supports three configurable methods via ModalityDetectionConfig:
//   - "classifier": ML-based (mmBERT-32K) errors if model not loaded
//   - "keyword":    Configurable keyword matching requires keywords in config
//   - "hybrid":     Classifier when available plus keyword confirmation/fallback
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
		logging.Errorf("[ModalitySignal] BUG: unknown detection method %q; defaulting to AR", method)
		return ModalityClassificationResult{Modality: "AR", Confidence: 0.0, Method: "error/unknown-method"}
	}
}

// classifyModalityByClassifier uses the mmBERT-32K ML classifier exclusively.
func (c *Classifier) classifyModalityByClassifier(text string, cfg *config.ModalityDetectionConfig) ModalityClassificationResult {
	result, err := candle_binding.ClassifyMmBert32KModality(text)
	if err == nil {
		logging.Debugf("[ModalitySignal] Classifier: %s (confidence=%.3f) for prompt: %.80s",
			result.Modality, result.Confidence, text)
		return ModalityClassificationResult{
			Modality:   result.Modality,
			Confidence: result.Confidence,
			Method:     "classifier",
		}
	}

	logging.Errorf("[ModalitySignal] Classifier unavailable: %v; defaulting to AR", err)
	return ModalityClassificationResult{Modality: "AR", Confidence: 0.0, Method: "classifier/error"}
}

// classifyModalityByKeyword uses keyword patterns from config to detect modality.
func (c *Classifier) classifyModalityByKeyword(text string, cfg *config.ModalityDetectionConfig) ModalityClassificationResult {
	if cfg == nil || len(cfg.Keywords) == 0 {
		logging.Warnf("[ModalitySignal] Keyword detection requested but no keywords configured; defaulting to AR")
		return ModalityClassificationResult{Modality: "AR", Confidence: 0.5, Method: "keyword/no-config"}
	}

	lowerContent := strings.ToLower(text)

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

	if len(cfg.BothKeywords) > 0 {
		for _, kw := range cfg.BothKeywords {
			if strings.Contains(lowerContent, strings.ToLower(kw)) {
				logging.Debugf("[ModalitySignal] Keyword: BOTH detected (image + both_keyword %q) for: %.80s", kw, text)
				return ModalityClassificationResult{Modality: "BOTH", Confidence: 0.75, Method: "keyword"}
			}
		}
	}

	logging.Debugf("[ModalitySignal] Keyword: DIFFUSION detected for: %.80s", text)
	return ModalityClassificationResult{Modality: "DIFFUSION", Confidence: 0.8, Method: "keyword"}
}

// classifyModalityHybrid uses the ML classifier as primary, with keyword matching as
// fallback when unavailable or confirmation when classifier confidence is low.
func (c *Classifier) classifyModalityHybrid(text string, cfg *config.ModalityDetectionConfig) ModalityClassificationResult {
	confThreshold := cfg.GetConfidenceThreshold()

	classifierResult, err := candle_binding.ClassifyMmBert32KModality(text)
	if err == nil && classifierResult.Confidence >= confThreshold {
		logging.Debugf("[ModalitySignal] Hybrid(classifier): %s (confidence=%.3f, threshold=%.2f) for: %.80s",
			classifierResult.Modality, classifierResult.Confidence, confThreshold, text)
		return ModalityClassificationResult{
			Modality:   classifierResult.Modality,
			Confidence: classifierResult.Confidence,
			Method:     "hybrid/classifier",
		}
	}

	if err == nil {
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

		logging.Debugf("[ModalitySignal] Hybrid(keyword-override): %s (classifier=%s@%.3f too low) for: %.80s",
			keywordResult.Modality, classifierResult.Modality, classifierResult.Confidence, text)
		return ModalityClassificationResult{
			Modality:   keywordResult.Modality,
			Confidence: keywordResult.Confidence,
			Method:     "hybrid/keyword-override",
		}
	}

	logging.Debugf("[ModalitySignal] Hybrid: classifier unavailable (%v), using keyword detection", err)
	return c.classifyModalityByKeyword(text, cfg)
}
