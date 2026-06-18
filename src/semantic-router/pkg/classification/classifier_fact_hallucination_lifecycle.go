package classification

import (
	"fmt"

	candle "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// IsFactCheckEnabled checks if fact-check classification is enabled and properly configured.
func (c *Classifier) IsFactCheckEnabled() bool {
	return c.Config.IsFactCheckClassifierEnabled()
}

// IsHallucinationDetectionEnabled checks if hallucination detection is enabled and properly configured.
func (c *Classifier) IsHallucinationDetectionEnabled() bool {
	return c.Config.IsHallucinationModelEnabled()
}

// initializeFactCheckClassifier initializes the fact-check classification model.
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

// initializeHallucinationDetector initializes the hallucination detection model.
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

	c.initializeHallucinationNLI(detector)
	c.hallucinationDetector = detector
	wireFusionGroundingBackends(detector)
	return nil
}

// wireFusionGroundingBackends injects the candle-backed NLI + hallucination
// detection functions into the looper package so grounding-aware fusion can score
// panel responses. This keeps the candle/CGO dependency out of the looper import
// graph (the looper package stays hermetically testable).
func wireFusionGroundingBackends(detector *HallucinationDetector) {
	looper.SetGroundingBackends(
		func(premise, hypothesis string) (float32, float32, error) {
			r, err := candle.ClassifyNLI(premise, hypothesis)
			if err != nil {
				return 0, 0, err
			}
			return r.EntailmentProb, r.ContradictProb, nil
		},
		func(context, question, answer string) ([]string, float32, error) {
			r, err := detector.Detect(context, question, answer)
			if err != nil {
				return nil, 0, err
			}
			return r.UnsupportedSpans, r.Confidence, nil
		},
	)
}

func (c *Classifier) initializeHallucinationNLI(detector *HallucinationDetector) {
	if c.Config.HallucinationMitigation.NLIModel.ModelID == "" {
		return
	}

	detector.SetNLIConfig(&c.Config.HallucinationMitigation.NLIModel)
	if err := detector.InitializeNLI(); err != nil {
		logging.ComponentWarnEvent("classifier", "hallucination_nli_init_failed", map[string]interface{}{
			"model_ref":          c.Config.HallucinationMitigation.NLIModel.ModelID,
			"error":              err.Error(),
			"enhanced_detection": false,
			"fallback_detection": "basic",
		})
	}
}

// ClassifyFactCheck performs fact-check classification on the given text.
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

// DetectHallucination checks if an answer contains hallucinations given the context.
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

// DetectHallucinationWithNLI checks if an answer contains hallucinations with NLI explanations.
func (c *Classifier) DetectHallucinationWithNLI(context, question, answer string) (*EnhancedHallucinationResult, error) {
	if c.hallucinationDetector == nil || !c.hallucinationDetector.IsInitialized() {
		return nil, fmt.Errorf("hallucination detector is not initialized")
	}

	if !c.hallucinationDetector.IsNLIInitialized() {
		return c.detectHallucinationWithBasicFallback(context, question, answer)
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

func (c *Classifier) detectHallucinationWithBasicFallback(context, question, answer string) (*EnhancedHallucinationResult, error) {
	logging.Warnf("NLI model not initialized, falling back to basic hallucination detection")
	basicResult, err := c.hallucinationDetector.Detect(context, question, answer)
	if err != nil {
		return nil, fmt.Errorf("hallucination detection failed: %w", err)
	}
	enhancedResult := &EnhancedHallucinationResult{
		HallucinationDetected: basicResult.HallucinationDetected,
		Confidence:            basicResult.Confidence,
		Spans:                 []EnhancedHallucinationSpan{},
	}
	for _, span := range basicResult.UnsupportedSpans {
		enhancedResult.Spans = append(enhancedResult.Spans, EnhancedHallucinationSpan{
			Text:                    span,
			HallucinationConfidence: basicResult.Confidence,
			NLILabel:                0,
			NLILabelStr:             "UNKNOWN",
			NLIConfidence:           0,
			Severity:                2,
			Explanation:             fmt.Sprintf("Unsupported claim detected (confidence: %.1f%%)", basicResult.Confidence*100),
		})
	}
	return enhancedResult, nil
}

// GetFactCheckClassifier returns the fact-check classifier instance.
func (c *Classifier) GetFactCheckClassifier() *FactCheckClassifier {
	return c.factCheckClassifier
}

// GetHallucinationDetector returns the hallucination detector instance.
func (c *Classifier) GetHallucinationDetector() *HallucinationDetector {
	return c.hallucinationDetector
}
