package classification

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// IsFactCheckEnabled checks if fact-check classification is enabled and properly configured.
func (c *Classifier) IsFactCheckEnabled() bool {
	return c.ownsDefaultAPIConsumer() && c.Config.NeedsFactCheckModelForAPI()
}

func (c *Classifier) needsFactCheckModelForRuntime() bool {
	return c != nil &&
		c.Config != nil &&
		(c.Config.NeedsFactCheckModelForRouting() ||
			(c.ownsDefaultAPIConsumer() && c.Config.NeedsFactCheckModelForAPI()))
}

// IsHallucinationDetectionEnabled reports whether the configured detector can
// run with the active backend. Endpoint detectors do not depend on native
// binding capabilities.
func (c *Classifier) IsHallucinationDetectionEnabled() bool {
	if !c.needsHallucinationDetectorForRuntime() {
		return false
	}
	if c.Config.HallucinationMitigation.HallucinationModel.NormalizedBackend() == config.HallucinationBackendEndpoint {
		return true
	}
	if c.models != nil {
		cfg := c.Config.HallucinationMitigation.HallucinationModel
		return c.models.localSpec("hallucination_detector", cfg.ModelID, "modernbert", config.RemoteClassifierContractTokenSpans, cfg.UseCPU).Deployment.Provider == "candle"
	}
	return CurrentNativeBackendCapabilities().LocalHallucinationDetection
}

func (c *Classifier) needsHallucinationDetectorForRuntime() bool {
	return c != nil &&
		c.Config != nil &&
		(c.Config.NeedsHallucinationDetectorForRouting() ||
			(c.ownsDefaultAPIConsumer() && c.Config.NeedsHallucinationDetectorForDefaultRuntime()))
}

func (c *Classifier) needsLocalHallucinationNLIForRuntime() bool {
	return c != nil &&
		c.Config != nil &&
		(c.Config.NeedsLocalHallucinationNLIForRouting() ||
			(c.ownsDefaultAPIConsumer() && c.Config.NeedsLocalHallucinationNLIForAPI()))
}

// initializeFactCheckClassifier initializes the fact-check classification model.
func (c *Classifier) initializeFactCheckClassifier() error {
	if !c.needsFactCheckModelForRuntime() {
		return nil
	}

	classifier, err := NewFactCheckClassifier(&c.Config.HallucinationMitigation.FactCheckModel, c.models)
	if err != nil {
		return fmt.Errorf("failed to create fact-check classifier: %w", err)
	}

	if err := classifier.Initialize(); err != nil {
		return fmt.Errorf("failed to initialize fact-check classifier: %w", err)
	}

	// The owned backend admits at its physical resource.
	c.factCheckClassifier = classifier
	return nil
}

// initializeHallucinationDetector initializes the hallucination detection model.
func (c *Classifier) initializeHallucinationDetector() error {
	if !c.needsHallucinationDetectorForRuntime() {
		return nil
	}

	if c.Config.HallucinationMitigation.HallucinationModel.NormalizedBackend() == config.HallucinationBackendEndpoint {
		detector, err := NewEndpointHallucinationDetector(&c.Config.HallucinationMitigation.HallucinationModel, c.models)
		if err != nil {
			return fmt.Errorf("failed to create endpoint hallucination detector: %w", err)
		}
		if err := detector.Initialize(); err != nil {
			return fmt.Errorf("failed to initialize endpoint hallucination detector: %w", err)
		}
		c.endpointHallucinationDetector = detector
		// Wire the detect callback but pass nil for NLI: the endpoint backend
		// does not ship a local NLI model, so panel-mode fusion grounding will
		// gracefully skip ("nli backend not configured") under on_error: skip.
		return nil
	}

	capabilities := CurrentNativeBackendCapabilities()
	if c.models == nil && !capabilities.LocalHallucinationDetection {
		return fmt.Errorf("native backend %q does not support local hallucination detection", capabilities.Name)
	}

	detector, err := NewHallucinationDetector(&c.Config.HallucinationMitigation.HallucinationModel, c.models)
	if err != nil {
		return fmt.Errorf("failed to create hallucination detector: %w", err)
	}

	if err := detector.Initialize(); err != nil {
		return fmt.Errorf("failed to initialize hallucination detector: %w", err)
	}

	if err := c.initializeHallucinationNLI(detector); err != nil {
		_ = detector.Close()
		return err
	}
	c.hallucinationDetector = detector
	return nil
}

// GroundingBackends returns this recipe's prepared functions. The request's
// generation lease keeps them alive; no process-global callback is replaced.
func (c *Classifier) GroundingBackends() *looper.GroundingBackends {
	if c == nil {
		return nil
	}
	backends := &looper.GroundingBackends{}
	if c.hallucinationDetector != nil && c.hallucinationDetector.IsNLIInitialized() {
		backends.NLI = func(ctx context.Context, premise, hypothesis string) (float32, float32, error) {
			result, err := c.hallucinationDetector.ClassifyNLI(ctx, premise, hypothesis)
			if err != nil {
				return 0, 0, err
			}
			return result.EntailmentProb, result.ContradictProb, nil
		}
	}
	if c.IsHallucinationDetectorReady() {
		backends.Detect = func(ctx context.Context, contextText, question, answer string) ([]string, float32, error) {
			result, err := c.DetectHallucination(ctx, contextText, question, answer)
			if err != nil {
				return nil, 0, err
			}
			// Faithfulness uses span overlap; confidence is optional evidence.
			return result.UnsupportedSpans, result.Confidence, nil
		}
	}
	return backends
}

func (c *Classifier) initializeHallucinationNLI(detector *HallucinationDetector) error {
	if !c.needsLocalHallucinationNLIForRuntime() {
		return nil
	}
	detector.SetNLIConfig(&c.Config.HallucinationMitigation.NLIModel)
	return detector.InitializeNLI()
}

// ClassifyFactCheck performs fact-check classification on the given text.
func (c *Classifier) ClassifyFactCheck(ctx context.Context, text string) (*FactCheckResult, error) {
	if c.factCheckClassifier == nil || !c.factCheckClassifier.IsInitialized() {
		return nil, fmt.Errorf("fact-check classifier is not initialized")
	}

	result, err := c.factCheckClassifier.Classify(ctx, text)
	if err != nil {
		return nil, fmt.Errorf("fact-check classification failed: %w", err)
	}

	return result, nil
}

// DetectHallucination checks if an answer contains hallucinations given the context.
func (c *Classifier) DetectHallucination(ctx context.Context, contextText, question, answer string) (*HallucinationResult, error) {
	if c.endpointHallucinationDetector != nil && c.endpointHallucinationDetector.IsInitialized() {
		return c.endpointHallucinationDetector.Detect(ctx, contextText, question, answer)
	}

	if c.hallucinationDetector == nil || !c.hallucinationDetector.IsInitialized() {
		return nil, fmt.Errorf("hallucination detector is not initialized")
	}

	result, err := c.hallucinationDetector.Detect(ctx, contextText, question, answer)
	if err != nil {
		return nil, fmt.Errorf("hallucination detection failed: %w", err)
	}

	return result, nil
}

// DetectHallucinationWithNLI checks if an answer contains hallucinations with NLI explanations.
func (c *Classifier) DetectHallucinationWithNLI(ctx context.Context, contextText, question, answer string) (*EnhancedHallucinationResult, error) {
	if c.endpointHallucinationDetector != nil && c.endpointHallucinationDetector.IsInitialized() {
		return c.endpointHallucinationDetector.DetectWithNLI(ctx, contextText, question, answer)
	}

	if c.hallucinationDetector == nil || !c.hallucinationDetector.IsInitialized() {
		return nil, fmt.Errorf("hallucination detector is not initialized")
	}

	result, err := c.hallucinationDetector.DetectWithNLI(ctx, contextText, question, answer)
	if err != nil {
		return nil, fmt.Errorf("hallucination detection with NLI failed: %w", err)
	}

	if result != nil {
		logging.Infof("Hallucination detection (NLI): detected=%v, confidence=%.3f, spans=%d",
			result.HallucinationDetected, result.Confidence, len(result.Spans))
	}

	return result, nil
}

// GetFactCheckClassifier returns the fact-check classifier instance.
func (c *Classifier) GetFactCheckClassifier() *FactCheckClassifier {
	return c.factCheckClassifier
}

// GetHallucinationDetector returns the hallucination detector instance (legacy candle backend).
func (c *Classifier) GetHallucinationDetector() *HallucinationDetector {
	return c.hallucinationDetector
}

// IsHallucinationDetectorReady returns true if either the candle or endpoint detector is ready.
func (c *Classifier) IsHallucinationDetectorReady() bool {
	if c.endpointHallucinationDetector != nil {
		return c.endpointHallucinationDetector.IsInitialized()
	}
	if c.hallucinationDetector != nil {
		return c.hallucinationDetector.IsInitialized()
	}
	return false
}

// IsHallucinationExplainerReady reports local NLI classifier readiness for the
// public NLI API. Endpoint-generated span explanations are part of detector
// responses and are not a local NLI classifier.
func (c *Classifier) IsHallucinationExplainerReady() bool {
	if c.endpointHallucinationDetector != nil {
		return c.endpointHallucinationDetector.IsNLIInitialized()
	}
	if c.hallucinationDetector != nil {
		return c.hallucinationDetector.IsNLIInitialized()
	}
	return false
}
