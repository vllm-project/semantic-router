package classification

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
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
	if c.usesRemoteHallucinationDetector() {
		return true
	}
	if c.models == nil {
		return false
	}
	cfg := c.Config.HallucinationMitigation.HallucinationModel
	_, err := c.models.localSpec("hallucination_detector", cfg.ModelID, "modernbert", config.RemoteClassifierContractTokenSpans, cfg.UseCPU)
	return err == nil
}

// usesRemoteHallucinationDetector reports whether the hallucination detector
// is remote. The compiled plan decides when there is one; without a plan the
// legacy scalar's own token decides, so a configuration that asks for an
// endpoint is never downgraded to the local model because it is incomplete.
// Its problems surface when the detector is built, with the validator's
// message, not as a silent switch of backend.
func (c *Classifier) usesRemoteHallucinationDetector() bool {
	if c.models != nil {
		if spec, ok := c.models.plan.Lookup(c.models.recipe, "hallucination_detector"); ok {
			return spec.Deployment.Provider == "http"
		}
	}
	return c.Config.HallucinationMitigation.HallucinationModel.NormalizedBackend() == config.HallucinationBackendEndpoint
}

func (c *Classifier) needsHallucinationDetectorForRuntime() bool {
	return c != nil &&
		c.Config != nil &&
		(c.Config.NeedsHallucinationDetectorForRouting() ||
			(c.ownsDefaultAPIConsumer() && c.Config.NeedsHallucinationDetectorForDefaultRuntime()))
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

	if c.usesRemoteHallucinationDetector() {
		detector, err := NewEndpointHallucinationDetector(&c.Config.HallucinationMitigation.HallucinationModel, c.models)
		if err != nil {
			return fmt.Errorf("failed to create endpoint hallucination detector: %w", err)
		}
		if err := detector.Initialize(); err != nil {
			return fmt.Errorf("failed to initialize endpoint hallucination detector: %w", err)
		}
		c.endpointHallucinationDetector = detector
		return nil
	}

	detector, err := NewHallucinationDetector(&c.Config.HallucinationMitigation.HallucinationModel, c.models)
	if err != nil {
		return fmt.Errorf("failed to create hallucination detector: %w", err)
	}

	if err := detector.Initialize(); err != nil {
		return fmt.Errorf("failed to initialize hallucination detector: %w", err)
	}
	c.hallucinationDetector = detector
	return nil
}

// GroundingBackends returns this recipe's prepared functions. The request's
// generation lease keeps them alive; no process-global callback is replaced.
// Both grounding references read the hallucination detector: the context
// reference against the request's context, the panel reference against each
// peer response.
func (c *Classifier) GroundingBackends() *looper.GroundingBackends {
	if c == nil {
		return nil
	}
	backends := &looper.GroundingBackends{}
	if c.IsHallucinationDetectorReady() {
		backends.Detect = func(ctx context.Context, contextText, question, answer string) ([]string, float32, error) {
			result, err := c.DetectHallucination(ctx, contextText, question, answer)
			if err != nil {
				return nil, 0, err
			}
			// The score is the detector's summary, its highest
			// hallucinated-token probability when it reports one.
			return result.UnsupportedSpans, result.Confidence, nil
		}
	}
	return backends
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

// DetectHallucinationWithExplanations checks an answer and explains each unsupported span.
func (c *Classifier) DetectHallucinationWithExplanations(ctx context.Context, contextText, question, answer string) (*EnhancedHallucinationResult, error) {
	if c.endpointHallucinationDetector != nil && c.endpointHallucinationDetector.IsInitialized() {
		return c.endpointHallucinationDetector.DetectWithExplanations(ctx, contextText, question, answer)
	}

	if c.hallucinationDetector == nil || !c.hallucinationDetector.IsInitialized() {
		return nil, fmt.Errorf("hallucination detector is not initialized")
	}

	result, err := c.hallucinationDetector.DetectWithExplanations(ctx, contextText, question, answer)
	if err != nil {
		return nil, fmt.Errorf("hallucination detection failed: %w", err)
	}
	return result, nil
}

// GetFactCheckClassifier returns the fact-check classifier instance.
func (c *Classifier) GetFactCheckClassifier() *FactCheckClassifier {
	return c.factCheckClassifier
}

// GetHallucinationDetector returns the local hallucination detector.
func (c *Classifier) GetHallucinationDetector() *HallucinationDetector {
	return c.hallucinationDetector
}

// IsHallucinationDetectorReady returns true if either the local or the endpoint detector is ready.
func (c *Classifier) IsHallucinationDetectorReady() bool {
	if c.endpointHallucinationDetector != nil {
		return c.endpointHallucinationDetector.IsInitialized()
	}
	if c.hallucinationDetector != nil {
		return c.hallucinationDetector.IsInitialized()
	}
	return false
}
