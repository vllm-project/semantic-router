package classification

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// IsFeedbackDetectorEnabled checks if feedback detection is enabled and properly configured.
func (c *Classifier) IsFeedbackDetectorEnabled() bool {
	return c.ownsDefaultAPIConsumer() && c.Config.NeedsFeedbackModelForAPI()
}

func (c *Classifier) needsFeedbackModelForRuntime() bool {
	return c != nil &&
		c.Config != nil &&
		(c.Config.NeedsFeedbackModelForRouting() ||
			(c.ownsDefaultAPIConsumer() && c.Config.NeedsFeedbackModelForAPI()))
}

// initializeFeedbackDetector initializes the feedback detection model.
func (c *Classifier) initializeFeedbackDetector() error {
	if !c.needsFeedbackModelForRuntime() {
		return nil
	}

	detector, err := NewFeedbackDetector(&c.Config.FeedbackDetector, c.models)
	if err != nil {
		return fmt.Errorf("failed to create feedback detector: %w", err)
	}

	if err := detector.Initialize(); err != nil {
		return fmt.Errorf("failed to initialize feedback detector: %w", err)
	}

	// The owned backend admits at its physical resource.

	c.feedbackDetector = detector
	return nil
}

// ClassifyFeedback performs user feedback classification on the given text.
func (c *Classifier) ClassifyFeedback(ctx context.Context, text string) (*FeedbackResult, error) {
	if c.feedbackDetector == nil || !c.feedbackDetector.IsInitialized() {
		return nil, fmt.Errorf("feedback detector is not initialized")
	}

	result, err := c.feedbackDetector.Classify(ctx, text)
	if err != nil {
		return nil, fmt.Errorf("feedback classification failed: %w", err)
	}

	if result != nil {
		logging.Infof("Feedback classification: feedback_type=%s, confidence_available=%v",
			result.FeedbackType, result.ConfidenceAvailable)
	}

	return result, nil
}

// GetFeedbackDetector returns the feedback detector instance.
func (c *Classifier) GetFeedbackDetector() *FeedbackDetector {
	return c.feedbackDetector
}
