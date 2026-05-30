package classification

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// IsFeedbackDetectorEnabled checks if feedback detection is enabled and properly configured.
func (c *Classifier) IsFeedbackDetectorEnabled() bool {
	return c.Config.IsFeedbackDetectorEnabled()
}

// initializeFeedbackDetector initializes the feedback detection model.
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

// ClassifyFeedback performs user feedback classification on the given text.
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

// GetFeedbackDetector returns the feedback detector instance.
func (c *Classifier) GetFeedbackDetector() *FeedbackDetector {
	return c.feedbackDetector
}
