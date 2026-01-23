package classification

import (
	"fmt"
	"sync"

	candle "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// Feedback type labels - matching model's id2label mapping
const (
	FeedbackLabelNeedClarification = "need_clarification" // Model class 0: NEED_CLARIFICATION
	FeedbackLabelSatisfied         = "satisfied"          // Model class 1: SAT
	FeedbackLabelWantDifferent     = "want_different"     // Model class 2: WANT_DIFFERENT
	FeedbackLabelWrongAnswer       = "wrong_answer"       // Model class 3: WRONG_ANSWER
)

// FeedbackResult represents the result of user feedback classification
type FeedbackResult struct {
	FeedbackType string  `json:"feedback_type"` // "satisfied", "need_clarification", "wrong_answer", "want_different"
	Confidence   float32 `json:"confidence"`
	Class        int     `json:"class"` // 0=SAT, 1=NEED_CLARIFICATION, 2=WRONG_ANSWER, 3=WANT_DIFFERENT
}

// FeedbackMapping maps feedback types to class indices
type FeedbackMapping struct {
	LabelToIdx map[string]int
	IdxToLabel map[string]string
}

// FeedbackDetector handles user feedback classification from follow-up messages
type FeedbackDetector struct {
	config      *config.FeedbackDetectorConfig
	mapping     *FeedbackMapping
	initialized bool
	mu          sync.RWMutex
}

// NewFeedbackDetector creates a new feedback detector
func NewFeedbackDetector(cfg *config.FeedbackDetectorConfig) (*FeedbackDetector, error) {
	if cfg == nil {
		return nil, nil // Disabled
	}

	detector := &FeedbackDetector{
		config: cfg,
	}

	return detector, nil
}

// Initialize initializes the feedback detector with the ModernBERT model
func (d *FeedbackDetector) Initialize() error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.initialized {
		return nil
	}

	// Use default mapping based on model outputs
	// Model's id2label: 0=NEED_CLARIFICATION, 1=SAT, 2=WANT_DIFFERENT, 3=WRONG_ANSWER
	d.mapping = &FeedbackMapping{
		LabelToIdx: map[string]int{
			FeedbackLabelNeedClarification: 0,
			FeedbackLabelSatisfied:         1,
			FeedbackLabelWantDifferent:     2,
			FeedbackLabelWrongAnswer:       3,
		},
		IdxToLabel: map[string]string{
			"0": FeedbackLabelNeedClarification,
			"1": FeedbackLabelSatisfied,
			"2": FeedbackLabelWantDifferent,
			"3": FeedbackLabelWrongAnswer,
		},
	}

	// Initialize ML model - ModelID is required
	if d.config.ModelID == "" {
		return fmt.Errorf("feedback detector requires ModelID to be configured")
	}

	logging.Infof("Initializing feedback detector ML model from: %s", d.config.ModelID)

	err := candle.InitFeedbackDetector(d.config.ModelID, d.config.UseCPU)
	if err != nil {
		return fmt.Errorf("failed to initialize feedback detector ML model from %s: %w", d.config.ModelID, err)
	}

	d.initialized = true
	logging.Infof("Feedback detector initialized with ML model")

	return nil
}

// Classify determines user feedback type from follow-up message using the ML model
func (d *FeedbackDetector) Classify(text string) (*FeedbackResult, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	if !d.initialized {
		return nil, fmt.Errorf("feedback detector not initialized")
	}

	if text == "" {
		return &FeedbackResult{
			FeedbackType: FeedbackLabelSatisfied,
			Confidence:   1.0,
			Class:        0,
		}, nil
	}

	result, err := candle.ClassifyFeedbackText(text)
	if err != nil {
		return nil, fmt.Errorf("feedback detection failed: %w", err)
	}

	// Model outputs: 0=SAT, 1=NEED_CLARIFICATION, 2=WRONG_ANSWER, 3=WANT_DIFFERENT
	feedbackType := d.mapping.IdxToLabel[fmt.Sprintf("%d", result.Class)]
	if feedbackType == "" {
		feedbackType = FeedbackLabelSatisfied // Default fallback
	}

	confidence := result.Confidence

	// Apply threshold check
	threshold := d.config.Threshold
	if threshold <= 0 {
		threshold = 0.5 // Default threshold
	}

	// If confidence is below threshold, mark as uncertain (default to satisfied)
	if confidence < threshold {
		feedbackType = FeedbackLabelSatisfied
		confidence = 1.0 - confidence
	}

	logging.Debugf("Feedback detection: text_len=%d, feedback_type=%s, confidence=%.3f",
		len(text), feedbackType, confidence)

	return &FeedbackResult{
		FeedbackType: feedbackType,
		Confidence:   confidence,
		Class:        result.Class,
	}, nil
}

// IsInitialized returns whether the detector is initialized
func (d *FeedbackDetector) IsInitialized() bool {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.initialized
}

// GetMapping returns the feedback mapping
func (d *FeedbackDetector) GetMapping() *FeedbackMapping {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.mapping
}
