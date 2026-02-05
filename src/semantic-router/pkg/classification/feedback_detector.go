package classification

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	candle "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// Default feedback type labels (used as fallback if config.json doesn't have id2label)
const (
	FeedbackLabelSatisfied         = "satisfied"
	FeedbackLabelNeedClarification = "need_clarification"
	FeedbackLabelWrongAnswer       = "wrong_answer"
	FeedbackLabelWantDifferent     = "want_different"
)

// FeedbackResult represents the result of user feedback classification
type FeedbackResult struct {
	FeedbackType string  `json:"feedback_type"` // feedback type label from model's id2label
	Confidence   float32 `json:"confidence"`
	Class        int     `json:"class"` // class index from model
}

// FeedbackMapping maps feedback types to class indices
type FeedbackMapping struct {
	LabelToIdx map[string]int
	IdxToLabel map[string]string
}

// FeedbackDetector handles user feedback classification from follow-up messages
type FeedbackDetector struct {
	config       *config.FeedbackDetectorConfig
	mapping      *FeedbackMapping
	initialized  bool
	useMmBERT32K bool // Track if mmBERT-32K is used for inference
	mu           sync.RWMutex
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

// loadMappingFromConfig loads the id2label mapping from the model's config.json
func (d *FeedbackDetector) loadMappingFromConfig(modelPath string) error {
	configPath := filepath.Join(modelPath, "config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return fmt.Errorf("failed to read config.json: %w", err)
	}

	var configData struct {
		ID2Label map[string]string `json:"id2label"`
		Label2ID map[string]int    `json:"label2id"`
	}

	if err := json.Unmarshal(data, &configData); err != nil {
		return fmt.Errorf("failed to parse config.json: %w", err)
	}

	// Build mapping from config.json
	d.mapping = &FeedbackMapping{
		LabelToIdx: make(map[string]int),
		IdxToLabel: make(map[string]string),
	}

	// Use id2label from config.json and normalize labels
	for idx, label := range configData.ID2Label {
		normalizedLabel := normalizeFeedbackLabel(label)
		d.mapping.IdxToLabel[idx] = normalizedLabel
	}

	// Use label2id from config.json and normalize labels
	for label, idx := range configData.Label2ID {
		normalizedLabel := normalizeFeedbackLabel(label)
		d.mapping.LabelToIdx[normalizedLabel] = idx
	}

	logging.Infof("Loaded feedback mapping from config.json: %d labels", len(d.mapping.IdxToLabel))
	for idx, label := range d.mapping.IdxToLabel {
		logging.Debugf("  %s -> %s", idx, label)
	}

	return nil
}

// normalizeFeedbackLabel converts model labels (e.g., "SAT", "NEED_CLARIFICATION") to standard form
func normalizeFeedbackLabel(label string) string {
	switch strings.ToUpper(label) {
	case "SAT", "SATISFIED":
		return FeedbackLabelSatisfied
	case "NEED_CLARIFICATION":
		return FeedbackLabelNeedClarification
	case "WRONG_ANSWER":
		return FeedbackLabelWrongAnswer
	case "WANT_DIFFERENT":
		return FeedbackLabelWantDifferent
	default:
		return strings.ToLower(label)
	}
}

// Initialize initializes the feedback detector with the ModernBERT model
func (d *FeedbackDetector) Initialize() error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.initialized {
		return nil
	}

	// Initialize ML model - ModelID is required
	if d.config.ModelID == "" {
		return fmt.Errorf("feedback detector requires ModelID to be configured")
	}

	// Load mapping from model's config.json (required - no hardcoded fallback)
	if err := d.loadMappingFromConfig(d.config.ModelID); err != nil {
		return fmt.Errorf("failed to load id2label mapping from %s/config.json: %w", d.config.ModelID, err)
	}

	logging.Infof("💬 Initializing Feedback Detector:")
	logging.Infof("Model: %s", d.config.ModelID)
	logging.Infof("CPU Mode: %v", d.config.UseCPU)

	// Check if mmBERT-32K is configured (takes precedence)
	if d.config.UseMmBERT32K {
		logging.Infof("Type: mmBERT-32K (32K context, YaRN RoPE)")
		err := candle.InitMmBert32KFeedbackClassifier(d.config.ModelID, d.config.UseCPU)
		if err != nil {
			return fmt.Errorf("failed to initialize mmBERT-32K feedback detector from %s: %w", d.config.ModelID, err)
		}
		d.useMmBERT32K = true
		d.initialized = true
		logging.Infof("✓ Feedback detector initialized successfully")
		return nil
	}

	logging.Infof("Type: ModernBERT (ML-based)")
	err := candle.InitFeedbackDetector(d.config.ModelID, d.config.UseCPU)
	if err != nil {
		return fmt.Errorf("failed to initialize feedback detector ML model from %s: %w", d.config.ModelID, err)
	}

	d.initialized = true
	logging.Infof("✓ Feedback detector initialized successfully")

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

	var result candle.ClassResult
	var err error
	if d.useMmBERT32K {
		result, err = candle.ClassifyMmBert32KFeedback(text)
	} else {
		result, err = candle.ClassifyFeedbackText(text)
	}
	if err != nil {
		return nil, fmt.Errorf("feedback detection failed: %w", err)
	}

	// Get feedback type from mapping loaded from config.json
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
