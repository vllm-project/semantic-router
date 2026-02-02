package classification

import (
	"fmt"
	"sync"

	candle "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// FactCheckResult represents the result of fact-check classification
type FactCheckResult struct {
	NeedsFactCheck bool    `json:"needs_fact_check"`
	Confidence     float32 `json:"confidence"`
	Label          string  `json:"label"` // "FACT_CHECK_NEEDED" or "NO_FACT_CHECK_NEEDED"
}

// FactCheckClassifier handles fact-check classification to determine if a prompt
// requires external factual verification using the halugate-sentinel ML model
type FactCheckClassifier struct {
	config       *config.FactCheckModelConfig
	mapping      *FactCheckMapping
	initialized  bool
	useMmBERT32K bool // Track if mmBERT-32K is used for inference
	mu           sync.RWMutex
}

// NewFactCheckClassifier creates a new fact-check classifier
func NewFactCheckClassifier(cfg *config.FactCheckModelConfig) (*FactCheckClassifier, error) {
	if cfg == nil {
		return nil, nil // Disabled
	}

	classifier := &FactCheckClassifier{
		config: cfg,
	}

	return classifier, nil
}

// Initialize initializes the fact-check classifier with the halugate-sentinel ML model
func (c *FactCheckClassifier) Initialize() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.initialized {
		return nil
	}

	// Use default mapping (no external mapping file needed)
	c.mapping = &FactCheckMapping{
		LabelToIdx: map[string]int{
			FactCheckLabelNotNeeded: 0,
			FactCheckLabelNeeded:    1,
		},
		IdxToLabel: map[string]string{
			"0": FactCheckLabelNotNeeded,
			"1": FactCheckLabelNeeded,
		},
	}

	// Initialize ML model - ModelID is required
	if c.config.ModelID == "" {
		return fmt.Errorf("fact-check classifier requires ModelID to be configured")
	}

	logging.Infof("✅ Initializing Fact-Check Classifier:")
	logging.Infof("Model: %s", c.config.ModelID)
	logging.Infof("CPU Mode: %v", c.config.UseCPU)

	// Check if mmBERT-32K is configured (takes precedence)
	if c.config.UseMmBERT32K {
		logging.Infof("Type: mmBERT-32K (32K context, YaRN RoPE)")
		err := candle.InitMmBert32KFactcheckClassifier(c.config.ModelID, c.config.UseCPU)
		if err != nil {
			return fmt.Errorf("failed to initialize mmBERT-32K fact-check model from %s: %w", c.config.ModelID, err)
		}
		c.useMmBERT32K = true
		c.initialized = true
		logging.Infof("✓ Fact-check classifier initialized successfully")
		return nil
	}

	logging.Infof("Type: halugate-sentinel (ML-based)")
	err := candle.InitFactCheckClassifier(c.config.ModelID, c.config.UseCPU)
	if err != nil {
		return fmt.Errorf("failed to initialize fact-check ML model from %s: %w", c.config.ModelID, err)
	}

	c.initialized = true
	logging.Infof("✓ Fact-check classifier initialized successfully")

	return nil
}

// Classify determines if a prompt needs fact-checking using the ML model
func (c *FactCheckClassifier) Classify(text string) (*FactCheckResult, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if !c.initialized {
		return nil, fmt.Errorf("fact-check classifier not initialized")
	}

	if text == "" {
		return &FactCheckResult{
			NeedsFactCheck: false,
			Confidence:     1.0,
			Label:          FactCheckLabelNotNeeded,
		}, nil
	}

	var result candle.ClassResult
	var err error
	if c.useMmBERT32K {
		result, err = candle.ClassifyMmBert32KFactcheck(text)
	} else {
		result, err = candle.ClassifyFactCheckText(text)
	}
	if err != nil {
		return nil, fmt.Errorf("fact-check ML classification failed: %w", err)
	}

	// Model outputs: 0=NO_FACT_CHECK_NEEDED, 1=FACT_CHECK_NEEDED
	needsFactCheck := result.Class == 1
	confidence := result.Confidence

	var label string
	if needsFactCheck {
		label = FactCheckLabelNeeded
	} else {
		label = FactCheckLabelNotNeeded
	}

	// Apply threshold check
	threshold := c.config.Threshold
	if threshold <= 0 {
		threshold = 0.7 // Default threshold
	}

	// Only mark as needing fact-check if confidence exceeds threshold
	if needsFactCheck && confidence < threshold {
		// Below threshold, flip decision
		needsFactCheck = false
		label = FactCheckLabelNotNeeded
		confidence = 1.0 - confidence // Invert confidence for the new label
	}

	logging.Debugf("Fact-check ML classification: text_len=%d, needs_fact_check=%v, confidence=%.3f",
		len(text), needsFactCheck, confidence)

	return &FactCheckResult{
		NeedsFactCheck: needsFactCheck,
		Confidence:     confidence,
		Label:          label,
	}, nil
}

// IsInitialized returns whether the classifier is initialized
func (c *FactCheckClassifier) IsInitialized() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.initialized
}

// GetMapping returns the fact-check mapping
func (c *FactCheckClassifier) GetMapping() *FactCheckMapping {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.mapping
}
