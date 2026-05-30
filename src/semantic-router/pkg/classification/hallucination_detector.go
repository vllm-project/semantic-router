package classification

import (
	"fmt"
	"strings"
	"sync"

	candle "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// HallucinationResult represents the result of hallucination detection
type HallucinationResult struct {
	HallucinationDetected bool     `json:"hallucination_detected"`
	Confidence            float32  `json:"confidence"`
	UnsupportedSpans      []string `json:"unsupported_spans,omitempty"` // Text spans not grounded in context
	SupportedSpans        []string `json:"supported_spans,omitempty"`   // Text spans grounded in context
}

// HallucinationDetector handles hallucination detection
// It checks if an LLM answer contains claims that are not supported by the provided context
type HallucinationDetector struct {
	config         *config.HallucinationModelConfig
	nliConfig      *config.NLIModelConfig // NLI model configuration for enhanced detection
	initialized    bool
	nliInitialized bool
	mu             sync.RWMutex
}

// NewHallucinationDetector creates a new hallucination detector
func NewHallucinationDetector(cfg *config.HallucinationModelConfig) (*HallucinationDetector, error) {
	if cfg == nil {
		return nil, fmt.Errorf("hallucination model config is required")
	}

	if cfg.ModelID == "" {
		return nil, fmt.Errorf("hallucination model_id is required")
	}

	detector := &HallucinationDetector{
		config: cfg,
	}

	return detector, nil
}

// Initialize initializes the hallucination detection model via Candle bindings
func (d *HallucinationDetector) Initialize() error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.initialized {
		return nil
	}

	err := candle.InitHallucinationModel(d.config.ModelID, d.config.UseCPU)
	if err != nil {
		return fmt.Errorf("failed to initialize hallucination detection model from %s: %w", d.config.ModelID, err)
	}

	d.initialized = true
	logging.ComponentEvent("classifier", "hallucination_detector_initialized", map[string]interface{}{
		"backend":   "candle",
		"model_ref": d.config.ModelID,
	})

	return nil
}

// Detect checks if an answer contains hallucinations given the context
// context: The tool results or RAG context that should ground the answer
// question: The original user question
// answer: The LLM-generated answer to verify
func (d *HallucinationDetector) Detect(context, question, answer string) (*HallucinationResult, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	if !d.initialized {
		return nil, fmt.Errorf("hallucination detection model not initialized")
	}

	if answer == "" {
		return &HallucinationResult{
			HallucinationDetected: false,
			Confidence:            1.0,
		}, nil
	}

	if context == "" {
		return nil, fmt.Errorf("context is required for hallucination detection")
	}

	// Get threshold from config (default 0.5)
	threshold := d.config.Threshold
	if threshold <= 0 {
		threshold = 0.5
	}

	// Call hallucination detection via candle bindings with threshold
	// Threshold is applied at token level in Rust - only tokens with confidence >= threshold
	// are considered hallucinated and included in spans
	candleResult, err := candle.DetectHallucinations(context, question, answer, threshold)
	if err != nil {
		return nil, fmt.Errorf("hallucination detection error: %w", err)
	}

	// Convert result
	result := &HallucinationResult{
		HallucinationDetected: candleResult.HasHallucination,
		Confidence:            candleResult.Confidence,
		UnsupportedSpans:      []string{},
		SupportedSpans:        []string{},
	}

	minSpanLength := d.config.MinSpanLength
	if minSpanLength <= 0 {
		minSpanLength = 1 // Default minimum span length
	}

	minSpanConfidence := d.config.MinSpanConfidence
	if minSpanConfidence < 0 {
		minSpanConfidence = 0.0 // Default minimum span confidence
	}

	// Extract hallucinated spans (already filtered by threshold in Rust)
	for _, span := range candleResult.Spans {
		spanTokensLen := len(strings.Fields(span.Text))

		// Skip spans below minimum length
		if spanTokensLen < minSpanLength {
			logging.Debugf("Filtered span (too short): '%s' (%d tokens < %d)",
				span.Text, spanTokensLen, minSpanLength)
			continue
		}

		// Skip spans below confidence threshold
		if span.Confidence < minSpanConfidence {
			logging.Debugf("Filtered span (low confidence): '%s' (%.3f < %.3f)",
				span.Text, span.Confidence, minSpanConfidence)
			continue
		}
		result.UnsupportedSpans = append(result.UnsupportedSpans, span.Text)
	}

	if len(result.UnsupportedSpans) == 0 && len(candleResult.Spans) > 0 {
		result.HallucinationDetected = false
	}
	logging.Debugf("Hallucination detection: hallucination=%v, confidence=%.3f, threshold=%.3f, spans=%d",
		result.HallucinationDetected, result.Confidence, threshold, len(result.UnsupportedSpans))

	return result, nil
}

// IsInitialized returns whether the detector is initialized
func (d *HallucinationDetector) IsInitialized() bool {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.initialized
}
