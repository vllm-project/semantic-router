package classification

import (
	"fmt"
	"strings"
	"sync"

	candle "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// NLILabel is an alias for candle.NLILabel
type NLILabel = candle.NLILabel

const (
	// NLIEntailment means the premise supports the hypothesis
	NLIEntailment = candle.NLIEntailment
	// NLINeutral means the premise neither supports nor contradicts
	NLINeutral = candle.NLINeutral
	// NLIContradiction means the premise contradicts the hypothesis
	NLIContradiction = candle.NLIContradiction
	// NLIError means an error occurred during classification
	NLIError = candle.NLIError
)

// HallucinationResult represents the result of hallucination detection
type HallucinationResult struct {
	HallucinationDetected bool     `json:"hallucination_detected"`
	Confidence            float32  `json:"confidence"`
	UnsupportedSpans      []string `json:"unsupported_spans,omitempty"` // Text spans not grounded in context
	SupportedSpans        []string `json:"supported_spans,omitempty"`   // Text spans grounded in context
}

// EnhancedHallucinationSpan represents a hallucinated span with NLI explanation
type EnhancedHallucinationSpan struct {
	Text                    string   `json:"text"`
	Start                   int      `json:"start"`
	End                     int      `json:"end"`
	HallucinationConfidence float32  `json:"hallucination_confidence"`
	NLILabel                NLILabel `json:"nli_label"`
	NLILabelStr             string   `json:"nli_label_str"`
	NLIConfidence           float32  `json:"nli_confidence"`
	Severity                int      `json:"severity"` // 0-4: 0=low, 4=critical
	Explanation             string   `json:"explanation"`
}

// EnhancedHallucinationResult represents hallucination detection with NLI explanations
type EnhancedHallucinationResult struct {
	HallucinationDetected bool                        `json:"hallucination_detected"`
	Confidence            float32                     `json:"confidence"`
	Spans                 []EnhancedHallucinationSpan `json:"spans,omitempty"`
}

// NLIResult represents the result of NLI classification
type NLIResult struct {
	Label          NLILabel `json:"label"`
	LabelStr       string   `json:"label_str"`
	Confidence     float32  `json:"confidence"`
	EntailmentProb float32  `json:"entailment_prob"`
	NeutralProb    float32  `json:"neutral_prob"`
	ContradictProb float32  `json:"contradiction_prob"`
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

	logging.Infof("Initializing hallucination detection model from: %s", d.config.ModelID)

	err := candle.InitHallucinationModel(d.config.ModelID, d.config.UseCPU)
	if err != nil {
		return fmt.Errorf("failed to initialize hallucination detection model from %s: %w", d.config.ModelID, err)
	}

	d.initialized = true
	logging.Infof("Hallucination detection model initialized successfully")

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

// SetNLIConfig sets the NLI model configuration for enhanced detection
// Recommended model: tasksource/ModernBERT-base-nli
func (d *HallucinationDetector) SetNLIConfig(cfg *config.NLIModelConfig) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.nliConfig = cfg
}

// InitializeNLI initializes the NLI model for enhanced hallucination detection
func (d *HallucinationDetector) InitializeNLI() error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.nliInitialized {
		return nil
	}

	if d.nliConfig == nil || d.nliConfig.ModelID == "" {
		return fmt.Errorf("NLI model config not set")
	}

	logging.Infof("Initializing NLI model from: %s", d.nliConfig.ModelID)

	err := candle.InitNLIModel(d.nliConfig.ModelID, d.nliConfig.UseCPU)
	if err != nil {
		return fmt.Errorf("failed to initialize NLI model from %s: %w", d.nliConfig.ModelID, err)
	}

	d.nliInitialized = true
	logging.Infof("NLI model initialized successfully")

	return nil
}

// IsNLIInitialized returns whether the NLI model is initialized
func (d *HallucinationDetector) IsNLIInitialized() bool {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.nliInitialized
}

// ClassifyNLI classifies the relationship between premise and hypothesis
// Returns: ENTAILMENT (supports), NEUTRAL (can't verify), CONTRADICTION (conflicts)
func (d *HallucinationDetector) ClassifyNLI(premise, hypothesis string) (*NLIResult, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	if !d.nliInitialized {
		return nil, fmt.Errorf("NLI model not initialized")
	}

	candleResult, err := candle.ClassifyNLI(premise, hypothesis)
	if err != nil {
		return nil, fmt.Errorf("NLI classification error: %w", err)
	}

	return &NLIResult{
		Label:          candleResult.Label,
		LabelStr:       candleResult.LabelStr,
		Confidence:     candleResult.Confidence,
		EntailmentProb: candleResult.EntailmentProb,
		NeutralProb:    candleResult.NeutralProb,
		ContradictProb: candleResult.ContradictProb,
	}, nil
}

// DetectWithNLI detects hallucinations and provides NLI-based explanations
// This combines token-level hallucination detection with NLI classification
// to provide detailed explanations for each hallucinated span
func (d *HallucinationDetector) DetectWithNLI(context, question, answer string) (*EnhancedHallucinationResult, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	if !d.initialized {
		return nil, fmt.Errorf("hallucination detection model not initialized")
	}

	if answer == "" {
		return &EnhancedHallucinationResult{
			HallucinationDetected: false,
			Confidence:            1.0,
			Spans:                 []EnhancedHallucinationSpan{},
		}, nil
	}

	if context == "" {
		return nil, fmt.Errorf("context is required for hallucination detection")
	}

	// Get thresholds from config
	hallucinationThreshold := d.config.Threshold
	if hallucinationThreshold <= 0 {
		hallucinationThreshold = 0.5 // Default hallucination threshold
	}

	nliThreshold := float32(0.7) // Default NLI threshold
	if d.nliConfig != nil && d.nliConfig.Threshold > 0 {
		nliThreshold = d.nliConfig.Threshold
	}

	// Call enhanced detection (hallucination model + NLI) via candle bindings
	// Hallucination threshold is applied at token level in Rust
	candleResult, err := candle.DetectHallucinationsWithNLI(context, question, answer, hallucinationThreshold)
	if err != nil {
		return nil, fmt.Errorf("enhanced hallucination detection error: %w", err)
	}

	// Convert result
	result := &EnhancedHallucinationResult{
		HallucinationDetected: candleResult.HasHallucination,
		Confidence:            candleResult.Confidence,
		Spans:                 []EnhancedHallucinationSpan{},
	}

	minSpanLen := d.config.MinSpanLength
	if minSpanLen <= 0 {
		minSpanLen = 1 // Default minimum span length
	}

	minSpanConfidence := d.config.MinSpanConfidence
	if minSpanConfidence < 0 {
		minSpanConfidence = 0.0 // Default minimum span confidence
	}
	enableNLIFiltering := d.config.EnableNLIFiltering
	nliEntailmentTheshold := d.config.NLIEntailmentThreshold
	if nliEntailmentTheshold <= 0 {
		nliEntailmentTheshold = 0.75 // Default entailment threshold
	}

	// Convert enanced spans with NLI filtering
	filteredCount := 0
	// Convert enhanced spans, adjusting severity based on NLI threshold
	for _, span := range candleResult.Spans {
		spanTokensLen := len(strings.Fields(span.Text))

		// Skip spans below minimum length
		if spanTokensLen < minSpanLen {
			logging.Debugf("Filtered span (too short): '%s' (%d tokens < %d)",
				span.Text, spanTokensLen, minSpanLen)
			filteredCount++
			continue
		}

		// Skip spans below confidence threshold
		if span.HallucinationConfidence < minSpanConfidence {
			logging.Debugf("Filtered span (low confidence): '%s' (%.3f < %.3f)",
				span.Text, span.HallucinationConfidence, minSpanConfidence)
			filteredCount++
			continue
		}

		// If NLI filtering is enabled, skip spans with high entailment confidence
		if enableNLIFiltering && span.NLILabel == NLIEntailment && span.NLIConfidence >= nliEntailmentTheshold {
			logging.Debugf("Filtered span (NLI entailment): '%s' (entailment confidence %.3f >= %.3f)",
				span.Text, span.NLIConfidence, nliEntailmentTheshold)
			filteredCount++
			continue
		}

		enhancedSpan := EnhancedHallucinationSpan{
			Text:                    span.Text,
			Start:                   span.Start,
			End:                     span.End,
			HallucinationConfidence: span.HallucinationConfidence,
			NLILabel:                span.NLILabel,
			NLILabelStr:             span.NLILabelStr,
			NLIConfidence:           span.NLIConfidence,
			Severity:                span.Severity,
			Explanation:             span.Explanation,
		}

		// If NLI confidence is below threshold, reduce severity and update explanation
		if span.NLIConfidence < nliThreshold {
			// Reduce severity by 1 (minimum 0)
			if enhancedSpan.Severity > 0 {
				enhancedSpan.Severity--
			}
			enhancedSpan.Explanation = fmt.Sprintf("%s (NLI confidence %.0f%% below threshold %.0f%%)",
				span.Explanation, span.NLIConfidence*100, nliThreshold*100)
		}

		result.Spans = append(result.Spans, enhancedSpan)
	}

	if len(result.Spans) == 0 && len(candleResult.Spans) > 0 {
		result.HallucinationDetected = false
		logging.Infof("All %d spans filtered out - marking as no hallucination", filteredCount)
	}

	logging.Debugf("Enhanced hallucination detection: detected=%v, confidence=%.3f, hal_threshold=%.3f, nli_threshold=%.3f, spans=%d",
		result.HallucinationDetected, result.Confidence, hallucinationThreshold, nliThreshold, len(result.Spans))

	return result, nil
}
