package classification

import (
	"fmt"
	"strings"

	candle "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// NLILabel is an alias for candle.NLILabel.
type NLILabel = candle.NLILabel

const (
	// NLIEntailment means the premise supports the hypothesis.
	NLIEntailment = candle.NLIEntailment
	// NLINeutral means the premise neither supports nor contradicts.
	NLINeutral = candle.NLINeutral
	// NLIContradiction means the premise contradicts the hypothesis.
	NLIContradiction = candle.NLIContradiction
	// NLIError means an error occurred during classification.
	NLIError = candle.NLIError
)

// EnhancedHallucinationSpan represents a hallucinated span with NLI explanation.
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

// EnhancedHallucinationResult represents hallucination detection with NLI explanations.
type EnhancedHallucinationResult struct {
	HallucinationDetected bool                        `json:"hallucination_detected"`
	Confidence            float32                     `json:"confidence"`
	Spans                 []EnhancedHallucinationSpan `json:"spans,omitempty"`
}

// NLIResult represents the result of NLI classification.
type NLIResult struct {
	Label          NLILabel `json:"label"`
	LabelStr       string   `json:"label_str"`
	Confidence     float32  `json:"confidence"`
	EntailmentProb float32  `json:"entailment_prob"`
	NeutralProb    float32  `json:"neutral_prob"`
	ContradictProb float32  `json:"contradiction_prob"`
}

// SetNLIConfig sets the NLI model configuration for enhanced detection.
// Recommended model: tasksource/ModernBERT-base-nli.
func (d *HallucinationDetector) SetNLIConfig(cfg *config.NLIModelConfig) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.nliConfig = cfg
}

// InitializeNLI initializes the NLI model for enhanced hallucination detection.
func (d *HallucinationDetector) InitializeNLI() error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.nliInitialized {
		return nil
	}

	if d.nliConfig == nil || d.nliConfig.ModelID == "" {
		return fmt.Errorf("NLI model config not set")
	}

	err := candle.InitNLIModel(d.nliConfig.ModelID, d.nliConfig.UseCPU)
	if err != nil {
		return fmt.Errorf("failed to initialize NLI model from %s: %w", d.nliConfig.ModelID, err)
	}

	d.nliInitialized = true
	logging.ComponentEvent("classifier", "hallucination_nli_initialized", map[string]interface{}{
		"backend":   "candle",
		"model_ref": d.nliConfig.ModelID,
	})

	return nil
}

// IsNLIInitialized returns whether the NLI model is initialized.
func (d *HallucinationDetector) IsNLIInitialized() bool {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.nliInitialized
}

// ClassifyNLI classifies the relationship between premise and hypothesis.
// Returns: ENTAILMENT (supports), NEUTRAL (can't verify), CONTRADICTION (conflicts).
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

// DetectWithNLI detects hallucinations and provides NLI-based explanations.
// It combines token-level hallucination detection with NLI classification.
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

	hallucinationThreshold := d.hallucinationThreshold()
	nliThreshold := d.nliThreshold()
	candleResult, err := candle.DetectHallucinationsWithNLI(context, question, answer, hallucinationThreshold)
	if err != nil {
		return nil, fmt.Errorf("enhanced hallucination detection error: %w", err)
	}

	result := &EnhancedHallucinationResult{
		HallucinationDetected: candleResult.HasHallucination,
		Confidence:            candleResult.Confidence,
		Spans:                 []EnhancedHallucinationSpan{},
	}

	filteredCount := 0
	for _, span := range candleResult.Spans {
		enhancedSpan, ok := d.convertEnhancedHallucinationSpan(span, nliThreshold)
		if !ok {
			filteredCount++
			continue
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

func (d *HallucinationDetector) hallucinationThreshold() float32 {
	threshold := d.config.Threshold
	if threshold <= 0 {
		return 0.5
	}
	return threshold
}

func (d *HallucinationDetector) nliThreshold() float32 {
	if d.nliConfig != nil && d.nliConfig.Threshold > 0 {
		return d.nliConfig.Threshold
	}
	return 0.7
}

func (d *HallucinationDetector) nliEntailmentThreshold() float32 {
	threshold := d.config.NLIEntailmentThreshold
	if threshold <= 0 {
		return 0.75
	}
	return threshold
}

func (d *HallucinationDetector) convertEnhancedHallucinationSpan(span candle.EnhancedHallucinationSpan, nliThreshold float32) (EnhancedHallucinationSpan, bool) {
	minSpanLen := d.config.MinSpanLength
	if minSpanLen <= 0 {
		minSpanLen = 1
	}
	minSpanConfidence := d.config.MinSpanConfidence
	if minSpanConfidence < 0 {
		minSpanConfidence = 0.0
	}

	spanTokensLen := len(strings.Fields(span.Text))
	if spanTokensLen < minSpanLen {
		logging.Debugf("Filtered span (too short): '%s' (%d tokens < %d)",
			span.Text, spanTokensLen, minSpanLen)
		return EnhancedHallucinationSpan{}, false
	}
	if span.HallucinationConfidence < minSpanConfidence {
		logging.Debugf("Filtered span (low confidence): '%s' (%.3f < %.3f)",
			span.Text, span.HallucinationConfidence, minSpanConfidence)
		return EnhancedHallucinationSpan{}, false
	}
	if d.config.EnableNLIFiltering && span.NLILabel == NLIEntailment && span.NLIConfidence >= d.nliEntailmentThreshold() {
		logging.Debugf("Filtered span (NLI entailment): '%s' (entailment confidence %.3f >= %.3f)",
			span.Text, span.NLIConfidence, d.nliEntailmentThreshold())
		return EnhancedHallucinationSpan{}, false
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
	if span.NLIConfidence < nliThreshold {
		if enhancedSpan.Severity > 0 {
			enhancedSpan.Severity--
		}
		enhancedSpan.Explanation = fmt.Sprintf("%s (NLI confidence %.0f%% below threshold %.0f%%)",
			span.Explanation, span.NLIConfidence*100, nliThreshold*100)
	}
	return enhancedSpan, true
}
