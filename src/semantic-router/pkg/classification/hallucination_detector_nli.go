package classification

import (
	"context"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// NLILabel is the engine-neutral premise/hypothesis relation.
type NLILabel = tasks.NLILabel

const (
	// NLIEntailment means the premise supports the hypothesis.
	NLIEntailment = tasks.NLIEntailment
	// NLINeutral means the premise neither supports nor contradicts.
	NLINeutral = tasks.NLINeutral
	// NLIContradiction means the premise contradicts the hypothesis.
	NLIContradiction = tasks.NLIContradiction
	// NLIUnknown means no NLI judgment is available (e.g. the endpoint backend,
	// which does not produce NLI labels).
	NLIUnknown = tasks.NLIUnknown
	// NLIError means an error occurred during classification.
	NLIError = tasks.NLIError
)

// EnhancedHallucinationSpan represents a hallucinated span with NLI explanation.
type EnhancedHallucinationSpan struct {
	Text                    string   `json:"text"`
	Start                   int      `json:"start"`
	End                     int      `json:"end"`
	HallucinationConfidence float32  `json:"hallucination_confidence,omitempty"`
	ScoreAvailable          bool     `json:"score_available"`
	NLILabel                NLILabel `json:"nli_label"`
	NLILabelStr             string   `json:"nli_label_str"`
	NLIConfidence           float32  `json:"nli_confidence"`
	Severity                int      `json:"severity"` // 0-4: 0=low, 4=critical
	Explanation             string   `json:"explanation"`
}

// EnhancedHallucinationResult represents hallucination detection with NLI explanations.
type EnhancedHallucinationResult struct {
	HallucinationDetected bool                        `json:"hallucination_detected"`
	Confidence            float32                     `json:"confidence,omitempty"`
	ScoreAvailable        bool                        `json:"score_available"`
	ScoreKind             string                      `json:"score_kind,omitempty"`
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

	if d.models == nil {
		d.models = standaloneModelRuntime()
	}
	d.nliSpec = d.models.localSpec("hallucination_explainer", d.nliConfig.ModelID, "modernbert", "text_pair_distribution.v1", d.nliConfig.UseCPU)
	handle, err := d.models.runtime.TextPair(context.Background(), d.nliSpec)
	if err != nil {
		return err
	}
	d.nliHandle = handle

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
func (d *HallucinationDetector) ClassifyNLI(ctx context.Context, premise, hypothesis string) (*NLIResult, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	if !d.nliInitialized {
		return nil, fmt.Errorf("NLI model not initialized")
	}

	return d.classifyNLILocked(ctx, premise, hypothesis)
}

func (d *HallucinationDetector) classifyNLILocked(ctx context.Context, premise, hypothesis string) (*NLIResult, error) {
	if d.nliHandle == nil {
		return nil, fmt.Errorf("NLI model not initialized")
	}
	distribution, err := d.nliHandle.Call(ctx, string(d.nliSpec.Recipe), tasks.TextPairRequest{Premise: premise, Hypothesis: hypothesis})
	if err != nil {
		return nil, err
	}
	if len(distribution.Probabilities) != 3 {
		return nil, fmt.Errorf("NLI requires entailment/neutral/contradiction probabilities")
	}
	class, confidence := deriveArgmax(distribution.Probabilities)
	label := NLILabel(class)
	return &NLIResult{Label: label, LabelStr: label.String(), Confidence: confidence, EntailmentProb: distribution.Probabilities[0], NeutralProb: distribution.Probabilities[1], ContradictProb: distribution.Probabilities[2]}, nil
}

// DetectWithNLI composes two owned typed tasks. No global detector or NLI slot
// can be changed by a candidate generation while this request is running.
func (d *HallucinationDetector) DetectWithNLI(ctx context.Context, contextText, question, answer string) (*EnhancedHallucinationResult, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()
	spans, err := d.detectSpans(ctx, contextText, question, answer)
	if err != nil {
		return nil, err
	}
	result := &EnhancedHallucinationResult{HallucinationDetected: len(spans.Entities) > 0, Spans: []EnhancedHallucinationSpan{}}
	if spans.Summary != nil {
		result.Confidence = float32(spans.Summary.Value)
		result.ScoreAvailable = true
	}
	if spans.SummarySemantics != nil {
		result.ScoreKind = spans.SummarySemantics.Unit
	}
	for _, span := range spans.Entities {
		enhanced := EnhancedHallucinationSpan{Text: span.Text, Start: span.Start, End: span.End, HallucinationConfidence: span.Confidence, ScoreAvailable: spans.HasScores(), NLILabel: NLIUnknown, NLILabelStr: NLIUnknown.String(), Severity: 2, Explanation: "Unsupported span detected"}
		if spans.HasScores() {
			enhanced.Explanation = fmt.Sprintf("Unsupported claim detected (token score: %.1f%%)", span.Confidence*100)
			if span.Confidence > 0.8 {
				enhanced.Severity = 3
			}
		}
		if d.nliInitialized {
			nli, err := d.classifyNLILocked(ctx, contextText+" "+question, span.Text)
			if err != nil {
				return nil, fmt.Errorf("explain hallucination span: %w", err)
			}
			enhanced.NLILabel = nli.Label
			enhanced.NLILabelStr = nli.LabelStr
			enhanced.NLIConfidence = nli.Confidence
			switch nli.Label {
			case NLIContradiction:
				enhanced.Severity = 4
				enhanced.Explanation = "CONTRADICTION: This claim directly conflicts with the provided context"
			case NLINeutral:
				enhanced.Severity = 2
				enhanced.Explanation = "FABRICATION: This claim is not supported by the provided context"
			case NLIEntailment:
				enhanced.Severity = 1
				enhanced.Explanation = "UNCERTAIN: Hallucination detector flagged this but NLI suggests it may be supported"
			}
			enhanced.Explanation += fmt.Sprintf(" (confidence: %.1f%%)", nli.Confidence*100)
		}
		converted, ok := d.convertEnhancedHallucinationSpan(enhanced, d.nliThreshold())
		if ok {
			result.Spans = append(result.Spans, converted)
		}
	}
	result.HallucinationDetected = len(result.Spans) > 0
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

func (d *HallucinationDetector) convertEnhancedHallucinationSpan(span EnhancedHallucinationSpan, nliThreshold float32) (EnhancedHallucinationSpan, bool) {
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
	if span.ScoreAvailable && span.HallucinationConfidence < minSpanConfidence {
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
		ScoreAvailable:          span.ScoreAvailable,
		NLILabel:                span.NLILabel,
		NLILabelStr:             span.NLILabelStr,
		NLIConfidence:           span.NLIConfidence,
		Severity:                span.Severity,
		Explanation:             span.Explanation,
	}
	if span.NLILabel != NLIUnknown && span.NLIConfidence < nliThreshold {
		if enhancedSpan.Severity > 0 {
			enhancedSpan.Severity--
		}
		enhancedSpan.Explanation = fmt.Sprintf("%s (NLI confidence %.0f%% below threshold %.0f%%)",
			span.Explanation, span.NLIConfidence*100, nliThreshold*100)
	}
	return enhancedSpan, true
}
