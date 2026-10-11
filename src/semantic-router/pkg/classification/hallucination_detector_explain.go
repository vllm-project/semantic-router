package classification

import (
	"context"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// EnhancedHallucinationSpan is one unsupported span with its offsets, score,
// severity and explanation.
type EnhancedHallucinationSpan struct {
	Text                    string  `json:"text"`
	Start                   int     `json:"start"`
	End                     int     `json:"end"`
	Label                   string  `json:"label,omitempty"`
	HallucinationConfidence float32 `json:"hallucination_confidence,omitempty"`
	ScoreAvailable          bool    `json:"score_available"`
	Severity                int     `json:"severity"` // 0-4: 0=low, 4=critical
	Explanation             string  `json:"explanation"`
}

// EnhancedHallucinationResult is a detection with span details.
type EnhancedHallucinationResult struct {
	HallucinationDetected bool                        `json:"hallucination_detected"`
	Confidence            float32                     `json:"confidence,omitempty"`
	ScoreAvailable        bool                        `json:"score_available"`
	ScoreKind             string                      `json:"score_kind,omitempty"`
	Spans                 []EnhancedHallucinationSpan `json:"spans,omitempty"`
}

// DetectWithExplanations returns the unsupported spans with a severity and a
// plain explanation each.
func (d *HallucinationDetector) DetectWithExplanations(ctx context.Context, contextText, question, answer string) (*EnhancedHallucinationResult, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()
	if d.judgment != nil {
		verdict, err := d.detectJudgment(ctx, contextText, question, answer)
		if err != nil {
			return nil, err
		}
		return &EnhancedHallucinationResult{HallucinationDetected: verdict.HallucinationDetected, Confidence: verdict.Confidence, ScoreAvailable: verdict.ScoreAvailable, ScoreKind: verdict.ScoreKind}, nil
	}
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
		enhanced := EnhancedHallucinationSpan{Text: span.Text, Start: span.Start, End: span.End, Label: span.EntityType, HallucinationConfidence: span.Confidence, ScoreAvailable: spans.HasScores(), Severity: 2, Explanation: "Unsupported span detected"}
		if spans.HasScores() {
			enhanced.Explanation = fmt.Sprintf("Unsupported claim detected (token score: %.1f%%)", span.Confidence*100)
			if span.Confidence > 0.8 {
				enhanced.Severity = 3
			}
		}
		converted, ok := d.convertEnhancedHallucinationSpan(enhanced)
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

func (d *HallucinationDetector) convertEnhancedHallucinationSpan(span EnhancedHallucinationSpan) (EnhancedHallucinationSpan, bool) {
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
	enhancedSpan := EnhancedHallucinationSpan{
		Text:                    span.Text,
		Start:                   span.Start,
		End:                     span.End,
		HallucinationConfidence: span.HallucinationConfidence,
		ScoreAvailable:          span.ScoreAvailable,
		Severity:                span.Severity,
		Explanation:             span.Explanation,
	}
	return enhancedSpan, true
}
