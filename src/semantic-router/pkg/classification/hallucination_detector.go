package classification

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/admission"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// HallucinationResult retains the detector verdict and its optional aggregate
// score. Confidence is a historical field name; ScoreKind declares its units.
type HallucinationResult struct {
	HallucinationDetected bool     `json:"hallucination_detected"`
	Confidence            float32  `json:"confidence,omitempty"`
	ScoreAvailable        bool     `json:"score_available"`
	ScoreKind             string   `json:"score_kind,omitempty"`
	UnsupportedSpans      []string `json:"unsupported_spans,omitempty"`
	SupportedSpans        []string `json:"supported_spans,omitempty"`
}

const (
	hallucinationAnswerChunkBudget  = 256 * 4
	hallucinationAnswerOverlapRunes = 64
)

func hallucinationAnswerChunks(answer string) []string {
	return securitySignalChunks(answer, hallucinationAnswerChunkBudget, hallucinationAnswerOverlapRunes)
}

func mergeChunkConfidence(merged bool, mergedConfidence float32, chunk bool, chunkConfidence float32) float32 {
	switch {
	case chunk && (!merged || chunkConfidence > mergedConfidence):
		return chunkConfidence
	case !chunk && !merged && chunkConfidence < mergedConfidence:
		return chunkConfidence
	}
	return mergedConfidence
}

type HallucinationDetector struct {
	config         *config.HallucinationModelConfig
	nliConfig      *config.NLIModelConfig
	models         *classifierModelRuntime
	spec           config.ResolvedModelBinding
	nliSpec        config.ResolvedModelBinding
	handle         *binding.Resolved[tasks.GroundedTextRequest, tasks.TokenClassificationResult]
	nliHandle      *binding.Resolved[tasks.TextPairRequest, tasks.LabelDistribution]
	initialized    bool
	nliInitialized bool
	gate           admission.Admissioner
	explainerGate  admission.Admissioner
	mu             sync.RWMutex
}

func (d *HallucinationDetector) SetAdmissioners(detector, explainer admission.Admissioner) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.gate = detector
	d.explainerGate = explainer
}

func NewHallucinationDetector(cfg *config.HallucinationModelConfig, models ...*classifierModelRuntime) (*HallucinationDetector, error) {
	if cfg == nil {
		return nil, fmt.Errorf("hallucination model config is required")
	}
	runtime := consumerModelRuntime(models)
	spec := runtime.localSpec("hallucination_detector", cfg.ModelID, "modernbert", config.RemoteClassifierContractTokenSpans, cfg.UseCPU)
	if spec.Deployment.Artifact == "" {
		return nil, fmt.Errorf("hallucination model_id is required")
	}
	return &HallucinationDetector{config: cfg, models: runtime, spec: spec}, nil
}

func (d *HallucinationDetector) Initialize() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.initialized {
		return nil
	}
	handle, err := d.models.runtime.Grounded(context.Background(), d.spec, d.hallucinationThreshold())
	if err != nil {
		return err
	}
	d.handle = handle
	d.initialized = true
	return nil
}

// detectSpans keeps offsets in the complete answer, including overlapping long
// answer windows. A failed/partial call remains an error, never a clean verdict.
func (d *HallucinationDetector) detectSpans(ctx context.Context, contextText, question, answer string) (tasks.TokenClassificationResult, error) {
	var merged tasks.TokenClassificationResult
	if !d.initialized || d.handle == nil {
		return merged, fmt.Errorf("hallucination detection model not initialized")
	}
	if answer == "" {
		return merged, nil
	}
	if contextText == "" {
		return merged, fmt.Errorf("context is required for hallucination detection")
	}
	chunks := hallucinationAnswerChunks(answer)
	searchStart := 0
	for _, chunk := range chunks {
		start := strings.Index(answer[searchStart:], chunk)
		if start < 0 {
			return merged, fmt.Errorf("answer window is not a substring of the original answer")
		}
		start += searchStart
		searchStart = start + 1
		result, err := d.handle.Call(ctx, string(d.spec.Recipe), tasks.GroundedTextRequest{Context: contextText, Question: question, Answer: chunk})
		if err != nil {
			return merged, err
		}
		if result.Summary != nil {
			if merged.Summary == nil {
				copy := *result.Summary
				merged.Summary = &copy
			} else {
				merged.Summary.Value = float64(mergeChunkConfidence(len(merged.Entities) > 0, float32(merged.Summary.Value), len(result.Entities) > 0, float32(result.Summary.Value)))
			}
			merged.SummarySemantics = result.SummarySemantics
		}
		merged.ScoresAvailable = result.ScoresAvailable
		for _, span := range result.Entities {
			span.Start += start
			span.End += start
			duplicate := false
			for i, previous := range merged.Entities {
				if previous.EntityType == span.EntityType && previous.Start == span.Start && previous.End == span.End {
					if span.Confidence > previous.Confidence {
						merged.Entities[i] = span
					}
					duplicate = true
					break
				}
			}
			if !duplicate {
				merged.Entities = append(merged.Entities, span)
			}
		}
	}
	return merged, nil
}

func (d *HallucinationDetector) Detect(ctx context.Context, contextText, question, answer string) (*HallucinationResult, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()
	spans, err := d.detectSpans(ctx, contextText, question, answer)
	if err != nil {
		return nil, err
	}
	result := &HallucinationResult{HallucinationDetected: len(spans.Entities) > 0}
	if spans.Summary != nil {
		result.Confidence = float32(spans.Summary.Value)
		result.ScoreAvailable = true
	}
	if spans.SummarySemantics != nil {
		result.ScoreKind = spans.SummarySemantics.Unit
	}
	for _, span := range spans.Entities {
		if d.acceptSpan(span.Text, span.Confidence, spans.HasScores()) {
			result.UnsupportedSpans = append(result.UnsupportedSpans, span.Text)
		}
	}
	if len(result.UnsupportedSpans) == 0 {
		result.HallucinationDetected = false
	}
	return result, nil
}

func (d *HallucinationDetector) acceptSpan(text string, confidence float32, hasScore bool) bool {
	minimum := d.config.MinSpanLength
	if minimum <= 0 {
		minimum = 1
	}
	if len(strings.Fields(text)) < minimum {
		return false
	}
	return !hasScore || confidence >= d.config.MinSpanConfidence
}

func (d *HallucinationDetector) IsInitialized() bool {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.initialized
}

func (d *HallucinationDetector) Close() error {
	if d == nil {
		return nil
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	d.initialized = false
	d.nliInitialized = false
	var errs []error
	if d.handle != nil {
		errs = append(errs, d.handle.Close())
	}
	if d.nliHandle != nil {
		errs = append(errs, d.nliHandle.Close())
	}
	return errors.Join(errs...)
}
