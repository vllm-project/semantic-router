package extproc

import (
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// hallucinationRules returns the hallucination rules of the recipe selected
// for this request, read through its classifier like every other rule: a rule
// declared on one entrypoint's recipe must not check, or fail to check,
// another entrypoint's answers.
func (r *OpenAIRouter) hallucinationRules(ctx *RequestContext) []config.HallucinationRule {
	cfg := classifierConfig(r.classifierForRequest(ctx))
	if cfg == nil {
		return nil
	}
	return cfg.HallucinationRules
}

// hallucinationSignalDeclared reports whether the selected recipe declares a
// hallucination rule, and therefore whether the plugin should read the signal
// instead of classifying the answer itself.
func (r *OpenAIRouter) hallucinationSignalDeclared(ctx *RequestContext) bool {
	return len(r.hallucinationRules(ctx)) > 0
}

// evaluateHallucinationSignal checks the model's answer against the grounding
// context the request carried, for every hallucination rule the selected
// recipe declares.
//
// It is driven by the rules, not by the plugin: the observation is published
// whether or not the selected decision carries a plugin that acts on it. The
// detector only has something to check when the request-stage fact-check
// signal said the prompt makes claims worth grounding and the request carried
// context to ground them against; without the former the rule is not
// applicable and stays unpublished, without the latter it is unavailable.
func (r *OpenAIRouter) evaluateHallucinationSignal(ctx *RequestContext, assistantContent string) {
	if ctx == nil || r == nil {
		return
	}
	rules := r.hallucinationRules(ctx)
	if len(rules) == 0 || !ctx.FactCheckNeeded {
		return
	}

	classifier := r.classifierForRequest(ctx)
	if classifier == nil || !classifier.IsHallucinationDetectionEnabled() {
		// Declared but unbacked. Unresolved rather than clean.
		r.publishHallucinationSignal(ctx, rules, nil, classification.HallucinationSignalFailedCode)
		return
	}
	if !ctx.HasToolsForFactCheck || ctx.ToolResultsContext == "" {
		// Nothing to ground the answer against: "could not look" is not
		// "looked and found nothing". This is the unverified-factual case the
		// plugin has its own action for.
		r.publishHallucinationSignal(ctx, rules, nil, classification.HallucinationSignalContextUnavailable)
		return
	}
	if assistantContent == "" {
		// A response made entirely of tool calls or media has no answer to check.
		r.publishHallucinationSignal(ctx, rules, nil, classification.HallucinationSignalFailedCode)
		return
	}

	start := time.Now()
	evidence, err := r.detectHallucinationEvidence(classifier, ctx, assistantContent, hallucinationRulesUseNLI(rules))
	latency := time.Since(start).Seconds()
	metrics.RecordHallucinationDetectionLatency(latency)
	if err != nil {
		logging.Errorf("Hallucination signal evaluation failed: %v", err)
		metrics.RecordPluginError("hallucination", "detection_error")
		r.publishHallucinationSignal(ctx, rules, nil, classification.HallucinationSignalFailedCode)
		return
	}
	for _, rule := range rules {
		classifier.RecordSignalExtraction(config.SignalTypeHallucination, rule.Name, latency)
	}
	r.publishHallucinationSignal(ctx, rules, evidence, "")
}

// hallucinationRulesUseNLI reports whether any declared rule asks for NLI
// explanations. One rule asking is enough: the detector runs once and every
// rule reads the same evidence.
func hallucinationRulesUseNLI(rules []config.HallucinationRule) bool {
	for _, rule := range rules {
		if rule.UseNLI {
			return true
		}
	}
	return false
}

// detectHallucinationEvidence runs the detector once for the response and
// shapes its output the way the plugin's actions and Router Replay read it.
func (r *OpenAIRouter) detectHallucinationEvidence(
	classifier *classification.Classifier,
	ctx *RequestContext,
	answer string,
	useNLI bool,
) (*ResponseHallucinationEvidence, error) {
	if !useNLI {
		result, err := classifier.DetectHallucination(ctx.embeddingContext(), ctx.ToolResultsContext, ctx.UserContent, answer)
		if err != nil {
			return nil, err
		}
		if result == nil {
			return nil, fmt.Errorf("hallucination detector returned no result")
		}
		return &ResponseHallucinationEvidence{
			Detected:   result.HallucinationDetected,
			Confidence: result.Confidence,
			Spans:      result.UnsupportedSpans,
		}, nil
	}

	result, err := classifier.DetectHallucinationWithNLI(ctx.embeddingContext(), ctx.ToolResultsContext, ctx.UserContent, answer)
	if err != nil {
		return nil, err
	}
	if result == nil {
		return nil, fmt.Errorf("hallucination detector returned no result")
	}
	return hallucinationEvidenceFromNLI(result), nil
}

// hallucinationEvidenceFromNLI shapes one NLI detection the way both response
// paths record it, so the signal path and the plugin-owned path cannot report
// the same detection differently.
func hallucinationEvidenceFromNLI(result *classification.EnhancedHallucinationResult) *ResponseHallucinationEvidence {
	evidence := &ResponseHallucinationEvidence{
		Detected:   result.HallucinationDetected,
		Confidence: result.Confidence,
	}
	if len(result.Spans) == 0 {
		return evidence
	}
	evidence.Enhanced = &EnhancedHallucinationInfo{
		Confidence: result.Confidence,
		Spans:      make([]EnhancedHallucinationSpan, 0, len(result.Spans)),
	}
	for _, span := range result.Spans {
		evidence.Spans = append(evidence.Spans, span.Text)
		evidence.Enhanced.Spans = append(evidence.Enhanced.Spans, EnhancedHallucinationSpan{
			Text:                    span.Text,
			Start:                   span.Start,
			End:                     span.End,
			HallucinationConfidence: span.HallucinationConfidence,
			NLILabel:                span.NLILabelStr,
			NLIConfidence:           span.NLIConfidence,
			Severity:                span.Severity,
			Explanation:             span.Explanation,
		})
	}
	return evidence
}

// publishHallucinationSignal records the response-stage observation where
// every other signal is recorded, under the "hallucination:<rule>" key, so
// Router Replay and the selected decision's plugin read one shape. The plugin
// still enforces; this only publishes the evidence it acts on.
func (r *OpenAIRouter) publishHallucinationSignal(ctx *RequestContext, rules []config.HallucinationRule, evidence *ResponseHallucinationEvidence, failureCode string) {
	detected, confidence := false, float32(0)
	if evidence != nil {
		detected, confidence = evidence.Detected, evidence.Confidence
	}
	signal := classification.EvaluateResponseHallucinationSignal(rules, detected, confidence, failureCode)
	if signal == nil {
		return
	}
	ctx.VSRHallucinationEvidence = evidence
	ctx.VSRMatchedHallucination = append(ctx.VSRMatchedHallucination, signal.MatchedRules...)
	recordResponseSignal(ctx, signal.Confidences, signal.Errors)
}

// hallucinationSignalOutcome reads the published signal back for the plugin:
// whether a declared rule matched, the failure code when the answer could not
// be checked, and whether the rule was evaluated at all for this request.
func (r *OpenAIRouter) hallucinationSignalOutcome(ctx *RequestContext) (matched bool, failureCode string, observed bool) {
	if ctx == nil {
		return false, "", false
	}
	for _, rule := range r.hallucinationRules(ctx) {
		key := signalKey(config.SignalTypeHallucination, rule.Name)
		if code, failed := ctx.VSRSignalErrors[key]; failed {
			return false, code, true
		}
		if _, ok := ctx.VSRSignalConfidences[key]; ok {
			observed = true
		}
	}
	return len(ctx.VSRMatchedHallucination) > 0, "", observed
}
