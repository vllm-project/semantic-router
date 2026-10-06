package extproc

import (
	"fmt"
	"strings"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func (r *OpenAIRouter) performSemanticHallucinationDetection(
	ctx *RequestContext,
	response *llmprotocol.Response,
) *ext_proc.ProcessingResponse {
	return r.performHallucinationDetectionText(ctx, semanticAssistantContent(response))
}

func (r *OpenAIRouter) performHallucinationDetectionText(
	ctx *RequestContext,
	assistantContent string,
) *ext_proc.ProcessingResponse {
	// With hallucination rules declared, detection already ran as a
	// response-stage signal and this plugin only consumes it. Without them it
	// still owns detection, so an existing configuration keeps working
	// unchanged.
	if r.hallucinationSignalDeclared(ctx) {
		r.consumeHallucinationSignal(ctx)
		return nil
	}

	// Only run if conditions are met
	if !r.shouldPerformHallucinationDetection(ctx) {
		return nil
	}
	if assistantContent == "" {
		logging.Debugf("No assistant content to check for hallucination")
		return nil
	}

	classifier := r.classifierForRequest(ctx)
	if hallucinationSpanDetails(classifier) {
		return r.performHallucinationDetectionWithSpans(ctx, classifier, assistantContent)
	}

	start := time.Now()
	result, err := classifier.DetectHallucination(
		ctx.embeddingContext(),
		ctx.ToolResultsContext,
		ctx.UserContent,
		assistantContent,
	)

	latency := time.Since(start).Seconds()
	metrics.RecordHallucinationDetectionLatency(latency)

	if err != nil {
		logging.Errorf("Hallucination detection failed: %v", err)
		metrics.RecordPluginError("hallucination", "detection_error")
		return nil // Don't block on error
	}

	if result == nil {
		logging.Debugf("Hallucination detection returned nil result")
		return nil
	}

	// Record result to context and metrics
	ctx.HallucinationDetected = result.HallucinationDetected
	ctx.HallucinationSpans = result.UnsupportedSpans
	ctx.HallucinationConfidence = result.Confidence
	ctx.HallucinationScoreAvailable = result.ScoreAvailable
	ctx.HallucinationScoreKind = result.ScoreKind

	decisionName := requestDecisionStateKey(ctx)

	if result.HallucinationDetected {
		metrics.RecordPluginExecution("hallucination", decisionName, "detected", latency)
		logging.Warnf("Hallucination detected: score=%s, unsupported_spans=%d, action=%s",
			hallucinationScoreDescription(result.ScoreAvailable, result.ScoreKind, result.Confidence), len(result.UnsupportedSpans), r.getHallucinationActionForDecision(ctx.VSRSelectedDecision))
	} else {
		metrics.RecordPluginExecution("hallucination", decisionName, "not_detected", latency)
		logging.Debugf("No hallucination detected: score=%s", hallucinationScoreDescription(result.ScoreAvailable, result.ScoreKind, result.Confidence))
	}

	return nil
}

// performHallucinationDetectionWithSpans runs detection with span details.
func (r *OpenAIRouter) performHallucinationDetectionWithSpans(ctx *RequestContext, classifier *classification.Classifier, assistantContent string) *ext_proc.ProcessingResponse {
	start := time.Now()
	result, err := classifier.DetectHallucinationWithExplanations(
		ctx.embeddingContext(),
		ctx.ToolResultsContext,
		ctx.UserContent,
		assistantContent,
	)

	latency := time.Since(start).Seconds()
	metrics.RecordHallucinationDetectionLatency(latency)

	if err != nil {
		logging.Errorf("Hallucination detection failed: %v", err)
		metrics.RecordPluginError("hallucination", "detection_error")
		return nil // Don't block on error
	}

	if result == nil {
		logging.Debugf("Hallucination detection returned nil result")
		return nil
	}

	// Record result to context, shaped exactly as the signal path shapes it.
	evidence := hallucinationEvidenceWithSpans(result)
	ctx.HallucinationDetected = evidence.Detected
	ctx.HallucinationConfidence = evidence.Confidence
	ctx.HallucinationScoreAvailable = evidence.ScoreAvailable
	ctx.HallucinationScoreKind = evidence.ScoreKind
	ctx.HallucinationSpans = append(ctx.HallucinationSpans, evidence.Spans...)
	ctx.EnhancedHallucinationInfo = evidence.Enhanced

	decisionName := requestDecisionStateKey(ctx)

	if result.HallucinationDetected {
		metrics.RecordPluginExecution("hallucination", decisionName, "detected", latency)
		logging.Warnf("Hallucination detected: score=%s, spans=%d, action=%s",
			hallucinationScoreDescription(result.ScoreAvailable, result.ScoreKind, result.Confidence), len(result.Spans), r.getHallucinationActionForDecision(ctx.VSRSelectedDecision))
	} else {
		metrics.RecordPluginExecution("hallucination", decisionName, "not_detected", latency)
		logging.Debugf("No hallucination detected: score=%s", hallucinationScoreDescription(result.ScoreAvailable, result.ScoreKind, result.Confidence))
	}

	return nil
}

// consumeHallucinationSignal is the plugin's path once hallucination rules
// are declared: it reads the published observation and carries it into the
// fields its actions read, so a disabled plugin leaves the evidence on record
// without acting on it. Nothing is classified here.
func (r *OpenAIRouter) consumeHallucinationSignal(ctx *RequestContext) {
	if !r.isHallucinationEnabledForDecision(ctx.VSRSelectedDecision) {
		return
	}
	matched, failureCode, observed := r.hallucinationSignalOutcome(ctx)
	decisionName := requestDecisionStateKey(ctx)
	if failureCode != "" {
		metrics.RecordPluginExecution("hallucination", decisionName, "unresolved", 0)
		logging.Debugf("Hallucination signal unresolved for decision %s: %s", decisionName, failureCode)
		return
	}
	evidence := ctx.VSRHallucinationEvidence
	if !observed || evidence == nil {
		return
	}
	ctx.HallucinationDetected = matched
	ctx.HallucinationSpans = evidence.Spans
	ctx.HallucinationConfidence = evidence.Confidence
	ctx.HallucinationScoreAvailable = evidence.ScoreAvailable
	ctx.HallucinationScoreKind = evidence.ScoreKind
	ctx.EnhancedHallucinationInfo = evidence.Enhanced
	if matched {
		metrics.RecordPluginExecution("hallucination", decisionName, "detected", 0)
		logging.Warnf("Hallucination detected: score=%s, unsupported_spans=%d, action=%s",
			hallucinationScoreDescription(evidence.ScoreAvailable, evidence.ScoreKind, evidence.Confidence), len(evidence.Spans), r.getHallucinationActionForDecision(ctx.VSRSelectedDecision))
		return
	}
	metrics.RecordPluginExecution("hallucination", decisionName, "not_detected", 0)
}

func (r *OpenAIRouter) applySemanticHallucinationWarning(
	ctx *RequestContext,
	response *llmprotocol.Response,
) (bool, string) {
	if !ctx.HallucinationDetected {
		return false, ""
	}
	switch r.getHallucinationActionForDecision(ctx.VSRSelectedDecision) {
	case "body":
		warning := r.buildHallucinationWarningText(
			ctx,
			r.shouldIncludeHallucinationDetails(ctx.VSRSelectedDecision),
		)
		return prependSemanticAssistantText(response, warning), ""
	case "none":
		return false, ""
	default:
		return false, headers.ResponseWarningHallucination
	}
}

// shouldIncludeHallucinationDetails checks if detailed hallucination info should be included in body warning
func (r *OpenAIRouter) shouldIncludeHallucinationDetails(decision *config.Decision) bool {
	if decision == nil {
		return false
	}

	halConfig := decision.GetHallucinationConfig()
	if halConfig == nil {
		return false
	}

	return halConfig.IncludeHallucinationDetails
}

// buildHallucinationWarningText builds the warning text for body prepending
func (r *OpenAIRouter) buildHallucinationWarningText(ctx *RequestContext, includeDetails bool) string {
	if !includeDetails {
		return "[Hallucination Warning] This response may contain unsupported claims. Please verify the information independently."
	}

	// Span details when the detector explains its spans
	if ctx.EnhancedHallucinationInfo != nil && len(ctx.EnhancedHallucinationInfo.Spans) > 0 {
		return r.buildEnhancedHallucinationWarningText(ctx)
	}

	// Plain spans
	warning := hallucinationWarningPrefix(ctx.HallucinationScoreAvailable, ctx.HallucinationScoreKind, ctx.HallucinationConfidence)

	if len(ctx.HallucinationSpans) > 0 {
		spans := strings.Join(ctx.HallucinationSpans, "\", \"")
		warning += fmt.Sprintf(" Unsupported spans: \"%s\".", spans)
	}

	warning += " Please verify the information independently."
	return warning
}

// buildEnhancedHallucinationWarningText builds warning text with span details
func (r *OpenAIRouter) buildEnhancedHallucinationWarningText(ctx *RequestContext) string {
	info := ctx.EnhancedHallucinationInfo

	warning := hallucinationWarningPrefix(info.ScoreAvailable, info.ScoreKind, info.Confidence)
	warning += " Detailed analysis:"

	for i, span := range info.Spans {
		warning += fmt.Sprintf(" [%d] \"%s\" (severity: %s)", i+1, span.Text, severityToString(span.Severity))
		if span.Explanation != "" {
			warning += fmt.Sprintf(" - %s", span.Explanation)
		}
	}

	warning += " Please verify the information independently."
	return warning
}

// severityToString converts severity level (0-4) to human-readable string
func severityToString(severity int) string {
	switch severity {
	case 0:
		return "low"
	case 1:
		return "low-medium"
	case 2:
		return "medium"
	case 3:
		return "high"
	case 4:
		return "critical"
	default:
		return "unknown"
	}
}

// checkUnverifiedFactualResponse checks if the response is a fact-check-needed prompt
// without tool context, and marks it as unverified
func (r *OpenAIRouter) checkUnverifiedFactualResponse(ctx *RequestContext) {
	if r.hallucinationSignalDeclared(ctx) {
		// The signal already said whether the answer could be checked: an
		// answer with no grounding context is the unverified-factual case.
		if _, code, _ := r.hallucinationSignalOutcome(ctx); code != classification.HallucinationSignalContextUnavailable {
			return
		}
	} else if !ctx.FactCheckNeeded || ctx.HasToolsForFactCheck {
		// Only applies when fact-check is needed but no tools are available
		return
	}

	// Mark as unverified factual response
	ctx.UnverifiedFactualResponse = true
	metrics.RecordUnverifiedFactualResponse()
	logging.Warnf("Unverified factual response: fact-check needed (confidence=%.3f) but no tool context available",
		ctx.FactCheckConfidence)
}

func (r *OpenAIRouter) applySemanticUnverifiedFactualWarning(
	ctx *RequestContext,
	response *llmprotocol.Response,
) (bool, string) {
	if !ctx.UnverifiedFactualResponse {
		return false, ""
	}
	switch r.getUnverifiedFactualActionForDecision(ctx.VSRSelectedDecision) {
	case "body":
		return prependSemanticAssistantText(
			response,
			"[Unverified Response] This response contains factual claims that could not be verified due to missing context.",
		), ""
	case "none":
		return false, ""
	default:
		return false, headers.ResponseWarningUnverifiedFactual
	}
}

func hallucinationWarningPrefix(available bool, kind string, score float32) string {
	prefix := "[Hallucination Warning] This response may contain unsupported claims"
	if !available {
		return prefix + "."
	}
	if kind == "probability" {
		return fmt.Sprintf("%s (confidence: %.0f%%).", prefix, score*100)
	}
	return fmt.Sprintf("%s (score: %.3f, kind: %s).", prefix, score, kind)
}

func hallucinationScoreDescription(available bool, kind string, score float32) string {
	if !available {
		return "unavailable"
	}
	return fmt.Sprintf("%.3f (%s)", score, kind)
}
