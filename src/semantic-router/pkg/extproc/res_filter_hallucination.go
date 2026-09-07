package extproc

import (
	"fmt"
	"strings"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

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
	// Only run if conditions are met
	if !r.shouldPerformHallucinationDetection(ctx) {
		return nil
	}
	if assistantContent == "" {
		logging.Debugf("No assistant content to check for hallucination")
		return nil
	}

	// Check if NLI is enabled for this decision
	useNLI := r.isNLIEnabledForDecision(ctx.VSRSelectedDecision)

	logging.Debugf("Hallucination detection: decision=%v, useNLI=%v",
		ctx.VSRSelectedDecision != nil, useNLI)

	start := time.Now()

	if useNLI {
		return r.performHallucinationDetectionWithNLI(ctx, assistantContent)
	}

	// Use basic hallucination detection
	classifier := r.classifierForRequest(ctx)
	result, err := classifier.DetectHallucination(
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

	decisionName := requestDecisionStateKey(ctx)

	if result.HallucinationDetected {
		metrics.RecordPluginExecution("hallucination", decisionName, "detected", latency)
		logging.Warnf("Hallucination detected: confidence=%.3f, unsupported_spans=%d, action=%s",
			result.Confidence, len(result.UnsupportedSpans), r.getHallucinationActionForDecision(ctx.VSRSelectedDecision))
	} else {
		metrics.RecordPluginExecution("hallucination", decisionName, "not_detected", latency)
		logging.Debugf("No hallucination detected: confidence=%.3f", result.Confidence)
	}

	return nil
}

// performHallucinationDetectionWithNLI performs hallucination detection with NLI explanations
func (r *OpenAIRouter) performHallucinationDetectionWithNLI(ctx *RequestContext, assistantContent string) *ext_proc.ProcessingResponse {
	start := time.Now()

	classifier := r.classifierForRequest(ctx)
	result, err := classifier.DetectHallucinationWithNLI(
		ctx.ToolResultsContext,
		ctx.UserContent,
		assistantContent,
	)

	latency := time.Since(start).Seconds()
	metrics.RecordHallucinationDetectionLatency(latency)

	if err != nil {
		logging.Errorf("Hallucination detection with NLI failed: %v", err)
		metrics.RecordPluginError("hallucination", "detection_nli_error")
		return nil // Don't block on error
	}

	if result == nil {
		logging.Debugf("Hallucination detection with NLI returned nil result")
		return nil
	}

	// Record result to context
	ctx.HallucinationDetected = result.HallucinationDetected
	ctx.HallucinationConfidence = result.Confidence

	// Convert enhanced spans to context format
	if len(result.Spans) > 0 {
		ctx.EnhancedHallucinationInfo = &EnhancedHallucinationInfo{
			Confidence: result.Confidence,
			Spans:      make([]EnhancedHallucinationSpan, 0, len(result.Spans)),
		}
		for _, span := range result.Spans {
			ctx.HallucinationSpans = append(ctx.HallucinationSpans, span.Text)
			ctx.EnhancedHallucinationInfo.Spans = append(ctx.EnhancedHallucinationInfo.Spans, EnhancedHallucinationSpan{
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
	}

	decisionName := requestDecisionStateKey(ctx)

	if result.HallucinationDetected {
		metrics.RecordPluginExecution("hallucination", decisionName, "detected_nli", latency)
		logging.Warnf("Hallucination detected (NLI): confidence=%.3f, spans=%d, action=%s",
			result.Confidence, len(result.Spans), r.getHallucinationActionForDecision(ctx.VSRSelectedDecision))
	} else {
		metrics.RecordPluginExecution("hallucination", decisionName, "not_detected", latency)
		logging.Debugf("No hallucination detected (NLI): confidence=%.3f", result.Confidence)
	}

	return nil
}

// isNLIEnabledForDecision checks if NLI is enabled for the given decision's hallucination plugin
func (r *OpenAIRouter) isNLIEnabledForDecision(decision *config.Decision) bool {
	if decision == nil {
		logging.Debugf("isNLIEnabledForDecision: decision is nil")
		return false
	}

	halConfig := decision.GetHallucinationConfig()
	if halConfig == nil {
		logging.Debugf("isNLIEnabledForDecision: halConfig is nil for decision %s", decision.Name)
		return false
	}

	logging.Debugf("isNLIEnabledForDecision: decision=%s, enabled=%v, useNLI=%v",
		decision.Name, halConfig.Enabled, halConfig.UseNLI)
	return halConfig.UseNLI
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

	// Check if we have enhanced NLI information
	if ctx.EnhancedHallucinationInfo != nil && len(ctx.EnhancedHallucinationInfo.Spans) > 0 {
		return r.buildEnhancedHallucinationWarningText(ctx)
	}

	// Basic details without NLI
	warning := fmt.Sprintf("[Hallucination Warning] This response may contain unsupported claims (confidence: %.0f%%).", ctx.HallucinationConfidence*100)

	if len(ctx.HallucinationSpans) > 0 {
		spans := strings.Join(ctx.HallucinationSpans, "\", \"")
		warning += fmt.Sprintf(" Unsupported spans: \"%s\".", spans)
	}

	warning += " Please verify the information independently."
	return warning
}

// buildEnhancedHallucinationWarningText builds warning text with NLI details
func (r *OpenAIRouter) buildEnhancedHallucinationWarningText(ctx *RequestContext) string {
	info := ctx.EnhancedHallucinationInfo

	warning := fmt.Sprintf("[Hallucination Warning] This response may contain unsupported claims (confidence: %.0f%%).", info.Confidence*100)
	warning += " Detailed analysis:"

	for i, span := range info.Spans {
		warning += fmt.Sprintf(" [%d] \"%s\"", i+1, span.Text)
		warning += fmt.Sprintf(" (NLI: %s, confidence: %.0f%%, severity: %s)", span.NLILabel, span.NLIConfidence*100, severityToString(span.Severity))
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
	// Only applies when fact-check is needed but no tools are available
	if !ctx.FactCheckNeeded || ctx.HasToolsForFactCheck {
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
