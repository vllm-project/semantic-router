package extproc

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// performHallucinationDetection checks the response for hallucinations
// Returns nil to allow the response to continue (warning handled in processor_res_body.go)
func (r *OpenAIRouter) performHallucinationDetection(ctx *RequestContext, responseBody []byte) *ext_proc.ProcessingResponse {
	// Only run if conditions are met
	if !r.shouldPerformHallucinationDetection(ctx) {
		return nil
	}

	// Extract assistant content from response
	assistantContent := extractAssistantContentFromResponse(responseBody)
	if assistantContent == "" {
		logging.Debugf("No assistant content to check for hallucination")
		return nil
	}

	// Check if NLI is enabled for this decision
	useNLI := r.isNLIEnabledForDecision(ctx.VSRSelectedDecision)

	logging.Infof("Hallucination detection: decision=%v, useNLI=%v",
		ctx.VSRSelectedDecision != nil, useNLI)

	start := time.Now()

	if useNLI {
		// Use enhanced detection with NLI explanations
		logging.Infof("Using NLI-enhanced hallucination detection")
		return r.performHallucinationDetectionWithNLI(ctx, assistantContent)
	}

	// Use basic hallucination detection
	result, err := r.Classifier.DetectHallucination(
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

	decisionName := ""
	if ctx.VSRSelectedDecision != nil {
		decisionName = ctx.VSRSelectedDecision.Name
	}

	if result.HallucinationDetected {
		metrics.RecordPluginExecution("hallucination", decisionName, "detected", latency)
		logging.Warnf("Hallucination detected: confidence=%.3f, unsupported_spans=%d",
			result.Confidence, len(result.UnsupportedSpans))

		// Check action from decision plugin config
		action := r.getHallucinationActionForDecision(ctx.VSRSelectedDecision)

		// For header/body/none actions, allow response through
		// Warning will be handled in processor_res_body.go
		logging.Infof("Hallucination detected, action is '%s'", action)
	} else {
		metrics.RecordPluginExecution("hallucination", decisionName, "not_detected", latency)
		logging.Debugf("No hallucination detected: confidence=%.3f", result.Confidence)
	}

	return nil
}

// performHallucinationDetectionWithNLI performs hallucination detection with NLI explanations
func (r *OpenAIRouter) performHallucinationDetectionWithNLI(ctx *RequestContext, assistantContent string) *ext_proc.ProcessingResponse {
	start := time.Now()

	result, err := r.Classifier.DetectHallucinationWithNLI(
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

	decisionName := ""
	if ctx.VSRSelectedDecision != nil {
		decisionName = ctx.VSRSelectedDecision.Name
	}

	if result.HallucinationDetected {
		metrics.RecordPluginExecution("hallucination", decisionName, "detected_nli", latency)
		logging.Warnf("Hallucination detected (NLI): confidence=%.3f, spans=%d",
			result.Confidence, len(result.Spans))

		// Check action from decision plugin config
		action := r.getHallucinationActionForDecision(ctx.VSRSelectedDecision)

		// For header/body/none actions, allow response through
		// Warning will be handled in processor_res_body.go
		logging.Infof("Hallucination detected (NLI), action is '%s'", action)
	} else {
		metrics.RecordPluginExecution("hallucination", decisionName, "not_detected", latency)
		logging.Debugf("No hallucination detected (NLI): confidence=%.3f", result.Confidence)
	}

	return nil
}

// extractAssistantContentFromResponse extracts the assistant's content from the response
func extractAssistantContentFromResponse(responseBody []byte) string {
	// Parse response using OpenAI SDK types
	var completion openai.ChatCompletion
	if err := json.Unmarshal(responseBody, &completion); err != nil {
		logging.Debugf("Failed to parse response for hallucination detection: %v", err)
		return ""
	}

	// Extract content from the first choice (most common case)
	if len(completion.Choices) == 0 {
		return ""
	}

	message := completion.Choices[0].Message
	if message.Content != "" {
		return message.Content
	}

	return ""
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

// applyHallucinationWarning applies hallucination warning based on the configured action
// Returns modified response body (for body action) and response with headers (for header action)
func (r *OpenAIRouter) applyHallucinationWarning(response *ext_proc.ProcessingResponse, ctx *RequestContext, responseBody []byte) ([]byte, *ext_proc.ProcessingResponse) {
	if !ctx.HallucinationDetected {
		return responseBody, response
	}

	action := r.getHallucinationActionForDecision(ctx.VSRSelectedDecision)
	includeDetails := r.shouldIncludeHallucinationDetails(ctx.VSRSelectedDecision)

	switch action {
	case "header":
		return responseBody, r.addHallucinationWarningHeaders(response, ctx)
	case "body":
		return r.prependHallucinationWarningToBody(responseBody, ctx, includeDetails), response
	case "none":
		logging.Infof("Hallucination detected but action is 'none', skipping warning")
		return responseBody, response
	default:
		// Default to header
		return responseBody, r.addHallucinationWarningHeaders(response, ctx)
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

// addHallucinationWarningHeaders adds warning headers to the response when hallucination is detected
func (r *OpenAIRouter) addHallucinationWarningHeaders(response *ext_proc.ProcessingResponse, ctx *RequestContext) *ext_proc.ProcessingResponse {
	if !ctx.HallucinationDetected {
		return response
	}

	// Get the body response from the response
	bodyResponse, ok := response.Response.(*ext_proc.ProcessingResponse_ResponseBody)
	if !ok {
		return response
	}

	// Create header mutation with hallucination warning
	headerMutation := &ext_proc.HeaderMutation{
		SetHeaders: []*core.HeaderValueOption{
			{
				Header: &core.HeaderValue{
					Key:      headers.HallucinationDetected,
					RawValue: []byte("true"),
				},
			},
		},
	}

	// Add fact-check-needed header if fact check was triggered
	if ctx.FactCheckNeeded {
		headerMutation.SetHeaders = append(headerMutation.SetHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      headers.FactCheckNeeded,
				RawValue: []byte("true"),
			},
		})
	}

	if len(ctx.HallucinationSpans) > 0 {
		spansSummary := strings.Join(ctx.HallucinationSpans, "; ")
		if len(spansSummary) > 500 {
			spansSummary = spansSummary[:500] + "..." // Truncate long spans
		}
		headerMutation.SetHeaders = append(headerMutation.SetHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      headers.HallucinationSpans,
				RawValue: []byte(spansSummary),
			},
		})
	}

	// Update the response with the header mutation
	if bodyResponse.ResponseBody.Response == nil {
		bodyResponse.ResponseBody.Response = &ext_proc.CommonResponse{}
	}
	bodyResponse.ResponseBody.Response.HeaderMutation = headerMutation

	return response
}

// prependHallucinationWarningToBody prepends warning text to the response content
func (r *OpenAIRouter) prependHallucinationWarningToBody(responseBody []byte, ctx *RequestContext, includeDetails bool) []byte {
	// Build warning text
	warningText := r.buildHallucinationWarningText(ctx, includeDetails)

	// Parse response
	var completion map[string]interface{}
	if err := json.Unmarshal(responseBody, &completion); err != nil {
		logging.Errorf("Failed to parse response for hallucination body warning: %v", err)
		return responseBody
	}

	// Modify content in choices
	choices, ok := completion["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return responseBody
	}

	for _, choice := range choices {
		choiceMap, ok := choice.(map[string]interface{})
		if !ok {
			continue
		}
		message, ok := choiceMap["message"].(map[string]interface{})
		if !ok {
			continue
		}
		content, ok := message["content"].(string)
		if !ok {
			continue
		}
		// Prepend warning to content
		message["content"] = warningText + "\n\n" + content
	}

	// Marshal back
	modifiedBody, err := json.Marshal(completion)
	if err != nil {
		logging.Errorf("Failed to marshal response with hallucination body warning: %v", err)
		return responseBody
	}

	return modifiedBody
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

// applyUnverifiedFactualWarning applies unverified factual warning based on the configured action
// Returns modified response body (for body action) and response with headers (for header action)
func (r *OpenAIRouter) applyUnverifiedFactualWarning(response *ext_proc.ProcessingResponse, ctx *RequestContext, responseBody []byte) ([]byte, *ext_proc.ProcessingResponse) {
	if !ctx.UnverifiedFactualResponse {
		return responseBody, response
	}

	action := r.getUnverifiedFactualActionForDecision(ctx.VSRSelectedDecision)

	switch action {
	case "header":
		return responseBody, r.addUnverifiedFactualWarningHeaders(response, ctx)
	case "body":
		return r.prependUnverifiedFactualWarningToBody(responseBody), response
	case "none":
		logging.Infof("Unverified factual response but action is 'none', skipping warning")
		return responseBody, response
	default:
		// Default to header
		return responseBody, r.addUnverifiedFactualWarningHeaders(response, ctx)
	}
}

// addUnverifiedFactualWarningHeaders adds warning headers when a factual response
// could not be verified due to lack of tool context
func (r *OpenAIRouter) addUnverifiedFactualWarningHeaders(response *ext_proc.ProcessingResponse, ctx *RequestContext) *ext_proc.ProcessingResponse {
	if !ctx.UnverifiedFactualResponse {
		return response
	}

	// Get the body response from the response
	bodyResponse, ok := response.Response.(*ext_proc.ProcessingResponse_ResponseBody)
	if !ok {
		return response
	}

	// Create header mutation with unverified warning
	headerMutation := &ext_proc.HeaderMutation{
		SetHeaders: []*core.HeaderValueOption{
			{
				Header: &core.HeaderValue{
					Key:      headers.UnverifiedFactualResponse,
					RawValue: []byte("true"),
				},
			},
			{
				Header: &core.HeaderValue{
					Key:      headers.FactCheckNeeded,
					RawValue: []byte("true"),
				},
			},
			{
				Header: &core.HeaderValue{
					Key:      headers.VerificationContextMissing,
					RawValue: []byte("true"),
				},
			},
		},
	}

	// Update the response with the header mutation
	if bodyResponse.ResponseBody.Response == nil {
		bodyResponse.ResponseBody.Response = &ext_proc.CommonResponse{}
	}

	// Merge with existing headers if any
	if bodyResponse.ResponseBody.Response.HeaderMutation != nil {
		bodyResponse.ResponseBody.Response.HeaderMutation.SetHeaders = append(
			bodyResponse.ResponseBody.Response.HeaderMutation.SetHeaders,
			headerMutation.SetHeaders...,
		)
	} else {
		bodyResponse.ResponseBody.Response.HeaderMutation = headerMutation
	}

	return response
}

// prependUnverifiedFactualWarningToBody prepends unverified factual warning text to the response content
func (r *OpenAIRouter) prependUnverifiedFactualWarningToBody(responseBody []byte) []byte {
	warningText := "[Unverified Response] This response contains factual claims that could not be verified due to missing context."

	// Parse response
	var completion map[string]interface{}
	if err := json.Unmarshal(responseBody, &completion); err != nil {
		logging.Errorf("Failed to parse response for unverified factual body warning: %v", err)
		return responseBody
	}

	// Modify content in choices
	choices, ok := completion["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return responseBody
	}

	for _, choice := range choices {
		choiceMap, ok := choice.(map[string]interface{})
		if !ok {
			continue
		}
		message, ok := choiceMap["message"].(map[string]interface{})
		if !ok {
			continue
		}
		content, ok := message["content"].(string)
		if !ok {
			continue
		}
		// Prepend warning to content
		message["content"] = warningText + "\n\n" + content
	}

	// Marshal back
	modifiedBody, err := json.Marshal(completion)
	if err != nil {
		logging.Errorf("Failed to marshal response with unverified factual body warning: %v", err)
		return responseBody
	}

	return modifiedBody
}
