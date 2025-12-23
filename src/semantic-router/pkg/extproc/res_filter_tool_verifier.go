package extproc

import (
	"encoding/json"
	"fmt"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/openai/openai-go"
	"go.opentelemetry.io/otel/attribute"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
)

// performToolCallVerification runs Stage 2 of the tool verification pipeline
// on the LLM response to verify tool calls are authorized by user intent.
// Returns a blocking response if unauthorized tool calls are detected, nil otherwise.
func (r *OpenAIRouter) performToolCallVerification(ctx *RequestContext, responseBody []byte) *ext_proc.ProcessingResponse {
	// Check if tool verification is enabled
	if r.ToolVerifier == nil || !r.isToolVerificationEnabled(ctx) {
		return nil
	}

	// Check if Stage 1 already ran and has results
	if !ctx.ToolVerificationStage1Ran {
		logging.Debugf("Tool verification Stage 2 skipped: Stage 1 did not run")
		return nil
	}

	// Extract tool calls from response
	toolCalls := extractToolCallsFromResponse(responseBody)
	if len(toolCalls) == 0 {
		logging.Debugf("Tool verification Stage 2 skipped: no tool calls in response")
		ctx.ToolVerificationStage2SkipReason = "no_tool_calls"
		return nil
	}

	// Start tracing span
	spanCtx, span := tracing.StartSpan(ctx.TraceContext, "tool_verifier_stage2")
	defer span.End()

	startTime := time.Now()

	// Run Stage 2 verification
	result, err := r.ToolVerifier.VerifyRequest(ctx.UserContent, toolCalls)
	if err != nil {
		logging.Errorf("Tool verification Stage 2 failed: %v", err)
		tracing.RecordError(span, err)
		metrics.RecordRequestError(ctx.RequestModel, "tool_verifier_stage2_error")
		return nil // Don't block on error
	}

	latencyMs := time.Since(startTime).Milliseconds()

	// Update context with Stage 2 results
	ctx.ToolVerificationStage2Ran = true
	ctx.ToolVerificationStage2LatencyMs = latencyMs
	ctx.ToolVerificationHasUnauthorized = len(result.UnauthorizedToolCalls) > 0

	// Record metrics
	metrics.RecordClassifierLatency("tool_verifier_stage2", float64(latencyMs)/1000.0)

	tracing.SetSpanAttributes(span,
		attribute.Int64("stage2_latency_ms", latencyMs),
		attribute.Bool("has_unauthorized", ctx.ToolVerificationHasUnauthorized),
		attribute.Int("unauthorized_count", len(result.UnauthorizedToolCalls)),
	)

	// Check if should take action based on enforcement policy
	if result.Stage2Blocked || ctx.ToolVerificationHasUnauthorized {
		action := r.Config.ToolVerifier.Enforcement.GetUnauthorizedToolCallAction()

		logging.Warnf("Tool verification Stage 2: unauthorized=%v, action=%s, reason=%s",
			ctx.ToolVerificationHasUnauthorized, action, result.BlockReason)

		tracing.SetSpanAttributes(span,
			attribute.Bool("unauthorized_detected", true),
			attribute.String("action", action),
			attribute.String("block_reason", result.BlockReason),
		)

		// Log security event
		logging.LogEvent("security_alert", map[string]interface{}{
			"reason_code":        "unauthorized_tool_call",
			"action":             action,
			"block_reason":       result.BlockReason,
			"request_id":         ctx.RequestID,
			"tool_calls":         len(toolCalls),
			"unauthorized_count": len(result.UnauthorizedToolCalls),
		})

		// Store for later use
		ctx.ToolVerificationBlockReason = result.BlockReason
		ctx.ToolVerificationUnauthorizedCalls = result.UnauthorizedToolCalls

		switch action {
		case "block":
			if result.Stage2Blocked {
				metrics.RecordRequestError(ctx.RequestModel, "tool_verifier_stage2_blocked")
				ctx.TraceContext = spanCtx
				return createToolVerificationBlockResponse(result, ctx.ExpectStreamingResponse)
			}
		case "header":
			// Allow through, headers will be added below
			ctx.ToolVerificationAddWarningHeaders = true
		case "body":
			// Allow through, warning will be prepended
			ctx.ToolVerificationAddWarningBody = true
		case "none":
			logging.Infof("Unauthorized tool call detected but action is 'none', allowing through")
		}
	}

	logging.Infof("Tool verification Stage 2 passed: %d tool calls verified", len(toolCalls))
	ctx.TraceContext = spanCtx
	return nil
}

// isToolVerificationEnabled checks if tool verification should run for this request
func (r *OpenAIRouter) isToolVerificationEnabled(ctx *RequestContext) bool {
	if r.Config == nil || !r.Config.ToolVerifier.Enabled {
		return false
	}

	// Check if enabled for the matched decision
	if ctx.VSRSelectedDecision != nil {
		// TODO: Add per-decision tool verification config
		// For now, use global setting
	}

	return r.Config.ToolVerifier.Stage2.Enabled
}

// extractToolCallsFromResponse extracts tool calls from OpenAI response JSON
func extractToolCallsFromResponse(responseBody []byte) []classification.ToolCallInfo {
	var response openai.ChatCompletion
	if err := json.Unmarshal(responseBody, &response); err != nil {
		logging.Debugf("Failed to parse response for tool calls: %v", err)
		return nil
	}

	var toolCalls []classification.ToolCallInfo
	for _, choice := range response.Choices {
		if choice.Message.ToolCalls == nil {
			continue
		}
		for _, tc := range choice.Message.ToolCalls {
			// Parse arguments
			var args map[string]interface{}
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
				logging.Debugf("Failed to parse tool call arguments: %v", err)
				args = map[string]interface{}{"raw": tc.Function.Arguments}
			}

			toolCalls = append(toolCalls, classification.ToolCallInfo{
				Name:      tc.Function.Name,
				Arguments: args,
			})
		}
	}

	return toolCalls
}

// createToolVerificationBlockResponse creates a blocking response for unauthorized tool calls
func createToolVerificationBlockResponse(result *classification.ToolVerificationResult, isStreaming bool) *ext_proc.ProcessingResponse {
	errorMessage := fmt.Sprintf("Tool call blocked: %s", result.BlockReason)

	// Create error response body
	errorBody := map[string]interface{}{
		"error": map[string]interface{}{
			"message": errorMessage,
			"type":    "security_violation",
			"code":    "unauthorized_tool_call",
			"details": map[string]interface{}{
				"stage":              "stage2",
				"unauthorized_calls": result.UnauthorizedToolCalls,
			},
		},
	}

	bodyBytes, _ := json.Marshal(errorBody)

	if isStreaming {
		// For streaming, use SSE format
		sseBody := fmt.Sprintf("data: %s\n\ndata: [DONE]\n\n", string(bodyBytes))
		bodyBytes = []byte(sseBody)
	}

	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: &ext_proc.ImmediateResponse{
				Status: &typev3.HttpStatus{
					Code: typev3.StatusCode_Forbidden,
				},
				Headers: &ext_proc.HeaderMutation{
					SetHeaders: []*core.HeaderValueOption{
						{
							Header: &core.HeaderValue{
								Key:   "Content-Type",
								Value: "application/json",
							},
						},
						{
							Header: &core.HeaderValue{
								Key:   headers.VSRToolVerificationBlocked,
								Value: "true",
							},
						},
						{
							Header: &core.HeaderValue{
								Key:   headers.VSRToolVerificationReason,
								Value: result.BlockReason,
							},
						},
					},
				},
				Body: bodyBytes,
			},
		},
	}
}

// performStage1ToolVerification runs Stage 1 (prompt classification) during request processing
// Called from handleRequestBody to classify the prompt before LLM inference
func (r *OpenAIRouter) performStage1ToolVerification(ctx *RequestContext, userContent string) *ext_proc.ProcessingResponse {
	// Check if tool verification is enabled
	if r.ToolVerifier == nil || !r.Config.ToolVerifier.Enabled || !r.Config.ToolVerifier.Stage1.Enabled {
		return nil
	}

	// Start tracing span
	spanCtx, span := tracing.StartSpan(ctx.TraceContext, "tool_verifier_stage1")
	defer span.End()

	startTime := time.Now()

	// Run Stage 1 classification
	stage1Result, err := r.ToolVerifier.VerifyRequest(userContent, nil) // No tool calls yet
	if err != nil {
		logging.Errorf("Tool verification Stage 1 failed: %v", err)
		tracing.RecordError(span, err)
		metrics.RecordRequestError(ctx.RequestModel, "tool_verifier_stage1_error")
		return nil // Don't block on error
	}

	latencyMs := time.Since(startTime).Milliseconds()

	// Update context with Stage 1 results
	ctx.ToolVerificationStage1Ran = true
	ctx.ToolVerificationInjectionRisk = stage1Result.InjectionRisk
	ctx.ToolVerificationInjectionConfidence = stage1Result.InjectionConfidence
	ctx.ToolVerificationStage1LatencyMs = latencyMs

	// Record metrics
	metrics.RecordClassifierLatency("tool_verifier_stage1", float64(latencyMs)/1000.0)

	tracing.SetSpanAttributes(span,
		attribute.Int64("stage1_latency_ms", latencyMs),
		attribute.Bool("injection_risk", stage1Result.InjectionRisk),
		attribute.Float64("injection_confidence", float64(stage1Result.InjectionConfidence)),
	)

	// Check if injection detected - apply enforcement policy
	if stage1Result.Stage1Blocked || stage1Result.InjectionRisk {
		action := r.Config.ToolVerifier.Enforcement.GetInjectionAction()

		logging.Warnf("Tool verification Stage 1: injection_risk=%v, confidence=%.3f, action=%s",
			stage1Result.InjectionRisk, stage1Result.InjectionConfidence, action)

		tracing.SetSpanAttributes(span,
			attribute.Bool("injection_detected", true),
			attribute.String("action", action),
			attribute.String("block_reason", stage1Result.BlockReason),
		)

		// Log security event
		logging.LogEvent("security_alert", map[string]interface{}{
			"reason_code":          "injection_detected",
			"action":               action,
			"block_reason":         stage1Result.BlockReason,
			"request_id":           ctx.RequestID,
			"injection_confidence": stage1Result.InjectionConfidence,
		})

		// Store for later header/body warning
		ctx.ToolVerificationBlockReason = stage1Result.BlockReason

		// Apply enforcement policy - this OVERRIDES the blocking decision from ToolVerifier
		switch action {
		case "block":
			// Block the request
			if stage1Result.InjectionRisk {
				metrics.RecordRequestError(ctx.RequestModel, "tool_verifier_stage1_blocked")
				ctx.TraceContext = spanCtx
				return createStage1BlockResponse(stage1Result, ctx.ExpectStreamingResponse)
			}
		case "header":
			// Allow through, headers will be added in response processing
			ctx.ToolVerificationAddWarningHeaders = true
			logging.Infof("Injection detected (confidence=%.2f%%), action='header' - adding warning headers",
				stage1Result.InjectionConfidence*100)
		case "body":
			// Allow through, warning will be prepended in response processing
			ctx.ToolVerificationAddWarningBody = true
			logging.Infof("Injection detected (confidence=%.2f%%), action='body' - adding warning to body",
				stage1Result.InjectionConfidence*100)
		case "none":
			// Just log, no action
			logging.Infof("Injection detected (confidence=%.2f%%) but action is 'none', allowing through",
				stage1Result.InjectionConfidence*100)
		}
	}

	logging.Infof("Tool verification Stage 1 passed: injection_risk=%v, confidence=%.2f",
		stage1Result.InjectionRisk, stage1Result.InjectionConfidence)

	ctx.TraceContext = spanCtx
	return nil
}

// applyToolVerificationWarning adds warning headers or body based on enforcement policy
// Returns modified response body (for body action) and response with headers (for header action)
func (r *OpenAIRouter) applyToolVerificationWarning(response *ext_proc.ProcessingResponse, ctx *RequestContext, responseBody []byte) ([]byte, *ext_proc.ProcessingResponse) {
	if !ctx.ToolVerificationAddWarningHeaders && !ctx.ToolVerificationAddWarningBody {
		return responseBody, response
	}

	includeDetails := r.Config.ToolVerifier.Enforcement.IncludeDetails

	if ctx.ToolVerificationAddWarningHeaders {
		return responseBody, r.addToolVerificationWarningHeaders(response, ctx)
	}

	if ctx.ToolVerificationAddWarningBody {
		return r.prependToolVerificationWarningToBody(responseBody, ctx, includeDetails), response
	}

	return responseBody, response
}

// addToolVerificationWarningHeaders adds warning headers to the response
func (r *OpenAIRouter) addToolVerificationWarningHeaders(response *ext_proc.ProcessingResponse, ctx *RequestContext) *ext_proc.ProcessingResponse {
	// Get the body response from the response
	bodyResponse, ok := response.Response.(*ext_proc.ProcessingResponse_ResponseBody)
	if !ok {
		return response
	}

	// Create header mutation with tool verification warning
	headerMutation := &ext_proc.HeaderMutation{
		SetHeaders: []*core.HeaderValueOption{
			{
				Header: &core.HeaderValue{
					Key:      headers.VSRToolVerificationInjectionRisk,
					RawValue: []byte(fmt.Sprintf("%v", ctx.ToolVerificationInjectionRisk)),
				},
			},
			{
				Header: &core.HeaderValue{
					Key:      headers.VSRToolVerificationConfidence,
					RawValue: []byte(fmt.Sprintf("%.3f", ctx.ToolVerificationInjectionConfidence)),
				},
			},
		},
	}

	// Add reason if available
	if ctx.ToolVerificationBlockReason != "" {
		headerMutation.SetHeaders = append(headerMutation.SetHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      headers.VSRToolVerificationReason,
				RawValue: []byte(ctx.ToolVerificationBlockReason),
			},
		})
	}

	// Add latency
	totalLatency := ctx.ToolVerificationStage1LatencyMs + ctx.ToolVerificationStage2LatencyMs
	headerMutation.SetHeaders = append(headerMutation.SetHeaders, &core.HeaderValueOption{
		Header: &core.HeaderValue{
			Key:      headers.VSRToolVerificationLatency,
			RawValue: []byte(fmt.Sprintf("%d", totalLatency)),
		},
	})

	// Update the response with the header mutation
	if bodyResponse.ResponseBody.Response == nil {
		bodyResponse.ResponseBody.Response = &ext_proc.CommonResponse{}
	}
	bodyResponse.ResponseBody.Response.HeaderMutation = headerMutation

	return response
}

// prependToolVerificationWarningToBody prepends a warning to the response body
func (r *OpenAIRouter) prependToolVerificationWarningToBody(responseBody []byte, ctx *RequestContext, includeDetails bool) []byte {
	var warning string

	if ctx.ToolVerificationInjectionRisk {
		warning = "[SECURITY WARNING: Potential injection detected in prompt. "
		if includeDetails {
			warning += fmt.Sprintf("Confidence: %.1f%%. ", ctx.ToolVerificationInjectionConfidence*100)
		}
		warning += "Response may be compromised.]\n\n"
	}

	if ctx.ToolVerificationHasUnauthorized && len(ctx.ToolVerificationUnauthorizedCalls) > 0 {
		warning += "[SECURITY WARNING: Unauthorized tool calls detected. "
		if includeDetails {
			warning += fmt.Sprintf("%d unauthorized calls. ", len(ctx.ToolVerificationUnauthorizedCalls))
		}
		warning += "Some operations may not have been authorized by your request.]\n\n"
	}

	if warning == "" {
		return responseBody
	}

	// Try to modify the content field in the response
	var response map[string]interface{}
	if err := json.Unmarshal(responseBody, &response); err != nil {
		logging.Debugf("Failed to parse response for warning injection: %v", err)
		return responseBody
	}

	// Find and modify the content in choices
	if choices, ok := response["choices"].([]interface{}); ok && len(choices) > 0 {
		if choice, ok := choices[0].(map[string]interface{}); ok {
			if message, ok := choice["message"].(map[string]interface{}); ok {
				if content, ok := message["content"].(string); ok {
					message["content"] = warning + content
				}
			}
		}
	}

	modifiedBody, err := json.Marshal(response)
	if err != nil {
		logging.Debugf("Failed to marshal modified response: %v", err)
		return responseBody
	}

	return modifiedBody
}

// createStage1BlockResponse creates a blocking response for Stage 1 injection detection
func createStage1BlockResponse(result *classification.ToolVerificationResult, isStreaming bool) *ext_proc.ProcessingResponse {
	errorMessage := fmt.Sprintf("Request blocked: %s", result.BlockReason)

	errorBody := map[string]interface{}{
		"error": map[string]interface{}{
			"message": errorMessage,
			"type":    "security_violation",
			"code":    "injection_detected",
			"details": map[string]interface{}{
				"stage":      "stage1",
				"confidence": result.InjectionConfidence,
			},
		},
	}

	bodyBytes, _ := json.Marshal(errorBody)

	if isStreaming {
		sseBody := fmt.Sprintf("data: %s\n\ndata: [DONE]\n\n", string(bodyBytes))
		bodyBytes = []byte(sseBody)
	}

	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: &ext_proc.ImmediateResponse{
				Status: &typev3.HttpStatus{
					Code: typev3.StatusCode_Forbidden,
				},
				Headers: &ext_proc.HeaderMutation{
					SetHeaders: []*core.HeaderValueOption{
						{
							Header: &core.HeaderValue{
								Key:   "Content-Type",
								Value: "application/json",
							},
						},
						{
							Header: &core.HeaderValue{
								Key:   headers.VSRToolVerificationBlocked,
								Value: "true",
							},
						},
						{
							Header: &core.HeaderValue{
								Key:   headers.VSRToolVerificationStage,
								Value: "stage1",
							},
						},
						{
							Header: &core.HeaderValue{
								Key:   headers.VSRToolVerificationReason,
								Value: result.BlockReason,
							},
						},
					},
				},
				Body: bodyBytes,
			},
		},
	}
}
