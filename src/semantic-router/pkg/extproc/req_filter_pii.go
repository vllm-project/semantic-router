package extproc

import (
	"strings"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"go.opentelemetry.io/otel/attribute"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/http"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/pii"
)

// performPIIDetection performs PII detection and policy check
// Returns (allowedModel, errorResponse).
// - If errorResponse is not nil, the request should be blocked.
// - If allowedModel is not empty, it's the model that passes PII policy (may be different from selectedModel)
// - isAutoModel indicates whether this is an auto model (true) or a specified model (false)
func (r *OpenAIRouter) performPIIDetection(ctx *RequestContext, userContent string, nonUserMessages []string, categoryName string, selectedModel string, isAutoModel bool) (string, *ext_proc.ProcessingResponse) {
	// Check if PII detection is enabled for this category
	if !r.isPIIDetectionEnabled(categoryName) {
		return selectedModel, nil
	}

	// Detect PII in content
	detectedPII := r.detectPIIWithTracing(ctx, userContent, nonUserMessages, categoryName)
	if len(detectedPII) == 0 {
		return selectedModel, nil
	}

	// Check PII policy and find alternative model if needed
	return r.checkPIIPolicyAndFindAlternative(ctx, selectedModel, detectedPII, categoryName, isAutoModel)
}

// isPIIDetectionEnabled checks if PII detection is enabled for the given category
func (r *OpenAIRouter) isPIIDetectionEnabled(categoryName string) bool {
	piiThreshold := float32(0.0)
	if categoryName != "" && r.Config != nil {
		piiThreshold = r.Config.GetPIIThresholdForCategory(categoryName)
	} else {
		piiThreshold = r.Config.PIIModel.Threshold
	}

	if piiThreshold == 0.0 {
		logging.Infof("PII detection disabled for category: %s", categoryName)
		return false
	}

	logging.Infof("PII detection enabled for category %s (threshold: %.3f)", categoryName, piiThreshold)
	return true
}

// detectPIIWithTracing performs PII detection with tracing and logging
func (r *OpenAIRouter) detectPIIWithTracing(ctx *RequestContext, userContent string, nonUserMessages []string, categoryName string) []string {
	allContent := pii.ExtractAllContent(userContent, nonUserMessages)

	// Start PII detection span
	piiCtx, piiSpan := tracing.StartSpan(ctx.TraceContext, tracing.SpanPIIDetection)
	piiStart := time.Now()

	detectedPII := r.Classifier.DetectPIIInContent(allContent)

	piiTime := time.Since(piiStart).Milliseconds()
	piiDetected := len(detectedPII) > 0

	tracing.SetSpanAttributes(piiSpan,
		attribute.Bool(tracing.AttrPIIDetected, piiDetected),
		attribute.Int64(tracing.AttrPIIDetectionTimeMs, piiTime))

	if piiDetected {
		piiTypesStr := strings.Join(detectedPII, ",")
		tracing.SetSpanAttributes(piiSpan,
			attribute.String(tracing.AttrPIITypes, piiTypesStr))
		logging.Infof("Detected PII types: %s", piiTypesStr)
	}

	piiSpan.End()
	ctx.TraceContext = piiCtx

	return detectedPII
}

// checkPIIPolicyAndFindAlternative checks if the selected model passes PII policy
// and finds an alternative model if needed
func (r *OpenAIRouter) checkPIIPolicyAndFindAlternative(ctx *RequestContext, selectedModel string, detectedPII []string, categoryName string, isAutoModel bool) (string, *ext_proc.ProcessingResponse) {
	// Check if PII policy is enabled for this model
	if selectedModel == "" || !r.PIIChecker.IsPIIEnabled(selectedModel) {
		return selectedModel, nil
	}

	// Check if the selected model passes PII policy
	allowed, deniedPII, err := r.PIIChecker.CheckPolicy(selectedModel, detectedPII)
	if err != nil {
		logging.Errorf("Error checking PII policy for model %s: %v", selectedModel, err)
		return selectedModel, nil
	}

	if allowed {
		return selectedModel, nil
	}

	// Model violates PII policy - find alternative or return error
	logging.Warnf("Model %s violates PII policy, finding alternative", selectedModel)

	if isAutoModel && categoryName != "" {
		// For auto models, try to find an alternative model from the same category
		return r.findAlternativeModelForPII(ctx, selectedModel, detectedPII, categoryName)
	}

	// For non-auto models, return error (no alternative available)
	return r.createPIIViolationResponse(ctx, selectedModel, deniedPII)
}

// findAlternativeModelForPII finds an alternative model that passes PII policy
func (r *OpenAIRouter) findAlternativeModelForPII(ctx *RequestContext, originalModel string, detectedPII []string, categoryName string) (string, *ext_proc.ProcessingResponse) {
	alternativeModels := r.Classifier.GetModelsForCategory(categoryName)
	allowedModels := r.PIIChecker.FilterModelsForPII(alternativeModels, detectedPII)

	if len(allowedModels) > 0 {
		// Select the best allowed model from this category
		allowedModel := r.Classifier.SelectBestModelFromList(allowedModels, categoryName)
		logging.Infof("Selected alternative model %s that passes PII policy", allowedModel)
		metrics.RecordRoutingReasonCode("pii_policy_alternative_selected", allowedModel)
		return allowedModel, nil
	}

	// No alternative models pass PII policy, try default model
	logging.Warnf("No models in category %s pass PII policy, trying default", categoryName)
	return r.tryDefaultModelForPII(ctx, detectedPII)
}

// tryDefaultModelForPII tries to use the default model if it passes PII policy
func (r *OpenAIRouter) tryDefaultModelForPII(ctx *RequestContext, detectedPII []string) (string, *ext_proc.ProcessingResponse) {
	defaultModel := r.Config.DefaultModel

	// Check if default model passes policy
	defaultAllowed, defaultDeniedPII, _ := r.PIIChecker.CheckPolicy(defaultModel, detectedPII)
	if defaultAllowed {
		return defaultModel, nil
	}

	// Default model also violates PII policy
	logging.Errorf("Default model %s also violates PII policy, returning error", defaultModel)
	logging.LogEvent("routing_block", map[string]interface{}{
		"reason_code": "pii_policy_denied_default_model",
		"request_id":  ctx.RequestID,
		"model":       defaultModel,
		"denied_pii":  defaultDeniedPII,
	})
	metrics.RecordRequestError(defaultModel, "pii_policy_denied")

	piiResponse := http.CreatePIIViolationResponse(defaultModel, defaultDeniedPII, ctx.ExpectStreamingResponse)
	return "", piiResponse
}

// createPIIViolationResponse creates an error response for PII policy violation
func (r *OpenAIRouter) createPIIViolationResponse(ctx *RequestContext, model string, deniedPII []string) (string, *ext_proc.ProcessingResponse) {
	logging.Warnf("Model %s violates PII policy, returning error", model)
	logging.LogEvent("routing_block", map[string]interface{}{
		"reason_code": "pii_policy_denied",
		"request_id":  ctx.RequestID,
		"model":       model,
		"denied_pii":  deniedPII,
	})
	metrics.RecordRequestError(model, "pii_policy_denied")

	piiResponse := http.CreatePIIViolationResponse(model, deniedPII, ctx.ExpectStreamingResponse)
	return "", piiResponse
}
