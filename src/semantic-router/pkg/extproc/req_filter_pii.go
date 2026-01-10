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
// Returns errorResponse if the request should be blocked, nil otherwise
func (r *OpenAIRouter) performPIIDetection(ctx *RequestContext, userContent string, nonUserMessages []string, decisionName string) *ext_proc.ProcessingResponse {
	// Check if PII detection is enabled for this decision
	if !r.isPIIDetectionEnabled(decisionName) {
		return nil
	}

	// Detect PII in content
	detectedPII := r.detectPIIWithTracing(ctx, userContent, nonUserMessages, decisionName)
	if len(detectedPII) == 0 {
		return nil
	}

	// Check PII policy
	return r.checkPIIPolicy(ctx, detectedPII, decisionName)
}

// isPIIDetectionEnabled checks if PII detection is enabled for the given decision
func (r *OpenAIRouter) isPIIDetectionEnabled(decisionName string) bool {
	// Use PIIChecker to check if PII detection is enabled for this decision
	// This checks if the decision has a PII plugin with enabled: true
	if !r.PIIChecker.IsPIIEnabled(decisionName) {
		return false
	}

	// Also check if there's a valid threshold configured
	piiThreshold := float32(0.0)
	if decisionName != "" && r.Config != nil {
		piiThreshold = r.Config.GetPIIThresholdForDecision(decisionName)
	} else {
		piiThreshold = r.Config.PIIModel.Threshold
	}

	if piiThreshold == 0.0 {
		logging.Infof("PII detection disabled for decision %s: threshold is 0", decisionName)
		return false
	}

	logging.Infof("PII detection enabled for decision %s (threshold: %.3f)", decisionName, piiThreshold)
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

// checkPIIPolicy checks if the decision allows the detected PII types
func (r *OpenAIRouter) checkPIIPolicy(ctx *RequestContext, detectedPII []string, decisionName string) *ext_proc.ProcessingResponse {
	// Check if the decision passes PII policy
	allowed, deniedPII, err := r.PIIChecker.CheckPolicy(decisionName, detectedPII)
	if err != nil {
		logging.Errorf("Error checking PII policy for decision %s: %v", decisionName, err)
		return nil
	}

	if allowed {
		return nil
	}

	// Decision violates PII policy - return error
	logging.Warnf("Decision %s violates PII policy, blocking request", decisionName)
	logging.LogEvent("routing_block", map[string]interface{}{
		"reason_code": "pii_policy_denied",
		"request_id":  ctx.RequestID,
		"decision":    decisionName,
		"denied_pii":  deniedPII,
	})
	metrics.RecordRequestError(decisionName, "pii_policy_denied")

	piiResponse := http.CreatePIIViolationResponse(decisionName, deniedPII, ctx.ExpectStreamingResponse, decisionName, ctx.VSRSelectedCategory, ctx.VSRMatchedKeywords)
	return piiResponse
}
