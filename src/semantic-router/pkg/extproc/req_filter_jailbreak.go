package extproc

import (
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"go.opentelemetry.io/otel/attribute"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/http"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/pii"
)

// performJailbreaks performs PII and jailbreak detection with category-specific settings
func (r *OpenAIRouter) performJailbreaks(ctx *RequestContext, userContent string, nonUserMessages []string, categoryName string) (*ext_proc.ProcessingResponse, bool) {
	// Perform PII classification on all message content
	allContent := pii.ExtractAllContent(userContent, nonUserMessages)

	// Check if jailbreak detection is enabled for this decision
	jailbreakEnabled := r.Classifier.IsJailbreakEnabled()
	if categoryName != "" && r.Config != nil {
		// Use decision-specific setting if available
		jailbreakEnabled = jailbreakEnabled && r.Config.IsJailbreakEnabledForDecision(categoryName)
	}

	// Get decision-specific threshold
	jailbreakThreshold := r.Config.PromptGuard.Threshold
	if categoryName != "" && r.Config != nil {
		jailbreakThreshold = r.Config.GetJailbreakThresholdForDecision(categoryName)
	}

	// Perform jailbreak detection on all message content
	if jailbreakEnabled {
		// Start jailbreak detection span
		spanCtx, span := tracing.StartSpan(ctx.TraceContext, tracing.SpanJailbreakDetection)
		defer span.End()

		startTime := time.Now()
		hasJailbreak, jailbreakDetections, err := r.Classifier.AnalyzeContentForJailbreakWithThreshold(allContent, jailbreakThreshold)
		detectionTime := time.Since(startTime).Milliseconds()

		tracing.SetSpanAttributes(span,
			attribute.Int64(tracing.AttrJailbreakDetectionTimeMs, detectionTime))

		if err != nil {
			logging.Errorf("Error performing jailbreak analysis: %v", err)
			tracing.RecordError(span, err)
			// Continue processing despite jailbreak analysis error
			metrics.RecordRequestError(ctx.RequestModel, "classification_failed")
		} else if hasJailbreak {
			// Find the first jailbreak detection for response
			var jailbreakType string
			var confidence float32
			for _, detection := range jailbreakDetections {
				if detection.IsJailbreak {
					jailbreakType = detection.JailbreakType
					confidence = detection.Confidence
					break
				}
			}

			tracing.SetSpanAttributes(span,
				attribute.Bool(tracing.AttrJailbreakDetected, true),
				attribute.String(tracing.AttrJailbreakType, jailbreakType),
				attribute.String(tracing.AttrSecurityAction, "blocked"))

			logging.Warnf("JAILBREAK ATTEMPT BLOCKED: %s (confidence: %.3f)", jailbreakType, confidence)

			// Return immediate jailbreak violation response
			// Structured log for security block
			logging.LogEvent("security_block", map[string]interface{}{
				"reason_code":    "jailbreak_detected",
				"jailbreak_type": jailbreakType,
				"confidence":     confidence,
				"request_id":     ctx.RequestID,
			})
			// Count this as a blocked request
			metrics.RecordRequestError(ctx.RequestModel, "jailbreak_block")
			jailbreakResponse := http.CreateJailbreakViolationResponse(jailbreakType, confidence, ctx.ExpectStreamingResponse)
			ctx.TraceContext = spanCtx
			return jailbreakResponse, true
		} else {
			tracing.SetSpanAttributes(span,
				attribute.Bool(tracing.AttrJailbreakDetected, false))
			logging.Infof("No jailbreak detected in request content")
			ctx.TraceContext = spanCtx
		}
	}

	return nil, false
}
