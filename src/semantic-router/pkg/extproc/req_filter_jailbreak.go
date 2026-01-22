package extproc

import (
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"go.opentelemetry.io/otel/attribute"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/http"
)

// performJailbreaks performs jailbreak detection with category-specific settings
// By default, only checks the current user message. Set include_history: true in plugin config to include conversation history.
func (r *OpenAIRouter) performJailbreaks(ctx *RequestContext, userContent string, nonUserMessages []string, categoryName string) (*ext_proc.ProcessingResponse, bool) {
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

	// Get decision-specific include_history setting
	includeHistory := false
	if categoryName != "" && r.Config != nil {
		includeHistory = r.Config.GetJailbreakIncludeHistoryForDecision(categoryName)
	}

	// Perform jailbreak detection
	if jailbreakEnabled {
		// Start jailbreak plugin span
		spanCtx, span := tracing.StartPluginSpan(ctx.TraceContext, "jailbreak", categoryName)

		// Build content to analyze based on include_history setting
		contentToAnalyze := []string{}
		if userContent != "" {
			contentToAnalyze = append(contentToAnalyze, userContent)
		}
		if includeHistory {
			contentToAnalyze = append(contentToAnalyze, nonUserMessages...)
		}

		startTime := time.Now()
		hasJailbreak, jailbreakDetections, err := r.Classifier.AnalyzeContentForJailbreakWithThreshold(contentToAnalyze, jailbreakThreshold)
		detectionTime := time.Since(startTime).Milliseconds()

		if err != nil {
			logging.Errorf("Error performing jailbreak analysis: %v", err)
			tracing.RecordError(span, err)
			tracing.EndPluginSpan(span, "error", detectionTime, "analysis_failed")
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

			// Keep legacy attributes for backward compatibility
			tracing.SetSpanAttributes(span,
				attribute.Bool(tracing.AttrJailbreakDetected, true),
				attribute.String(tracing.AttrJailbreakType, jailbreakType),
				attribute.String(tracing.AttrSecurityAction, "blocked"))

			// End plugin span with blocked status
			tracing.EndPluginSpan(span, "blocked", detectionTime, "jailbreak_detected:"+jailbreakType)

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
			// Keep legacy attributes for backward compatibility
			tracing.SetSpanAttributes(span,
				attribute.Bool(tracing.AttrJailbreakDetected, false))

			// End plugin span with success status
			tracing.EndPluginSpan(span, "success", detectionTime, "no_jailbreak_detected")

			logging.Infof("No jailbreak detected in request content")
			ctx.TraceContext = spanCtx
		}
	}

	return nil, false
}
