package extproc

import (
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// performResponseJailbreakDetection runs the jailbreak classifier on the LLM
// response body to catch adversarial content that passed input-level detection.
// Returns a blocking response if the action is "block"; otherwise returns nil
// and sets ctx.ResponseJailbreakDetected for downstream handling.
func (r *OpenAIRouter) performResponseJailbreakDetection(ctx *RequestContext, responseBody []byte) *ext_proc.ProcessingResponse {
	if !r.shouldPerformResponseJailbreakDetection(ctx) {
		return nil
	}

	assistantContent := extractAssistantContentFromResponse(responseBody)
	if assistantContent == "" {
		logging.Debugf("No assistant content to check for response jailbreak")
		return nil
	}

	rjCfg := ctx.VSRSelectedDecision.GetResponseJailbreakConfig()
	threshold := rjCfg.Threshold
	if threshold <= 0 && r.Config != nil {
		threshold = r.Config.PromptGuard.Threshold
	}
	if threshold <= 0 {
		threshold = 0.5
	}

	start := time.Now()
	isJailbreak, jailbreakType, confidence, err := r.Classifier.CheckForJailbreakWithThreshold(assistantContent, threshold)
	latency := time.Since(start).Seconds()

	decisionName := ""
	if ctx.VSRSelectedDecision != nil {
		decisionName = ctx.VSRSelectedDecision.Name
	}

	if err != nil {
		logging.Errorf("Response jailbreak detection failed: %v", err)
		metrics.RecordPluginError("response_jailbreak", "detection_error")
		return nil
	}

	if isJailbreak {
		ctx.ResponseJailbreakDetected = true
		ctx.ResponseJailbreakType = jailbreakType
		ctx.ResponseJailbreakConfidence = confidence

		metrics.RecordPluginExecution("response_jailbreak", decisionName, "detected", latency)
		logging.Warnf("Response jailbreak detected: type=%s, confidence=%.3f", jailbreakType, confidence)

		action := r.getResponseJailbreakAction(ctx.VSRSelectedDecision)
		if action == "block" {
			logging.Infof("Response jailbreak action is 'block', returning error response")
			return r.createErrorResponse(403, "Response blocked: jailbreak content detected in LLM output")
		}
		logging.Infof("Response jailbreak detected, action is '%s'", action)
	} else {
		metrics.RecordPluginExecution("response_jailbreak", decisionName, "not_detected", latency)
		logging.Debugf("No jailbreak detected in response: confidence=%.3f", confidence)
	}

	return nil
}

// shouldPerformResponseJailbreakDetection checks whether response-level
// jailbreak detection should run for the current request.
func (r *OpenAIRouter) shouldPerformResponseJailbreakDetection(ctx *RequestContext) bool {
	if r.Classifier == nil || !r.Classifier.IsJailbreakEnabled() {
		return false
	}

	if ctx.VSRSelectedDecision == nil {
		return false
	}

	rjCfg := ctx.VSRSelectedDecision.GetResponseJailbreakConfig()
	if rjCfg == nil || !rjCfg.Enabled {
		logging.Debugf("Skipping response jailbreak detection: not enabled for decision %s",
			ctx.VSRSelectedDecisionName)
		return false
	}

	return true
}

// getResponseJailbreakAction returns the configured action for response jailbreak.
// Defaults to "header".
func (r *OpenAIRouter) getResponseJailbreakAction(decision *config.Decision) string {
	if decision == nil {
		return "header"
	}

	rjCfg := decision.GetResponseJailbreakConfig()
	if rjCfg == nil {
		return "header"
	}

	action := rjCfg.Action
	if action == "" {
		return "header"
	}

	return action
}

// responseJailbreakWarningCode returns the response-warnings code for a detected
// response jailbreak, or "" when the configured action suppresses it. The
// jailbreak type and confidence detail stay in the replay record (#2204); the
// "block" action is handled earlier via an immediate error response.
func (r *OpenAIRouter) responseJailbreakWarningCode(ctx *RequestContext) string {
	if !ctx.ResponseJailbreakDetected {
		return ""
	}
	if r.getResponseJailbreakAction(ctx.VSRSelectedDecision) == "none" {
		logging.Infof("Response jailbreak detected but action is 'none', skipping warning")
		return ""
	}
	return headers.ResponseWarningJailbreak
}
