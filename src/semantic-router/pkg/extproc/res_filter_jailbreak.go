package extproc

import (
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// performSemanticResponseJailbreakDetection consumes protocol-neutral output,
// so response safety policy is identical for every public wire format.
func (r *OpenAIRouter) performSemanticResponseJailbreakDetection(
	ctx *RequestContext,
	response *llmprotocol.Response,
) *ext_proc.ProcessingResponse {
	return r.performResponseJailbreakDetectionText(ctx, semanticAssistantContent(response))
}

func (r *OpenAIRouter) performResponseJailbreakDetectionText(
	ctx *RequestContext,
	assistantContent string,
) *ext_proc.ProcessingResponse {
	if !r.shouldPerformResponseJailbreakDetection(ctx) {
		return nil
	}
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
	classifier := r.classifierForRequest(ctx)
	isJailbreak, jailbreakType, confidence, err := classifier.CheckForJailbreakWithThreshold(selectionRequestContext(ctx), assistantContent, threshold)
	latency := time.Since(start).Seconds()

	decisionName := requestDecisionStateKey(ctx)

	if err != nil {
		logging.Errorf("Response jailbreak detection failed: %v", err)
		metrics.RecordPluginError("response_jailbreak", "detection_error")
		return r.responseJailbreakOnClassifyError(ctx, responseJailbreakFailsClosed(classifierConfig(classifier)), decisionName, latency)
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

// classifierConfig returns the router config a classifier was built from, or
// nil when there is no classifier.
func classifierConfig(classifier *classification.Classifier) *config.RouterConfig {
	if classifier == nil {
		return nil
	}
	return classifier.Config
}

// responseJailbreakFailsClosed reports whether prompt_guard's on_error policy
// requires a classify failure to count as a detection.
func responseJailbreakFailsClosed(cfg *config.RouterConfig) bool {
	return cfg != nil && cfg.PromptGuard.IsBlock()
}

// responseJailbreakOnClassifyError applies prompt_guard's on_error policy to a
// response-path classify failure.
//
// The response path scans LLM output with the same prompt_guard backend the
// request path uses, so the same policy has to hold: under on_error: block an
// inference failure means the response could not be verified safe. It is
// recorded as a detection carrying the shared sentinel type, then reported
// through the decision's configured action - a real detection and a failure
// take the same route, so "block" is not hardcoded here.
func (r *OpenAIRouter) responseJailbreakOnClassifyError(ctx *RequestContext, failClosed bool, decisionName string, latency float64) *ext_proc.ProcessingResponse {
	if !failClosed {
		return nil
	}

	ctx.ResponseJailbreakDetected = true
	ctx.ResponseJailbreakType = classification.JailbreakClassificationErrorType
	ctx.ResponseJailbreakConfidence = 1.0

	metrics.RecordPluginExecution("response_jailbreak", decisionName, "fail_closed", latency)
	logging.Warnf("Response jailbreak classifier failed and prompt_guard.on_error is %q; treating the response as unverified",
		config.OnErrorBlock)

	if r.getResponseJailbreakAction(ctx.VSRSelectedDecision) == "block" {
		// Deliberately says no more than the real-detection message above:
		// telling a caller the guardrail itself is down hands an attacker a
		// probe for when the safety backend is offline. The cause is already in
		// the log line and the replay record.
		return r.createErrorResponse(403, "Response blocked: jailbreak content detected in LLM output")
	}
	return nil
}

// shouldPerformResponseJailbreakDetection checks whether response-level
// jailbreak detection should run for the current request.
func (r *OpenAIRouter) shouldPerformResponseJailbreakDetection(ctx *RequestContext) bool {
	classifier := r.classifierForRequest(ctx)
	if classifier == nil || !classifier.IsJailbreakEnabled() {
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
