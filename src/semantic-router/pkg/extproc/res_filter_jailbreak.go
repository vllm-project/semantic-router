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
	decisionName := requestDecisionStateKey(ctx)

	// With response-direction rules declared, detection already ran as a
	// response-stage signal and this plugin only enforces. Without them it
	// still owns detection, so an existing configuration keeps working
	// unchanged.
	if r.responseJailbreakSignalDeclared(ctx) {
		return r.enforceResponseJailbreakFromSignal(ctx, decisionName)
	}
	return r.detectAndEnforceResponseJailbreak(ctx, assistantContent, decisionName)
}

// enforceResponseJailbreakFromSignal acts on the published response-stage
// signal. The plugin's own threshold is not consulted: two thresholds for one
// detection can disagree, so config loading reports a plugin threshold as
// ignored rather than letting it decide silently.
func (r *OpenAIRouter) enforceResponseJailbreakFromSignal(
	ctx *RequestContext,
	decisionName string,
) *ext_proc.ProcessingResponse {
	matched, resolved := r.responseJailbreakSignalOutcome(ctx)
	if !resolved {
		classifier := r.classifierForRequest(ctx)
		metrics.RecordPluginExecution("response_jailbreak", decisionName, "unresolved", 0)
		return r.responseJailbreakOnClassifyError(ctx, responseJailbreakFailsClosed(classifierConfig(classifier)), decisionName, 0)
	}
	if !matched {
		metrics.RecordPluginExecution("response_jailbreak", decisionName, "not_detected", 0)
		logging.Debugf("No jailbreak detected in response: score_available=%v", ctx.VSRResponseJailbreakScoreAvailable)
		return nil
	}

	ctx.ResponseJailbreakDetected = true
	ctx.ResponseJailbreakType = ctx.VSRResponseJailbreakType
	ctx.ResponseJailbreakConfidence = ctx.VSRResponseJailbreakRisk
	ctx.ResponseJailbreakScoreAvailable = ctx.VSRResponseJailbreakScoreAvailable
	ctx.ResponseJailbreakDecision = ctx.VSRResponseJailbreakDecision

	metrics.RecordPluginExecution("response_jailbreak", decisionName, "detected", 0)
	logging.Warnf("Response jailbreak detected: type=%s, score_available=%v, rules=%v, decision=%s",
		ctx.VSRResponseJailbreakType, ctx.VSRResponseJailbreakScoreAvailable, ctx.VSRMatchedResponseJailbreak, decisionName)

	if r.getResponseJailbreakAction(ctx.VSRSelectedDecision) == "block" {
		logging.Infof("Response jailbreak action is 'block', returning error response")
		return r.createErrorResponse(403, "Response blocked: jailbreak content detected in LLM output")
	}
	return nil
}

// detectAndEnforceResponseJailbreak is the path for a configuration that has
// not declared a response-direction jailbreak rule yet, where the plugin still
// owns detection.
func (r *OpenAIRouter) detectAndEnforceResponseJailbreak(
	ctx *RequestContext,
	assistantContent string,
	decisionName string,
) *ext_proc.ProcessingResponse {
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
	// Scans the whole response in chunks and thresholds P(jailbreak) rather than
	// the winning class's confidence, so this surface answers the same question
	// as the routing signal and the classification API on the same text.
	verdict, err := classifier.CheckForJailbreakVerdict(selectionRequestContext(ctx), assistantContent, threshold)
	latency := time.Since(start).Seconds()

	if err != nil {
		logging.Errorf("Response jailbreak detection failed: %v", err)
		metrics.RecordPluginError("response_jailbreak", "detection_error")
		return r.responseJailbreakOnClassifyError(ctx, responseJailbreakFailsClosed(classifierConfig(classifier)), decisionName, latency)
	}

	if verdict.Detected {
		ctx.ResponseJailbreakDetected = true
		ctx.ResponseJailbreakType = verdict.Label
		ctx.ResponseJailbreakDecision = verdict.Decision
		if verdict.RiskScore != nil {
			ctx.ResponseJailbreakConfidence = *verdict.RiskScore
			ctx.ResponseJailbreakScoreAvailable = true
		}

		metrics.RecordPluginExecution("response_jailbreak", decisionName, "detected", latency)
		logging.Warnf("Response jailbreak detected: type=%s, probabilities_available=%v", verdict.Label, verdict.RiskScore != nil)

		action := r.getResponseJailbreakAction(ctx.VSRSelectedDecision)
		if action == "block" {
			logging.Infof("Response jailbreak action is 'block', returning error response")
			return r.createErrorResponse(403, "Response blocked: jailbreak content detected in LLM output")
		}
		logging.Infof("Response jailbreak detected, action is '%s'", action)
	} else {
		metrics.RecordPluginExecution("response_jailbreak", decisionName, "not_detected", latency)
		logging.Debugf("No jailbreak detected in response: probabilities_available=%v", verdict.RiskScore != nil)
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
	ctx.ResponseJailbreakConfidence = 0
	ctx.ResponseJailbreakScoreAvailable = false
	ctx.ResponseJailbreakDecision = nil

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

	decision := ctx.VSRSelectedDecision
	if decision == nil {
		return false
	}

	rjCfg := decision.GetResponseJailbreakConfig()
	if rjCfg == nil || !rjCfg.Enabled {
		logging.Debugf("Skipping response jailbreak detection: not enabled for decision %s",
			decision.Name)
		return false
	}

	return true
}

// responseJailbreakPluginAction is the action the selected decision's
// response_jailbreak plugin applies to a detection, or "" when the decision
// carries no enabled plugin and the observation is recorded only.
func (r *OpenAIRouter) responseJailbreakPluginAction(ctx *RequestContext) string {
	if ctx == nil || ctx.VSRSelectedDecision == nil {
		return ""
	}
	plugin := ctx.VSRSelectedDecision.GetResponseJailbreakConfig()
	if plugin == nil || !plugin.Enabled {
		return ""
	}
	return r.getResponseJailbreakAction(ctx.VSRSelectedDecision)
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

// publishResponseJailbreakSignal records the response-stage observation where
// every other signal is recorded, under the same "jailbreak:<rule>" key a
// request-direction rule uses, so Router Replay and the selected decision's
// plugins read one shape.
//
// The plugin still enforces. This only publishes the evidence it acts on.
func (r *OpenAIRouter) publishResponseJailbreakSignal(ctx *RequestContext, rules []config.JailbreakRule, scan *classification.JailbreakScan) {
	if ctx == nil {
		return
	}
	signal := classification.EvaluateResponseJailbreakSignal(rules, scan)
	if signal == nil {
		return
	}
	ctx.VSRMatchedResponseJailbreak = append(ctx.VSRMatchedResponseJailbreak, signal.MatchedRules...)
	recordResponseSignal(ctx, signal.Confidences, signal.Errors)
}

// recordResponseSignal merges a response-stage observation's per-rule scores
// and failures into the request's signal maps, where request-stage signals
// already live.
func recordResponseSignal(ctx *RequestContext, confidences map[string]float64, errors map[string]string) {
	if len(confidences) > 0 && ctx.VSRSignalConfidences == nil {
		ctx.VSRSignalConfidences = make(map[string]float64, len(confidences))
	}
	for key, value := range confidences {
		ctx.VSRSignalConfidences[key] = value
	}
	if len(errors) > 0 && ctx.VSRSignalErrors == nil {
		ctx.VSRSignalErrors = make(map[string]string, len(errors))
	}
	for key, value := range errors {
		ctx.VSRSignalErrors[key] = value
	}
}
