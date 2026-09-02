package extproc

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// evaluateResponseJailbreakSignal scores the model's output against the
// declared response_jailbreak rules.
//
// It is driven by the rules, not by the plugin: a signal that only existed
// while an enforcement plugin happened to be enabled would not be a signal, it
// would be plugin state with a signal's name. A decision can therefore read
// this without any plugin on it.
func (r *OpenAIRouter) evaluateResponseJailbreakSignal(ctx *RequestContext, assistantContent string) {
	if ctx == nil || r.Config == nil {
		return
	}
	rules := r.Config.ResponseJailbreakRules
	if len(rules) == 0 {
		return
	}

	classifier := r.classifierForRequest(ctx)
	if classifier == nil || !classifier.IsJailbreakEnabled() {
		// Declared but unbacked. Unresolved rather than clean, so a decision
		// resolves it through on_unknown instead of reading silence as safe.
		r.publishResponseJailbreakSignal(ctx, 0, false)
		return
	}
	if assistantContent == "" {
		// Nothing to score. semanticAssistantContent collects text and refusal
		// blocks only, so a response made entirely of tool calls or media lands
		// here, and "could not look" is not "looked and found nothing".
		r.publishResponseJailbreakSignal(ctx, 0, false)
		return
	}

	// One call serves every rule: they ask the same model the same question
	// about the same text and differ only in where they draw the line. The
	// threshold passed here only decides the boolean this call returns, which
	// is discarded; each rule thresholds the score itself.
	start := time.Now()
	_, jailbreakType, _, riskScore, err := classifier.CheckForJailbreakRiskWithThreshold(
		selectionRequestContext(ctx), assistantContent, lowestResponseJailbreakThreshold(rules))
	latency := time.Since(start).Seconds()

	if err != nil {
		logging.Errorf("Response jailbreak signal evaluation failed: %v", err)
		metrics.RecordPluginError("response_jailbreak", "detection_error")
		r.publishResponseJailbreakSignal(ctx, 0, false)
		return
	}
	for _, rule := range rules {
		metrics.RecordSignalExtraction(config.SignalTypeResponseJailbreak, rule.Name, latency)
	}
	ctx.VSRResponseJailbreakType = jailbreakType
	ctx.VSRResponseJailbreakRisk = riskScore
	r.publishResponseJailbreakSignal(ctx, riskScore, true)
}

// lowestResponseJailbreakThreshold returns the most permissive declared
// threshold, so the single classify call is never the reason a rule misses.
func lowestResponseJailbreakThreshold(rules []config.ResponseJailbreakRule) float32 {
	lowest := float32(1)
	for _, rule := range rules {
		if rule.Threshold < lowest {
			lowest = rule.Threshold
		}
	}
	return lowest
}

// responseJailbreakSignalDeclared reports whether the signal is configured, and
// therefore whether the plugin should read it instead of classifying again.
func (r *OpenAIRouter) responseJailbreakSignalDeclared() bool {
	return r.Config != nil && len(r.Config.ResponseJailbreakRules) > 0
}

// responseJailbreakSignalOutcome reads the published signal back for the
// plugin: whether any declared rule matched, and whether the detector resolved.
func responseJailbreakSignalOutcome(ctx *RequestContext) (matched bool, resolved bool) {
	if ctx == nil {
		return false, false
	}
	for key := range ctx.VSRSignalErrors {
		if isResponseJailbreakSignalKey(key) {
			return false, false
		}
	}
	return len(ctx.VSRMatchedResponseJailbreak) > 0, true
}

func isResponseJailbreakSignalKey(key string) bool {
	prefix := classification.ResponseJailbreakSignalKeyPrefix
	return len(key) > len(prefix) && key[:len(prefix)] == prefix
}
