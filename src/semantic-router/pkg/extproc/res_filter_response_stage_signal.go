package extproc

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// responseJailbreakRules returns the response-direction jailbreak rules of the
// recipe selected for this request. Classification is recipe-scoped: the
// classifier a request gets is built from ConfigForRecipe, so its Config holds
// the recipe's rules, while the root config only describes the default recipe.
// A rule declared on one entrypoint's recipe must not score, or fail to score,
// another entrypoint's responses.
func (r *OpenAIRouter) responseJailbreakRules(ctx *RequestContext) []config.JailbreakRule {
	return classifierConfig(r.classifierForRequest(ctx)).ResponseJailbreakRules()
}

// evaluateResponseJailbreakSignal scores the model's output against the
// jailbreak rules declared with direction: response.
//
// It is driven by the rules, not by the plugin: a signal that only existed
// while an enforcement plugin happened to be enabled would not be a signal, it
// would be plugin state with a signal's name. The observation is published
// whether or not the selected decision carries a plugin that acts on it.
func (r *OpenAIRouter) evaluateResponseJailbreakSignal(ctx *RequestContext, assistantContent string) {
	if ctx == nil || r == nil {
		return
	}
	rules := r.responseJailbreakRules(ctx)
	if len(rules) == 0 {
		return
	}

	classifier := r.classifierForRequest(ctx)
	if classifier == nil || !classifier.IsJailbreakEnabled() {
		// Declared but unbacked. Unresolved rather than clean, so the plugin
		// applies its failure policy instead of reading silence as safe.
		r.publishResponseJailbreakSignal(ctx, rules, 0, false)
		return
	}
	if assistantContent == "" {
		// Nothing to score. semanticAssistantContent collects text and refusal
		// blocks only, so a response made entirely of tool calls or media lands
		// here, and "could not look" is not "looked and found nothing".
		r.publishResponseJailbreakSignal(ctx, rules, 0, false)
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
		r.publishResponseJailbreakSignal(ctx, rules, 0, false)
		return
	}
	for _, rule := range rules {
		metrics.RecordSignalExtraction(config.SignalTypeJailbreak, rule.Name, latency)
	}
	ctx.VSRResponseJailbreakType = jailbreakType
	ctx.VSRResponseJailbreakRisk = riskScore
	r.publishResponseJailbreakSignal(ctx, rules, riskScore, true)
}

// lowestResponseJailbreakThreshold returns the most permissive declared
// threshold, so the single classify call is never the reason a rule misses.
func lowestResponseJailbreakThreshold(rules []config.JailbreakRule) float32 {
	lowest := float32(1)
	for _, rule := range rules {
		if rule.Threshold < lowest {
			lowest = rule.Threshold
		}
	}
	return lowest
}

// responseJailbreakSignalDeclared reports whether the selected recipe declares
// a response-direction rule, and therefore whether the plugin should read the
// signal instead of classifying again.
func (r *OpenAIRouter) responseJailbreakSignalDeclared(ctx *RequestContext) bool {
	return len(r.responseJailbreakRules(ctx)) > 0
}

// responseJailbreakSignalOutcome reads the published signal back for the
// plugin: whether any declared rule matched, and whether the detector resolved.
func (r *OpenAIRouter) responseJailbreakSignalOutcome(ctx *RequestContext) (matched bool, resolved bool) {
	if ctx == nil {
		return false, false
	}
	for _, rule := range r.responseJailbreakRules(ctx) {
		if _, failed := ctx.VSRSignalErrors[signalKey(config.SignalTypeJailbreak, rule.Name)]; failed {
			return false, false
		}
	}
	return len(ctx.VSRMatchedResponseJailbreak) > 0, true
}

func signalKey(signalType, name string) string {
	return signalType + ":" + name
}
