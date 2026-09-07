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
		r.publishResponseJailbreakSignal(ctx, rules, nil)
		return
	}
	if assistantContent == "" {
		// Nothing to score. semanticAssistantContent collects text and refusal
		// blocks only, so a response made entirely of tool calls or media lands
		// here, and "could not look" is not "looked and found nothing".
		r.publishResponseJailbreakSignal(ctx, rules, nil)
		return
	}

	// One scan serves every rule: they ask the same model the same question
	// about the same text and differ only in where they draw the line, so the
	// scan draws none and each rule thresholds the score itself. It also
	// reports whether the whole response was scored, which no single threshold
	// can answer for every rule.
	start := time.Now()
	scan, err := classifier.ScanJailbreakRisk(selectionRequestContext(ctx), assistantContent)
	latency := time.Since(start).Seconds()

	if err != nil {
		logging.Errorf("Response jailbreak signal evaluation failed: %v", err)
		metrics.RecordPluginError("response_jailbreak", "detection_error")
		r.publishResponseJailbreakSignal(ctx, rules, nil)
		return
	}
	for _, rule := range rules {
		metrics.RecordSignalExtraction(config.SignalTypeJailbreak, rule.Name, latency)
	}
	ctx.VSRResponseJailbreakType = scan.Type
	ctx.VSRResponseJailbreakRisk = scan.RiskScore
	r.publishResponseJailbreakSignal(ctx, rules, &scan)
}

// responseJailbreakSignalDeclared reports whether the selected recipe declares
// a response-direction rule, and therefore whether the plugin should read the
// signal instead of classifying again.
func (r *OpenAIRouter) responseJailbreakSignalDeclared(ctx *RequestContext) bool {
	return len(r.responseJailbreakRules(ctx)) > 0
}

// responseJailbreakSignalOutcome reads the published signal back for the
// plugin: whether any declared rule matched, and whether the detector resolved.
//
// A match outranks another rule's failure. A partial scan can match a
// permissive rule and leave a stricter one unresolved, and reporting that as
// unresolved would send a real detection through on_error, which under
// on_error: allow would let the matched response through.
func (r *OpenAIRouter) responseJailbreakSignalOutcome(ctx *RequestContext) (matched bool, resolved bool) {
	if ctx == nil {
		return false, false
	}
	if len(ctx.VSRMatchedResponseJailbreak) > 0 {
		return true, true
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
