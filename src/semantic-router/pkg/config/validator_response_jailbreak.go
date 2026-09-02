package config

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// validateResponseJailbreakRules checks the response-stage jailbreak rules.
// A rule that cannot match is worse than a missing one: the decision reading it
// silently never fires, and the response goes out unchecked.
func validateResponseJailbreakRules(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	seen := make(map[string]bool, len(cfg.Signals.ResponseJailbreakRules))
	for i, rule := range cfg.Signals.ResponseJailbreakRules {
		if rule.Name == "" {
			return fmt.Errorf("routing.signals.response_jailbreak[%d]: name is required", i)
		}
		if seen[rule.Name] {
			return fmt.Errorf("routing.signals.response_jailbreak[%d]: duplicate name %q", i, rule.Name)
		}
		seen[rule.Name] = true
		// A threshold of 0 matches every response, including a benign one, and
		// an omitted threshold decodes to the same value. Both are almost
		// certainly a mistake rather than a request to flag everything.
		if rule.Threshold <= 0 || rule.Threshold > 1 {
			return fmt.Errorf("routing.signals.response_jailbreak[%d] %q: threshold must be greater than 0 and at most 1, got %v",
				i, rule.Name, rule.Threshold)
		}
	}
	warnResponseJailbreakPluginOverlap(cfg, len(cfg.Signals.ResponseJailbreakRules) > 0)
	return nil
}

// warnResponseJailbreakPluginOverlap reports a decision that still carries its
// own response_jailbreak threshold once rules are declared.
//
// With rules present the plugin enforces on the signal and its threshold is no
// longer read. That is a real change in what a deployment blocks, so it is said
// out loud rather than left to be discovered. It is a diagnostic rather than a
// rejection because the threshold is still a valid field for a configuration
// that has not declared rules yet, and refusing to load would strand those.
func warnResponseJailbreakPluginOverlap(cfg *RouterConfig, rulesDeclared bool) {
	if !rulesDeclared {
		return
	}
	for _, decision := range cfg.AllRoutingDecisions() {
		plugin := decision.GetResponseJailbreakConfig()
		if plugin == nil || plugin.Threshold <= 0 {
			continue
		}
		logging.ComponentWarnEvent("config", "response_jailbreak_plugin_threshold_ignored", map[string]interface{}{
			"decision":  decision.Name,
			"threshold": plugin.Threshold,
			"reason":    "routing.signals.response_jailbreak owns detection; move the threshold onto the rule",
		})
	}
}
