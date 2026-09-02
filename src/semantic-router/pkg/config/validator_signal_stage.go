package config

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// validateSignalStageContracts checks what a response-direction signal rule
// implies for the decisions around it.
func validateSignalStageContracts(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	if err := validateDecisionStages(cfg); err != nil {
		return err
	}
	warnResponseJailbreakPluginOverlap(cfg)
	return nil
}

// validateDecisionStages keeps a configuration from routing nothing. A decision
// that reads a response-stage signal is skipped while the request is still
// being routed, because that signal does not exist yet, so a configuration made
// only of those has no decision left to select a model with. Caught here rather
// than as an unresolved request at runtime.
func validateDecisionStages(cfg *RouterConfig) error {
	decisions := cfg.AllRoutingDecisions()
	if len(decisions) == 0 {
		return nil
	}
	for _, decision := range decisions {
		if cfg.DecisionStage(&decision.Rules) == SignalStageRequest {
			return nil
		}
	}
	return fmt.Errorf("every decision reads a response-direction jailbreak rule, so no decision can select a model at request time; at least one decision must be resolvable from request-stage signals")
}

// warnResponseJailbreakPluginOverlap reports a decision that still carries its
// own response_jailbreak threshold once response-direction rules are declared.
//
// With rules present the plugin enforces on the signal and its threshold is no
// longer read. That is a real change in what a deployment blocks, so it is said
// out loud rather than left to be discovered. It is a diagnostic rather than a
// rejection because the threshold is still a valid field for a configuration
// that has not declared rules yet, and refusing to load would strand those.
func warnResponseJailbreakPluginOverlap(cfg *RouterConfig) {
	if len(cfg.ResponseJailbreakRules()) == 0 {
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
			"reason":    "routing.signals.jailbreak rules with direction: response own detection; move the threshold onto the rule",
		})
	}
}
