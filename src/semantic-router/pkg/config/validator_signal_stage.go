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
	if err := validateDecisionsReadRequestStageSignals(cfg); err != nil {
		return err
	}
	warnResponseJailbreakPluginOverlap(cfg)
	return nil
}

// validateDecisionsReadRequestStageSignals rejects a decision rule that names a
// response-direction jailbreak rule. Decisions are selected while the request
// is being routed and the model has not answered, so the rule could only ever
// read as unknown there; the observation is consumed by the selected decision's
// response_jailbreak plugin once the response exists. Caught at load rather
// than as a decision that silently never matches.
func validateDecisionsReadRequestStageSignals(cfg *RouterConfig) error {
	for _, decision := range cfg.AllRoutingDecisions() {
		if rule, ok := cfg.decisionReadsResponseSignal(&decision.Rules); ok {
			return fmt.Errorf("decision %q reads jailbreak rule %q, which has direction: response; a response-direction rule is consumed by the selected decision's response_jailbreak plugin, not by decision rules", decision.Name, rule)
		}
	}
	return nil
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
