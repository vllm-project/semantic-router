package config

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// validateHallucinationSignalContracts checks the hallucination rules of one
// routing profile and reports what they imply for the hallucination plugins
// around them.
func validateHallucinationSignalContracts(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	seen := make(map[string]struct{}, len(cfg.HallucinationRules))
	for _, rule := range cfg.HallucinationRules {
		name := strings.TrimSpace(rule.Name)
		if name == "" {
			return fmt.Errorf("routing.signals.hallucination: every rule needs a name")
		}
		if _, dup := seen[name]; dup {
			return fmt.Errorf("routing.signals.hallucination: duplicate rule name %q", name)
		}
		seen[name] = struct{}{}
	}
	warnHallucinationPluginNLIIgnored(cfg)
	warnHallucinationPluginOwnsDetection(cfg)
	return nil
}

// warnHallucinationPluginNLIIgnored reports a decision whose hallucination
// plugin still asks for use_nli once hallucination rules are declared. With
// rules present the rule decides how the answer is explained and the plugin's
// use_nli is no longer read. Said at load rather than left to be discovered:
// a deployment that relied on it would otherwise lose its explanations
// silently.
func warnHallucinationPluginNLIIgnored(cfg *RouterConfig) {
	if len(cfg.HallucinationRules) == 0 {
		return
	}
	for _, decision := range cfg.AllRoutingDecisions() {
		plugin := decision.GetHallucinationConfig()
		if plugin == nil || !plugin.UseNLI {
			continue
		}
		logging.ComponentWarnEvent("config", "hallucination_plugin_use_nli_ignored", map[string]interface{}{
			"decision": decision.Name,
			"reason":   "routing.signals.hallucination rules own detection; set use_nli on the rule",
		})
	}
}

// warnHallucinationPluginOwnsDetection reports a decision whose hallucination
// plugin runs with no hallucination rule declared. The plugin then classifies
// the answer itself, which is the compatibility path: nothing is published as
// a signal. Said at load, once, rather than on every request.
func warnHallucinationPluginOwnsDetection(cfg *RouterConfig) {
	if len(cfg.HallucinationRules) > 0 {
		return
	}
	for _, decision := range cfg.AllRoutingDecisions() {
		plugin := decision.GetHallucinationConfig()
		if plugin == nil || !plugin.Enabled {
			continue
		}
		logging.ComponentWarnEvent("config", "hallucination_plugin_owns_detection", map[string]interface{}{
			"decision": decision.Name,
			"reason":   "no routing.signals.hallucination rule is declared, so the plugin classifies the answer itself; declare one so the observation is published as a signal and the plugin only enforces",
		})
	}
}
