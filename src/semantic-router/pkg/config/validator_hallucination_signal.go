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
	warnHallucinationPluginOwnsDetection(cfg)
	return nil
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
