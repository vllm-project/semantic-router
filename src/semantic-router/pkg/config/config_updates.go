package config

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// Replace replaces the globally cached config. It is safe for concurrent readers.
func Replace(newCfg *RouterConfig) {
	decisionNames := make([]string, 0, len(newCfg.Decisions))
	for _, decision := range newCfg.Decisions {
		decisionNames = append(decisionNames, decision.Name)
	}
	logging.ComponentDebugEvent("config", "config_replace_started", map[string]interface{}{
		"decision_count": len(newCfg.Decisions),
		"decision_names": decisionNames,
	})

	configMu.Lock()
	config = newCfg
	configErr = nil
	configMu.Unlock()
}

// Get returns the current configuration.
func Get() *RouterConfig {
	configMu.RLock()
	defer configMu.RUnlock()
	return config
}
