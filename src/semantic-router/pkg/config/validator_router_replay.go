package config

import "fmt"

// Tool results are raw client data. Storing them while Replay withholds
// personal data would contradict the declared policy, so the combination is
// refused wherever it can take effect.
func validateGlobalRouterReplayCaptureContracts(cfg *RouterConfig) error {
	defaults := cfg.RouterReplay.captureDefaults()
	if defaults.CaptureToolResults && !defaults.CapturesPersonalData() {
		return fmt.Errorf("global.services.router_replay: capture_tool_results: true requires capture_personal_data: true")
	}
	return nil
}

func validateDecisionRouterReplayCaptureContracts(cfg *RouterConfig) error {
	for i := range cfg.Decisions {
		decision := &cfg.Decisions[i]
		if decision.GetPlugin(DecisionPluginRouterReplay) == nil {
			continue
		}
		policy := cfg.EffectiveRouterReplayConfig(decision)
		if policy != nil && policy.CaptureToolResults && !policy.CapturesPersonalData() {
			return fmt.Errorf("decision %q router_replay: capture_tool_results: true requires capture_personal_data: true; "+
				"set capture_tool_results: false on this decision or allow personal data", decision.Name)
		}
	}
	return nil
}
