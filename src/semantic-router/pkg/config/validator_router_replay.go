package config

import "fmt"

// Tool results are raw client data. Storing them while Replay keeps only a
// verified prompt would contradict the declared policy, so the combination is
// refused wherever it can take effect. Under none nothing is stored, tool
// results included.
func validateGlobalRouterReplayCaptureContracts(cfg *RouterConfig) error {
	policy := RouterReplayPolicyOf(cfg.GlobalPlugins)
	if policy != nil && policy.CaptureToolResults && policy.VerifiesPrompt() {
		return fmt.Errorf("global.plugins[router_replay].configuration: capture_tool_results: true requires prompt_capture: raw, not verified")
	}
	return nil
}

func validateDecisionRouterReplayCaptureContracts(cfg *RouterConfig) error {
	if policy := RouterReplayPolicyOf(cfg.DefaultPlugins); policy != nil && policy.CaptureToolResults && policy.VerifiesPrompt() {
		return fmt.Errorf("%s: router_replay, layered over global.plugins: capture_tool_results: true requires prompt_capture: raw, not verified",
			cfg.ScopePath("plugins"))
	}
	for i := range cfg.Decisions {
		decision := &cfg.Decisions[i]
		if decision.GetPlugin(DecisionPluginRouterReplay) == nil {
			continue
		}
		policy := cfg.EffectiveRouterReplayConfig(decision)
		if policy != nil && policy.CaptureToolResults && policy.VerifiesPrompt() {
			return fmt.Errorf("decision %q router_replay: capture_tool_results: true requires prompt_capture: raw, not verified; "+
				"set capture_tool_results: false on this decision or prompt_capture: raw", decision.Name)
		}
	}
	return nil
}
