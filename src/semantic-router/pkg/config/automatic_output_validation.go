package config

import "fmt"

func validateDecisionAutomaticOutput(decision Decision) error {
	params := decision.GetRequestParamsConfig()
	if params == nil || !params.DefaultMaxTokens.IsAuto() {
		return nil
	}
	if replay := decision.GetRouterReplayConfig(); replay != nil && replay.Shadow != nil {
		return fmt.Errorf("decision %q: automatic output does not support a router_replay shadow", decision.Name)
	}
	if decision.Algorithm == nil {
		return nil
	}
	if IsExecutorAlgorithmType(decision.Algorithm.Type) {
		return fmt.Errorf("decision %q: automatic output does not support executor algorithms", decision.Name)
	}
	if decision.Algorithm.Type == DecisionAlgorithmMultiFactor && decision.Algorithm.MultiFactor != nil && decision.Algorithm.MultiFactor.ExpectedOutputTokens == nil {
		return fmt.Errorf("decision %q: automatic output with multi_factor requires expected_output_tokens as a cost forecast", decision.Name)
	}
	return nil
}

// validateDecisionAutomaticOutputCandidates refuses automatic output over a
// delegated candidate: the render that sizes the output is a Router-own call.
func validateDecisionAutomaticOutputCandidates(cfg *RouterConfig, decision Decision) error {
	params := decision.GetRequestParamsConfig()
	if cfg == nil || params == nil || !params.DefaultMaxTokens.IsAuto() {
		return nil
	}
	for _, ref := range decision.ModelRefs {
		if path, mode, ok := cfg.CredentialBackendOf(ref.Model); ok {
			return requestCredentialRouterOwnError(path, mode, fmt.Sprintf("the automatic output render (decisions[%s]) calls it as the Router", decision.Name))
		}
	}
	return nil
}
