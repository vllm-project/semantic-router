package config

import "fmt"

func validateDecisionAutomaticOutput(decision Decision) error {
	params := decision.GetRequestParamsConfig()
	if params == nil || !params.DefaultMaxTokens.IsAuto() {
		return nil
	}
	if shadow := decision.GetShadowDispatchConfig(); shadow != nil && shadow.Enabled {
		return fmt.Errorf("decision %q: automatic output does not support shadow_dispatch", decision.Name)
	}
	if decision.Algorithm == nil {
		return nil
	}
	if IsLooperAlgorithmType(decision.Algorithm.Type) {
		return fmt.Errorf("decision %q: automatic output does not support Looper algorithms", decision.Name)
	}
	if decision.Algorithm.Type == DecisionAlgorithmMultiFactor && decision.Algorithm.MultiFactor != nil && decision.Algorithm.MultiFactor.ExpectedOutputTokens == nil {
		return fmt.Errorf("decision %q: automatic output with multi_factor requires expected_output_tokens as a cost forecast", decision.Name)
	}
	return nil
}
