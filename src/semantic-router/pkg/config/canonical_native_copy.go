package config

// cloneNativeAlgorithm keeps exported and newly normalized policy values from
// mutating a retained generation's stage graph, thresholds or artifact identity.
func cloneNativeAlgorithm(source *AlgorithmConfig) *AlgorithmConfig {
	if !source.IsNative() {
		return source
	}
	result := *source
	result.Budget = cloneNativeValue(source.Budget)
	result.Quality = cloneNativeValue(source.Quality)
	if result.Quality != nil {
		result.Quality.Acceptance = cloneNativeAcceptance(source.Quality.Acceptance)
		result.Quality.MaxRisk = cloneNativeValue(source.Quality.MaxRisk)
	}
	result.Stages = append([]CascadeStage(nil), source.Stages...)
	for index, stage := range source.Stages {
		result.Stages[index].Enabled = cloneNativeValue(stage.Enabled)
		result.Stages[index].Accept = cloneNativeAcceptance(stage.Accept)
		result.Stages[index].Generation = cloneNativeValue(stage.Generation)
	}
	return &result
}

func cloneNativeAcceptance(source *NativeAcceptance) *NativeAcceptance {
	if source == nil {
		return nil
	}
	result := &NativeAcceptance{Rules: append([]NativeAcceptanceRule(nil), source.Rules...)}
	for index, rule := range source.Rules {
		result.Rules[index].State = cloneNativeValue(rule.State)
		result.Rules[index].Predicate = NumericPredicate{
			GT: cloneNativeValue(rule.Predicate.GT), GTE: cloneNativeValue(rule.Predicate.GTE),
			LT: cloneNativeValue(rule.Predicate.LT), LTE: cloneNativeValue(rule.Predicate.LTE),
		}
	}
	return result
}

func cloneNativeValue[T any](source *T) *T {
	if source == nil {
		return nil
	}
	value := *source
	return &value
}
