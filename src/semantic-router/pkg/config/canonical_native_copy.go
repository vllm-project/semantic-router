package config

// cloneNativeAlgorithm keeps exported and newly normalized policy values from
// mutating a retained generation's stage graph, thresholds or artifact identity.
func cloneNativeAlgorithm(source *AlgorithmConfig) *AlgorithmConfig {
	if !source.IsNative() {
		return source
	}
	result := *source
	result.Budget = cloneNativeValue(source.Budget)
	escalate := *source.Escalate
	result.Escalate = &escalate
	escalate.Quality = cloneNativeValue(source.Escalate.Quality)
	if escalate.Quality != nil {
		escalate.Quality.Accept = cloneNativeAccept(source.Escalate.Quality.Accept)
		escalate.Quality.MaxRisk = cloneNativeValue(source.Escalate.Quality.MaxRisk)
	}
	escalate.Stages = append([]EscalateStage(nil), source.Escalate.Stages...)
	for index, stage := range source.Escalate.Stages {
		escalate.Stages[index].Enabled = cloneNativeValue(stage.Enabled)
		escalate.Stages[index].Accept = cloneNativeAccept(stage.Accept)
		escalate.Stages[index].Generation = cloneNativeValue(stage.Generation)
	}
	return &result
}

func cloneNativeAccept(source *EscalateAccept) *EscalateAccept {
	if source == nil {
		return nil
	}
	result := &EscalateAccept{Native: append([]NativeAcceptanceRule(nil), source.Native...)}
	for index, rule := range source.Native {
		result.Native[index].State = cloneNativeValue(rule.State)
		result.Native[index].NumericPredicate = NumericPredicate{
			GT: cloneNativeValue(rule.GT), GTE: cloneNativeValue(rule.GTE),
			LT: cloneNativeValue(rule.LT), LTE: cloneNativeValue(rule.LTE),
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
