package config

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"

const SignalTypeSafety = "safety"

// EffectiveStage is the stage a safety rule is asked in.
func (r SafetyRule) EffectiveStage() string {
	if r.Stage == "" {
		return SystemOneStageRequest
	}
	return r.Stage
}

// EffectiveStage is the stage a jailbreak rule is asked in.
func (r JailbreakRule) EffectiveStage() string {
	if r.Stage == "" {
		return SystemOneStageRequest
	}
	return r.Stage
}

// ResponseThreshold is the score a response-stage jailbreak rule's predicate
// requires: its lower bound, else the neutral threshold, which a response
// stage asks with (no model has a calibrated one for it).
func (r JailbreakRule) ResponseThreshold() float32 {
	if predicate := r.Predicate; predicate != nil {
		if predicate.GTE != nil {
			return float32(*predicate.GTE)
		}
		if predicate.GT != nil {
			return float32(*predicate.GT)
		}
	}
	return presets.NeutralThreshold
}
