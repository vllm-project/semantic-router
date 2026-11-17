package config

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

// An application rule reads every part its preset declares.
func TestAnApplicationRuleReadsThePartsItsPresetDeclares(t *testing.T) {
	cfg := &RouterConfig{}
	for name, tc := range map[string]struct {
		spec applicationRuleSpec
		want SystemOneInput
	}{
		"feedback":      {applicationRuleSpec{signalType: SignalTypeUserFeedback, preset: presets.Feedback, name: "wrong"}, SystemOneInput{PreviousAnswer: true}},
		"turn relation": {applicationRuleSpec{signalType: SignalTypeTurnRelation, preset: presets.TurnRelation, name: "reasked"}, SystemOneInput{PreviousTurns: 4}},
		"refusal":       {applicationRuleSpec{signalType: SignalTypeRefusal, preset: presets.Refusal, stage: presets.StageResponse, name: "refused"}, SystemOneInput{Answer: true}},
		"domain":        {applicationRuleSpec{signalType: SignalTypeDomain, preset: presets.Domain, name: "math"}, SystemOneInput{}},
	} {
		if got := cfg.compileApplicationRule(tc.spec).Rule.Input; got != tc.want {
			t.Errorf("%s: input = %+v, want %+v", name, got, tc.want)
		}
	}
}
