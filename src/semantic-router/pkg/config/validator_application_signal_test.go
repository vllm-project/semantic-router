package config

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

func TestApplicationRuleOnUnknownIsMatchOrNoMatchAtTheRequestStage(t *testing.T) {
	for name, tc := range map[string]struct {
		spec applicationRuleSpec
		want string
	}{
		"match":          {applicationRuleSpec{signalType: SignalTypeJailbreak, preset: presets.Jailbreak, onUnknown: "match"}, ""},
		"no_match":       {applicationRuleSpec{signalType: SignalTypePII, preset: presets.PII, onUnknown: "no_match"}, ""},
		"another value":  {applicationRuleSpec{signalType: SignalTypeJailbreak, preset: presets.Jailbreak, onUnknown: "fail_request"}, "on_unknown must be match or no_match"},
		"response stage": {applicationRuleSpec{signalType: SignalTypeJailbreak, preset: presets.Jailbreak, stage: presets.StageResponse, onUnknown: "match"}, "response_guard plugin's on_unknown"},
	} {
		err := validateApplicationRuleSpec(tc.spec)
		if tc.want == "" && err != nil {
			t.Errorf("%s: %v", name, err)
		}
		if tc.want != "" && (err == nil || !strings.Contains(err.Error(), tc.want)) {
			t.Errorf("%s: error = %v, want %q", name, err, tc.want)
		}
	}
}

func TestRequestStageGuardRulesFailClosedWithOnUnknownMatch(t *testing.T) {
	for name, tc := range map[string]struct {
		spec applicationRuleSpec
		want bool
	}{
		"jailbreak":          {applicationRuleSpec{signalType: SignalTypeJailbreak, stage: presets.StageRequest}, false},
		"jailbreak match":    {applicationRuleSpec{signalType: SignalTypeJailbreak, stage: presets.StageRequest, onUnknown: "match"}, true},
		"response jailbreak": {applicationRuleSpec{signalType: SignalTypeJailbreak, stage: presets.StageResponse, onUnknown: "match"}, false},
		"pii match":          {applicationRuleSpec{signalType: SignalTypePII, onUnknown: "match"}, true},
		"domain":             {applicationRuleSpec{signalType: SignalTypeDomain, onUnknown: "match"}, false},
	} {
		if got := tc.spec.failsClosed(); got != tc.want {
			t.Errorf("%s: fails closed = %v, want %v", name, got, tc.want)
		}
	}
}
