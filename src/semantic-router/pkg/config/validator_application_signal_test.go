package config

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

func TestApplicationRuleOnUnknownAndOnUnscannedTakeTheirValues(t *testing.T) {
	for name, tc := range map[string]struct {
		spec applicationRuleSpec
		want string
	}{
		"match":          {applicationRuleSpec{signalType: SignalTypeJailbreak, preset: presets.Jailbreak, onUnknown: "match"}, ""},
		"no_match":       {applicationRuleSpec{signalType: SignalTypePII, preset: presets.PII, onUnknown: "no_match"}, ""},
		"another value":  {applicationRuleSpec{signalType: SignalTypeJailbreak, preset: presets.Jailbreak, onUnknown: "fail_request"}, "on_unknown must be match or no_match"},
		"response stage": {applicationRuleSpec{signalType: SignalTypeJailbreak, preset: presets.Jailbreak, stage: presets.StageResponse, onUnknown: "match"}, "response_guard plugin's on_unknown"},
		"best effort":    {applicationRuleSpec{signalType: SignalTypePII, preset: presets.PII, onUnscanned: "best_effort"}, ""},
		"v0.4 allow":     {applicationRuleSpec{signalType: SignalTypeJailbreak, preset: presets.Jailbreak, onUnscanned: "allow"}, "on_unscanned must be best_effort, match, no_match or unknown"},
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

func TestRequestStageGuardRulesFailClosedUnlessOnUnknownIsNoMatch(t *testing.T) {
	for name, tc := range map[string]struct {
		spec applicationRuleSpec
		want bool
	}{
		"jailbreak":          {applicationRuleSpec{signalType: SignalTypeJailbreak, stage: presets.StageRequest}, true},
		"jailbreak match":    {applicationRuleSpec{signalType: SignalTypeJailbreak, stage: presets.StageRequest, onUnknown: "match"}, true},
		"jailbreak no_match": {applicationRuleSpec{signalType: SignalTypeJailbreak, stage: presets.StageRequest, onUnknown: "no_match"}, false},
		"pii":                {applicationRuleSpec{signalType: SignalTypePII}, true},
		"response jailbreak": {applicationRuleSpec{signalType: SignalTypeJailbreak, stage: presets.StageResponse, onUnknown: "match"}, false},
		"pii match":          {applicationRuleSpec{signalType: SignalTypePII, onUnknown: "match"}, true},
		"domain":             {applicationRuleSpec{signalType: SignalTypeDomain, onUnknown: "match"}, false},
	} {
		if got := tc.spec.failsClosed(); got != tc.want {
			t.Errorf("%s: fails closed = %v, want %v", name, got, tc.want)
		}
	}
}

// A condition on a set or span rule with labels names one of them: the rule
// matches no other label, so such a condition is refused at load.
func TestALabelConditionOutsideASetOrSpanRulesLabelsIsRefused(t *testing.T) {
	for name, tc := range map[string]struct{ signals, counted, outside, want string }{
		"safety": {
			signals: "safety: [{name: violent, labels: [S1]}]",
			counted: "{type: safety, name: violent, label: S1}",
			outside: "{type: safety, name: violent, label: S9}",
			want:    `decision 'route': safety condition "violent" names label "S9", which is not one of the rule's labels (S1)`,
		},
		"systemone set": {
			signals: `systemone: [{name: topics, labels: [billing], question: {type: set, instructions: "Which topics?", labels: [{key: billing}, {key: shipping}]}}]`,
			counted: "{type: systemone, name: topics, label: billing}",
			outside: "{type: systemone, name: topics, label: shipping}",
			want:    `decision 'route': systemone condition "topics" names label "shipping", which is not one of the rule's labels (billing)`,
		},
	} {
		doc := func(condition string) []byte {
			return []byte(`version: v0.5
providers:
  defaults: {model: general}
  models:
    - name: general
      backend_refs: [{name: local, endpoint: "127.0.0.1:8000", protocol: http, weight: 1}]
recipes:
  - name: default
    signals:
      ` + tc.signals + `
    decisions:
      - name: route
        rules: ` + condition + `
        modelRefs: [{model: general}]
`)
		}
		if _, err := ParseYAMLBytes(doc(tc.counted)); err != nil {
			t.Fatalf("%s: a condition on a counted label loads: %v", name, err)
		}
		if _, err := ParseYAMLBytes(doc(tc.outside)); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: error = %v, want %q", name, err, tc.want)
		}
	}
}
