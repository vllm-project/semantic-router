package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

func TestJailbreakUnreadContentUsesConfiguredPolicy(t *testing.T) {
	reading := systemOneReading{}.withJailbreakInput(&JailbreakInput{
		Current: []JailbreakContent{{Role: "user", Text: "Describe the attachment.", Incomplete: true}},
	})
	for _, tc := range []struct {
		policy    string
		matchType string
		withScore bool
		error     string
	}{
		{policy: config.SystemOneUnscannedMatch, matchType: JailbreakUnscannedType},
		{policy: config.SystemOneUnscannedNoMatch},
		{policy: config.SystemOneUnscannedUnknown, matchType: JailbreakUnscannedType, error: "systemone_unreadable_user_content"},
		{policy: config.SystemOneUnscannedBestEffort, withScore: true},
	} {
		t.Run(tc.policy, func(t *testing.T) {
			decider := jailbreakAnswering("benign")
			results := evaluateJailbreakReading(t, config.JailbreakRule{Name: "guard", Deployment: "kai", OnUnscanned: tc.policy}, decider, reading)
			if len(decider.asked) != 1 || askedText(decider.asked[0]) != "Describe the attachment." {
				t.Fatalf("the guard must still read the available text, calls: %+v", decider.asked)
			}
			if results.JailbreakType != tc.matchType || results.JailbreakDetected != (tc.matchType != "") ||
				results.JailbreakScoreAvailable != tc.withScore || results.SignalErrors["jailbreak:guard"] != tc.error {
				t.Fatalf("unexpected verdict: type=%q matched=%v score available=%v errors=%v",
					results.JailbreakType, results.JailbreakDetected, results.JailbreakScoreAvailable, results.SignalErrors)
			}
		})
	}
}

func TestJailbreakUnreadContentDoesNotUndoTextMatch(t *testing.T) {
	reading := systemOneReading{}.withJailbreakInput(&JailbreakInput{
		Current: []JailbreakContent{{Role: "user", Text: "Text under review.", Incomplete: true}},
	})
	results := evaluateJailbreakReading(t, config.JailbreakRule{
		Name: "guard", Deployment: "kai", OnUnscanned: config.SystemOneUnscannedNoMatch,
	}, jailbreakAnswering("jailbreak"), reading)
	if !results.JailbreakDetected || !results.JailbreakScoreAvailable || results.JailbreakType != presets.Jailbreak {
		t.Fatalf("a policy for unread content must not override read evidence: detected=%v score available=%v type=%q",
			results.JailbreakDetected, results.JailbreakScoreAvailable, results.JailbreakType)
	}
}

func TestJailbreakUnreadContentIgnoresToolResultsOutsideTheCurrentTurn(t *testing.T) {
	reading := (systemOneReading{toolResultsIncomplete: true}).withJailbreakInput(&JailbreakInput{
		Current: []JailbreakContent{{Role: "user", Text: "Summarize the record."}},
		History: []JailbreakContent{{Role: "tool", Incomplete: true}},
	})
	results := evaluateJailbreakReading(t, config.JailbreakRule{
		Name: "guard", Deployment: "kai", OnUnscanned: config.SystemOneUnscannedMatch,
		Input: config.JailbreakRuleInput{History: true, ToolResults: true},
	}, jailbreakAnswering("benign"), reading)
	if results.JailbreakDetected || !results.JailbreakScoreAvailable || len(results.SignalErrors) != 0 {
		t.Fatalf("a tool result outside the rule input must not affect coverage: detected=%v score available=%v errors=%v",
			results.JailbreakDetected, results.JailbreakScoreAvailable, results.SignalErrors)
	}
}
