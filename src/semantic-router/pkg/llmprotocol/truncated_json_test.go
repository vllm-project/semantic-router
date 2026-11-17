package llmprotocol

import (
	"strings"
	"testing"
)

func TestTruncatedJSONObjectKeepsTheMembersCompletedBeforeTheCut(t *testing.T) {
	tests := []struct {
		arguments string
		members   string
	}{
		{``, `{}`},
		{" \n", `{}`},
		{`{`, `{}`},
		{`{"city"`, `{}`},
		{`{"city":`, `{}`},
		{`{"city": "Par`, `{}`},
		{`{"city": "Paris"`, `{"city": "Paris"}`},
		{`{"city": "Paris", `, `{"city": "Paris"}`},
		{`{"city": "Paris", "unit": "cel`, `{"city": "Paris"}`},
		{`{"days": 1`, `{}`},
		{`{"days": 12 `, `{"days": 12}`},
		{`{"days": [1, 2`, `{"days": [1]}`},
		{`{"days": [1, 2,`, `{"days": [1, 2]}`},
		{`{"days": [`, `{"days": []}`},
		{`{"metric": true`, `{"metric": true}`},
		{`{"metric": tr`, `{}`},
		{`{"metric": null, "x": fal`, `{"metric": null}`},
		{`{"where": {"city": "Paris", "zip": "75`, `{"where": {"city": "Paris"}}`},
		{`{"stops": [{"city": "Paris"}, {"city": "Ly`, `{"stops": [{"city": "Paris"}, {}]}`},
		{`{"note": "a \"quoted\" word", "more": "x\`, `{"note": "a \"quoted\" word"}`},
		{`{"note": "r\u00e9sum\u00e9", "more": "\u00`, `{"note": "r\u00e9sum\u00e9"}`},
		{`{"text": "naïve", "n": 3`, `{"text": "naïve"}`},
		{`  {"city": "Paris", "unit"`, `{"city": "Paris"}`},
	}
	for _, test := range tests {
		members, truncated := TruncatedJSONObject(test.arguments, 32)
		if !truncated || members != test.members {
			t.Errorf("TruncatedJSONObject(%q) = %q, %t; want %q, true", test.arguments, members, truncated, test.members)
		}
	}
}

func TestTruncatedJSONObjectRefusesWhatNoCutExplains(t *testing.T) {
	for _, arguments := range []string{
		`{"city": "Paris"}`,
		`{"city": "Paris"} `,
		`{"city": "Paris"}{`,
		`{"city": "Paris"} x`,
		`[`,
		`"Paris`,
		`city`,
		`{"city" "Paris"`,
		`{"city": "Paris",, "unit"`,
		`{"city": "Paris" "unit"`,
		`{"city": , "unit": "c`,
		`{,`,
		`{"days": [1 2`,
		`{"days": 01, "x": "a`,
		`{"days": 1.e`,
		`{"metric": trueish`,
		`{"metric": yes`,
		`{"city": "Paris", "city": "Lyon`,
		`{"note": "bad \q escape`,
		`{"note": "control ` + "\x01" + ` char`,
		`{"note": "\u12g4`,
		`{"where": {"a": 1]`,
		"{\"text\": \"\xff",
	} {
		if members, truncated := TruncatedJSONObject(arguments, 32); truncated {
			t.Errorf("TruncatedJSONObject(%q) = %q, true; want false", arguments, members)
		}
	}
}

func TestTruncatedJSONObjectAppliesTheDepthLimit(t *testing.T) {
	if _, truncated := TruncatedJSONObject(`{"a": {"b": {"c": "d`, 3); !truncated {
		t.Fatal("arguments within the depth limit were refused")
	}
	if _, truncated := TruncatedJSONObject(`{"a": {"b": {"c": {"d": "e`, 3); truncated {
		t.Fatal("arguments deeper than the limit were accepted")
	}
	deep := strings.Repeat(`{"a": `, 40)
	if _, truncated := TruncatedJSONObject(deep, 32); truncated {
		t.Fatal("nesting deeper than the limit was accepted before the cut")
	}
}

func TestTruncatedToolCallFindsOnlyTheCallTheOutputLimitCut(t *testing.T) {
	limits := DefaultPolicy().Limits
	for _, stop := range []StopReason{StopMaxTokens, StopContextWindow} {
		response := cutToolCallResponse(stop, `{"city": "Paris"}`, `{"city": "Ly`)
		if call := TruncatedToolCall(response, limits.JSONDepth); call == nil || call.ID != "call_b" {
			t.Errorf("stop %s: TruncatedToolCall = %+v, want call_b", stop, call)
		}
	}
	if call := TruncatedToolCall(cutToolCallResponse(StopMaxTokens, `{"city": "Paris"}`), limits.JSONDepth); call != nil {
		t.Errorf("a whole call was reported as cut: %+v", call)
	}
	for name, response := range cutArgumentsNoOutputLimitExplains() {
		if call := TruncatedToolCall(response, limits.JSONDepth); call != nil {
			t.Errorf("%s: TruncatedToolCall = %+v, want nil", name, call)
		}
	}
}
