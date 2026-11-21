package llmprotocol

import (
	"slices"
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

const cutHistoryArguments = `{"to": "team@example.com", "body": "Dear team, the review moves to ten.`

func historyRequest(messages ...Message) Request {
	user := Message{Role: RoleUser, Content: []Content{{Kind: ContentText, Text: "Tell the team the review moves to ten."}}}
	return Request{Generation: 1, Model: "m", Messages: append([]Message{user}, messages...)}
}

func assistantTurn(contents ...Content) Message {
	return Message{Role: RoleAssistant, Content: contents}
}

func historyCall(id, arguments string) Content {
	return Content{Kind: ContentToolCall, ToolCall: &ToolCall{ID: id, Name: "send_email", Arguments: arguments}}
}

func historyResult(id string) Message {
	return Message{Role: RoleTool, Content: []Content{{Kind: ContentToolResult, ToolResult: &ToolResult{
		CallID: id, Content: []Content{{Kind: ContentText, Text: "sent"}},
	}}}}
}

func historyCallArguments(request Request) []string {
	var arguments []string
	for _, message := range request.Messages {
		for _, content := range message.Content {
			if content.ToolCall != nil {
				arguments = append(arguments, content.ToolCall.Arguments)
			}
		}
	}
	return arguments
}

// A client sends back the call an output limit cut, as the Router passed it
// on, at the end of its assistant turn: the last call of a Chat assistant
// message, or the last function_call item of a Responses turn. The call gets
// the members the model completed before the cut, and the request is valid.
func TestCloseTruncatedHistoryCallsClosesTheCallThatEndsItsTurn(t *testing.T) {
	text := Content{Kind: ContentText, Text: "Writing it now."}
	for name, test := range map[string]struct {
		request Request
		want    []string
	}{
		"the call of a turn": {
			historyRequest(assistantTurn(text, historyCall("call_1", cutHistoryArguments)), historyResult("call_1")),
			[]string{`{"to": "team@example.com"}`},
		},
		"the last of parallel calls": {
			historyRequest(assistantTurn(historyCall("call_1", `{"to":"a@example.com"}`), historyCall("call_2", cutHistoryArguments)),
				historyResult("call_1"), historyResult("call_2")),
			[]string{`{"to":"a@example.com"}`, `{"to": "team@example.com"}`},
		},
		"the last call item of a turn": {
			historyRequest(assistantTurn(historyCall("call_1", `{"to":"a@example.com"}`)), assistantTurn(historyCall("call_2", cutHistoryArguments)),
				historyResult("call_1"), historyResult("call_2")),
			[]string{`{"to":"a@example.com"}`, `{"to": "team@example.com"}`},
		},
		"cut right after its name": {
			historyRequest(assistantTurn(historyCall("call_1", ``)), historyResult("call_1")),
			[]string{`{}`},
		},
		"at the end of the history": {
			historyRequest(assistantTurn(historyCall("call_1", cutHistoryArguments))),
			[]string{`{"to": "team@example.com"}`},
		},
	} {
		t.Run(name, func(t *testing.T) {
			request := test.request
			if closed := CloseTruncatedHistoryCalls(&request, DefaultPolicy().Limits.JSONDepth); closed != 1 {
				t.Fatalf("closed %d calls, want 1", closed)
			}
			if got := historyCallArguments(request); !slices.Equal(got, test.want) {
				t.Fatalf("call arguments = %q, want %q", got, test.want)
			}
			if err := ValidateRequest(request, DefaultPolicy().Limits); err != nil {
				t.Fatalf("the closed request is invalid: %v", err)
			}
		})
	}
}

// Malformed arguments that no output limit explains stay as they are, so the
// request is still refused: a cut call that more of its turn follows, and
// arguments that don't start an object or carry an error before the cut.
func TestCloseTruncatedHistoryCallsLeavesWhatNoCutExplains(t *testing.T) {
	text := Content{Kind: ContentText, Text: "Done."}
	for name, request := range map[string]Request{
		"followed by another call": historyRequest(
			assistantTurn(historyCall("call_1", cutHistoryArguments), historyCall("call_2", `{}`)),
			historyResult("call_1"), historyResult("call_2")),
		"followed by another call item": historyRequest(
			assistantTurn(historyCall("call_1", cutHistoryArguments)), assistantTurn(historyCall("call_2", `{}`)),
			historyResult("call_1"), historyResult("call_2")),
		"followed by text in its turn": historyRequest(
			assistantTurn(historyCall("call_1", cutHistoryArguments)), assistantTurn(text), historyResult("call_1")),
		"not the start of an object": historyRequest(assistantTurn(historyCall("call_1", `["team`)), historyResult("call_1")),
		"an error before the cut":    historyRequest(assistantTurn(historyCall("call_1", `{"to" "team`)), historyResult("call_1")),
	} {
		t.Run(name, func(t *testing.T) {
			before := historyCallArguments(request)
			if closed := CloseTruncatedHistoryCalls(&request, DefaultPolicy().Limits.JSONDepth); closed != 0 {
				t.Fatalf("closed %d calls, want none", closed)
			}
			if got := historyCallArguments(request); !slices.Equal(got, before) {
				t.Fatalf("call arguments = %q, want them unchanged %q", got, before)
			}
			requireLLMProtocolErrorCode(t, ValidateRequest(request, DefaultPolicy().Limits), "invalid_tool_call")
		})
	}
	custom := historyRequest(assistantTurn(Content{Kind: ContentToolCall, ToolCall: &ToolCall{
		Kind: ToolKindCustom, ID: "call_1", Name: "patch", Arguments: `{"free": "form`,
	}}), historyResult("call_1"))
	if closed := CloseTruncatedHistoryCalls(&custom, DefaultPolicy().Limits.JSONDepth); closed != 0 {
		t.Fatalf("closed %d custom calls, whose input is free-form", closed)
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
