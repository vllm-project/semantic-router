package llmprotocol

import "testing"

func cutToolCallResponse(stop StopReason, calls ...string) Response {
	contents := make([]Content, 0, len(calls))
	for index, arguments := range calls {
		contents = append(contents, Content{Kind: ContentToolCall, ToolCall: &ToolCall{
			ID: "call_" + string(rune('a'+index)), Name: "lookup", Arguments: arguments,
		}})
	}
	return Response{
		Generation: 1, ID: "response", StopReason: stop,
		Output: []OutputItem{{ID: "item", Role: RoleAssistant, Content: contents}},
		Usage:  Usage{State: UsageUnavailable},
	}
}

// The output limit can cut a reply inside its last call, whose arguments then
// start a JSON object: nothing yet, or an object that stops before it closes.
func TestValidateResponseAcceptsTheCallTheOutputLimitCut(t *testing.T) {
	limits := DefaultPolicy().Limits
	for _, stop := range []StopReason{StopMaxTokens, StopContextWindow} {
		for _, arguments := range []string{``, `{"city": "Par`, `{"city": "Paris", "days": [1`} {
			response := cutToolCallResponse(stop, `{"city": "Paris"}`, arguments)
			if err := ValidateResponse(response, limits); err != nil {
				t.Errorf("stop %s, arguments %q: %v", stop, arguments, err)
			}
		}
	}
}

func TestValidateResponseRefusesCutArgumentsNoOutputLimitExplains(t *testing.T) {
	limits := DefaultPolicy().Limits
	for name, response := range cutArgumentsNoOutputLimitExplains() {
		t.Run(name, func(t *testing.T) {
			requireLLMProtocolErrorCode(t, ValidateResponse(response, limits), "invalid_tool_call")
		})
	}
}

func cutArgumentsNoOutputLimitExplains() map[string]Response {
	return map[string]Response{
		"stopped for the call":          cutToolCallResponse(StopToolCall, `{"city": "Par`),
		"ended its turn":                cutToolCallResponse(StopEndTurn, `{"city": "Par`),
		"followed by another call":      cutToolCallResponse(StopMaxTokens, `{"city": "Par`, `{"city": "Lyon"}`),
		"not the start of an object":    cutToolCallResponse(StopMaxTokens, `["Paris"`),
		"an error before the cut":       cutToolCallResponse(StopMaxTokens, `{"city" "Par`),
		"a complete non-object":         cutToolCallResponse(StopMaxTokens, `[]`),
		"data after a complete object":  cutToolCallResponse(StopMaxTokens, `{"city": "Paris"} {`),
		"followed by text in its item":  withTrailingText(cutToolCallResponse(StopMaxTokens, `{"city": "Par`)),
		"in an alternative":             withAlternative(cutToolCallResponse(StopMaxTokens, `{"city": "Paris"}`), `{"city": "Par`),
		"followed by another item":      withTrailingItem(cutToolCallResponse(StopMaxTokens, `{"city": "Par`)),
		"duplicate keys before the cut": cutToolCallResponse(StopMaxTokens, `{"city": "Paris", "city": "Ly`),
	}
}

func withTrailingText(response Response) Response {
	item := &response.Output[0]
	item.Content = append(item.Content, Content{Kind: ContentText, Text: "and then"})
	return response
}

func withTrailingItem(response Response) Response {
	response.Output = append(response.Output, OutputItem{
		ID: "item_2", Role: RoleAssistant, Content: []Content{{Kind: ContentText, Text: "and then"}},
	})
	return response
}

func withAlternative(response Response, arguments string) Response {
	alternative := cutToolCallResponse(response.StopReason, arguments).Output
	alternative[0].ID = "alternative"
	response.Alternatives = [][]OutputItem{alternative}
	return response
}
