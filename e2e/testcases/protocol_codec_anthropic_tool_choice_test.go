package testcases

import "testing"

func TestAnthropicNoneToolChoiceAssertion(t *testing.T) {
	if err := assertAnthropicNoneToolChoice([]byte(`{"body":{"tool_choice":{"type":"none"}}}`)); err != nil {
		t.Fatal(err)
	}
	for _, invalid := range []string{
		`{"body":{"tool_choice":{"type":"none","disable_parallel_tool_use":true}}}`,
		`{"body":{"tool_choice":{"type":"none","disable_parallel_tool_use":false}}}`,
		`{"body":{"tool_choice":{"type":"none","name":"lookup"}}}`,
		`{"body":{"tool_choice":{"type":"auto"}}}`,
		`{"body":{"tool_choice":{"type":"none"},"parallel_tool_calls":false}}`,
		`{"body":{"tool_choice":null}}`,
		`{"body":{}}`,
		`{`,
	} {
		if err := assertAnthropicNoneToolChoice([]byte(invalid)); err == nil {
			t.Errorf("accepted invalid provider request: %s", invalid)
		}
	}
}
