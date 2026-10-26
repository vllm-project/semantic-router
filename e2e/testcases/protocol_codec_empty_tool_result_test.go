package testcases

import "testing"

func TestChatToolResultContentAssertion(t *testing.T) {
	for _, test := range []struct {
		name, body, want string
		valid            bool
	}{
		{"empty", `{"body":{"messages":[{"role":"tool","tool_call_id":"call_a","content":""}]}}`, `""`, true},
		{"native empty array", `{"body":{"messages":[{"role":"tool","tool_call_id":"call_a","content":[]}]}}`, `[]`, true},
		{"reencoded array not normalized", `{"body":{"messages":[{"role":"tool","tool_call_id":"call_a","content":[]}]}}`, `""`, false},
		{"native missing content", `{"body":{"messages":[{"role":"tool","tool_call_id":"call_a"}]}}`, `[]`, false},
		{"native null", `{"body":{"messages":[{"role":"tool","tool_call_id":"call_a","content":null}]}}`, `[]`, false},
		{"text", `{"body":{"messages":[{"role":"tool","tool_call_id":"call_a","content":"result"}]}}`, `"result"`, true},
		{"missing content", `{"body":{"messages":[{"role":"tool","tool_call_id":"call_a"}]}}`, `""`, false},
		{"null", `{"body":{"messages":[{"role":"tool","tool_call_id":"call_a","content":null}]}}`, `""`, false},
		{"wrong ID", `{"body":{"messages":[{"role":"tool","tool_call_id":"other","content":""}]}}`, `""`, false},
		{"dropped result", `{"body":{"messages":[]}}`, `""`, false},
		{"invalid JSON", `{`, `""`, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			if err := assertChatToolResultContent([]byte(test.body), test.want); (err == nil) != test.valid {
				t.Fatalf("valid=%t error=%v", test.valid, err)
			}
		})
	}
}
