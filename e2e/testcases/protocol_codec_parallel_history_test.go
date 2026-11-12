package testcases

import "testing"

func TestParallelHistoryProviderAssertion(t *testing.T) {
	const callA = `{"id":"call_a","type":"function","function":{"name":"lookup","arguments":"{\"query\":\"Paris\"}"}}`
	const callB = `{"id":"call_b","type":"function","function":{"name":"lookup","arguments":"{\"query\":\"London\"}"}}`
	const prefix = `{"body":{"messages":[{"role":"user","content":"Summarize"},`
	const results = `,{"role":"tool","tool_call_id":"call_a","content":"sunny"},{"role":"tool","tool_call_id":"call_b","content":"rainy"}]}}`
	grouped := prefix + `{"role":"assistant","tool_calls":[` + callA + `,` + callB + `]}` + results
	if err := assertParallelHistoryProviderRequest([]byte(grouped)); err != nil {
		t.Fatal(err)
	}
	split := prefix + `{"role":"assistant","tool_calls":[` + callA + `]},{"role":"assistant","tool_calls":[` + callB + `]}` + results
	if err := assertParallelHistoryProviderRequest([]byte(split)); err == nil {
		t.Fatal("accepted the original split-assistant regression")
	}
}
