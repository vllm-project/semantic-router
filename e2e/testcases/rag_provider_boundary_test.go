package testcases

import (
	"encoding/json"
	"testing"
)

func TestRAGProviderBoundaryContract(t *testing.T) {
	for _, testCase := range []struct {
		name      string
		context   string
		toolName  string
		wantError bool
	}{
		{name: "retrieved context", context: ragProviderBoundaryContext, toolName: "vsr_rag_context"},
		{name: "missing context", toolName: "vsr_rag_context", wantError: true},
		{name: "wrong tool", context: ragProviderBoundaryContext, toolName: "search", wantError: true},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			observed := ragProviderBoundaryObservedRequest(t, testCase.context, testCase.toolName)
			err := validateRAGProviderBoundaryRequest(observed)
			if (err != nil) != testCase.wantError {
				t.Fatalf("validation error = %v, wantError = %t", err, testCase.wantError)
			}
		})
	}
}

func ragProviderBoundaryObservedRequest(t *testing.T, context, toolName string) []byte {
	t.Helper()
	body, err := json.Marshal(map[string]any{
		"body": map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": ragProviderBoundaryMarker +
						" What is Project Zephyr's signing-key rotation interval?",
				},
				map[string]any{
					"role": "assistant",
					"tool_calls": []any{map[string]any{
						"id":   "rag_request_1",
						"type": "function",
						"function": map[string]any{
							"name":      toolName,
							"arguments": "{}",
						},
					}},
				},
				map[string]any{
					"role":         "tool",
					"tool_call_id": "rag_request_1",
					"content":      context,
				},
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	return body
}
