/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package looper

import (
	"encoding/json"
	"testing"

	"github.com/openai/openai-go"
)

func TestToolFreeLooperRequestRemovesCallableSchemasWithoutMutatingOriginal(t *testing.T) {
	var req openai.ChatCompletionNewParams
	if err := json.Unmarshal([]byte(`{
		"model": "test",
		"messages": [{"role": "user", "content": "compare answers"}],
		"tools": [{
			"type": "function",
			"function": {
				"name": "search",
				"description": "Search",
				"parameters": {"type": "object"}
			}
		}],
		"tool_choice": "auto"
	}`), &req); err != nil {
		t.Fatalf("decode request: %v", err)
	}

	stripped := toolFreeLooperRequest(&req)
	assertRequestToolFields(t, &req, true)
	assertRequestToolFields(t, stripped, false)
}

func TestNormalizeCompletionToolFinishReasonRepairsBackendStop(t *testing.T) {
	completion := map[string]interface{}{
		"choices": []interface{}{
			map[string]interface{}{
				"message": map[string]interface{}{
					"tool_calls": []interface{}{map[string]interface{}{"id": "call-1"}},
				},
				"finish_reason": "stop",
			},
		},
	}

	normalizeCompletionToolFinishReason(completion)

	choice := completion["choices"].([]interface{})[0].(map[string]interface{})
	if choice["finish_reason"] != "tool_calls" {
		t.Fatalf("finish_reason = %v, want tool_calls", choice["finish_reason"])
	}
}

func assertRequestToolFields(
	t *testing.T,
	req *openai.ChatCompletionNewParams,
	wantTools bool,
) {
	t.Helper()
	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("encode request: %v", err)
	}
	var payload map[string]interface{}
	if err := json.Unmarshal(data, &payload); err != nil {
		t.Fatalf("decode request map: %v", err)
	}
	if wantTools {
		for _, field := range []string{"tools", "tool_choice"} {
			if _, exists := payload[field]; !exists {
				t.Fatalf("field %q missing from original request: %s", field, data)
			}
		}
		return
	}
	for _, field := range []string{"tools", "tool_choice", "functions", "function_call"} {
		if _, exists := payload[field]; exists {
			t.Fatalf("field %q leaked into tool-free request: %s", field, data)
		}
	}
}
