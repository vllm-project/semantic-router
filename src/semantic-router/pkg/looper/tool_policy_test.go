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

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
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

func TestTaggedToolSemanticResponseUsesToolStopReason(t *testing.T) {
	response, ok := newTaggedToolSemanticResponse(
		"response-test",
		"model-a",
		`<tool_call>{"name":"search","arguments":{"query":"neutral"}}</tool_call>`,
		NewActualTokenUsage(4, 2, 6),
	)
	if !ok {
		t.Fatal("tagged tool call was not recognized")
	}
	if response.StopReason != llmprotocol.StopToolCall {
		t.Fatalf("stop reason = %q, want %q", response.StopReason, llmprotocol.StopToolCall)
	}
	if len(response.Output) != 1 || len(response.Output[0].Content) != 1 ||
		response.Output[0].Content[0].ToolCall == nil {
		t.Fatalf("neutral tool response = %+v", response)
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
