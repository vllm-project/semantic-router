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
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sync"
	"testing"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// stageRecorder is a ToolStageGate that records the stage of every attempt it
// judges and leaves the request unchanged.
type stageRecorder struct {
	mu     sync.Mutex
	stages []llmprotocol.TrustedStage
}

func (s *stageRecorder) gate(stage llmprotocol.TrustedStage, req *openai.ChatCompletionNewParams) *openai.ChatCompletionNewParams {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.stages = append(s.stages, stage)
	return req
}

func twoToolRequest(t *testing.T) *openai.ChatCompletionNewParams {
	t.Helper()
	var req openai.ChatCompletionNewParams
	if err := json.Unmarshal([]byte(`{
		"model": "test",
		"messages": [{"role": "user", "content": "look it up"}],
		"tools": [
			{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}},
			{"type": "function", "function": {"name": "calculator", "parameters": {"type": "object"}}}
		],
		"tool_choice": {"type": "function", "function": {"name": "calculator"}}
	}`), &req); err != nil {
		t.Fatalf("decode request: %v", err)
	}
	return &req
}

func TestGateAttemptToolsSkipsAttemptsWithoutTools(t *testing.T) {
	rec := &stageRecorder{}
	options := CallOptions{
		ToolStage:        llmprotocol.TrustedStageCandidate,
		candidateRequest: &Request{ToolStageGate: rec.gate},
	}

	gateAttemptTools(options, *toolFreeLooperRequest(twoToolRequest(t)))
	if len(rec.stages) != 0 {
		t.Fatalf("tool-free attempt reached the gate: %v", rec.stages)
	}
	gateAttemptTools(options, *twoToolRequest(t))
	if want := []llmprotocol.TrustedStage{llmprotocol.TrustedStageCandidate}; !reflect.DeepEqual(rec.stages, want) {
		t.Fatalf("gated stages = %v, want %v", rec.stages, want)
	}
}

func TestNarrowToolUseKeepsPolicyToolsAndClearsForcedChoice(t *testing.T) {
	req := twoToolRequest(t)

	narrowed := NarrowToolUse(req, nil, []string{"calculator"})
	if len(narrowed.Tools) != 1 || narrowed.Tools[0].Function.Name != "search" {
		t.Fatalf("narrowed tools = %+v, want only search", narrowed.Tools)
	}
	data, err := json.Marshal(narrowed)
	if err != nil {
		t.Fatalf("encode request: %v", err)
	}
	var payload map[string]json.RawMessage
	if err := json.Unmarshal(data, &payload); err != nil {
		t.Fatalf("decode request: %v", err)
	}
	if _, ok := payload["tool_choice"]; ok {
		t.Fatalf("forced choice of a removed tool survived narrowing: %s", payload["tool_choice"])
	}
	if len(req.Tools) != 2 {
		t.Fatalf("narrowing mutated the original request: %d tools", len(req.Tools))
	}
	if kept := NarrowToolUse(req, []string{"search", "calculator"}, nil); kept != req {
		t.Fatal("narrowing that removes nothing must return the request unchanged")
	}
}

// Workflows step agents carry the client's tools but are not the final answer,
// so each call must reach the gate at its own stage: the worker as candidate,
// both before and after its tool result, and the synthesis as final.
func TestWorkflowsGatesEachToolCallAtItsOwnStage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			Model    string                   `json:"model"`
			Messages []map[string]interface{} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		switch {
		case payload.Model == "worker-model" && payloadHasToolMessage(payload.Messages):
			_, _ = w.Write(workflowChatCompletion("worker-model", "worker done"))
		case payload.Model == "worker-model":
			_, _ = w.Write(workflowToolCallCompletion("worker-model", "call_lookup"))
		default:
			_, _ = w.Write(workflowChatCompletion(payload.Model, "final answer"))
		}
	}))
	defer server.Close()

	rec := &stageRecorder{}
	looperCfg := workflowToolLooperConfig(server.URL, t.TempDir())
	first := workflowToolLooperRequest(workflowToolTestRequest())
	first.ToolStageGate = rec.gate
	firstResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), first)
	if err != nil {
		t.Fatalf("first Execute failed: %v", err)
	}
	assistantMessage, toolCallID := assistantToolMessageFromResponse(t, firstResp.Body)
	resume := workflowToolLooperRequest(workflowToolResumeRequest(t, assistantMessage, toolCallID))
	resume.ToolStageGate = rec.gate
	if _, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), resume); err != nil {
		t.Fatalf("resume Execute failed: %v", err)
	}

	want := []llmprotocol.TrustedStage{
		llmprotocol.TrustedStageCandidate,
		llmprotocol.TrustedStageCandidate,
		llmprotocol.TrustedStageFinal,
	}
	if !reflect.DeepEqual(rec.stages, want) {
		t.Fatalf("gated stages = %v, want %v", rec.stages, want)
	}
}
