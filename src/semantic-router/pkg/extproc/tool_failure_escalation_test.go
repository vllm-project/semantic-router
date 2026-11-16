package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

const toolFailureEscalationConfig = `
version: v0.5
providers:
  defaults:
    model: worker
  models:
    - name: worker
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
    - name: premium
      backend_refs:
        - endpoint: 127.0.0.1:8001
          provider: vllm
recipes:
  - name: default
    signals:
      conversation:
        - name: tool_failures
          feature:
            type: count
            source:
              type: failed_tool_result
          predicate:
            gte: 2
    decisions:
      - name: escalate
        priority: 200
        rules:
          type: conversation
          name: tool_failures
        modelRefs:
          - model: premium
      - name: work
        priority: 1
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: worker
`

// anthropicToolLoop is a Claude Code tool loop whose results fail where
// failed is true, the way Claude Code reports a command that exited nonzero.
// tail is appended as content blocks of the last tool-result message.
func anthropicToolLoop(t *testing.T, failed []bool, tail ...map[string]any) []byte {
	t.Helper()
	messages := []map[string]any{{"role": "user", "content": "Fix the failing build."}}
	for index, fail := range failed {
		id := fmt.Sprintf("call_%d", index)
		result := map[string]any{"type": "tool_result", "tool_use_id": id, "content": "ok"}
		if fail {
			result["is_error"], result["content"] = true, "Exit code 2\nmake: *** [all] Error 2"
		}
		content := []map[string]any{result}
		if index == len(failed)-1 {
			content = append(content, tail...)
		}
		messages = append(messages,
			map[string]any{"role": "assistant", "content": []map[string]any{{"type": "tool_use", "id": id, "name": "Bash", "input": map[string]any{"command": "make"}}}},
			map[string]any{"role": "user", "content": content})
	}
	return mustToolLoopJSON(t, map[string]any{"model": "vllm-sr/auto", "max_tokens": 64, "messages": messages})
}

// codexToolLoop is a Codex exec_command loop with one result per exit code.
func codexToolLoop(t *testing.T, outputs ...string) []byte {
	t.Helper()
	input := []map[string]any{{"type": "message", "role": "user", "content": []map[string]any{{"type": "input_text", "text": "Fix the failing build."}}}}
	for index, output := range outputs {
		id := fmt.Sprintf("call_%d", index)
		input = append(input,
			map[string]any{"type": "function_call", "call_id": id, "name": "exec_command", "arguments": `{"cmd":"make"}`},
			map[string]any{"type": "function_call_output", "call_id": id, "output": output})
	}
	return mustToolLoopJSON(t, map[string]any{"model": "vllm-sr/auto", "input": input})
}

func codexExecOutput(code int, output string) string {
	return fmt.Sprintf("Chunk ID: d9a39a\nWall time: 0.1000 seconds\nProcess exited with code %d\nOriginal token count: 4\nOutput:\n%s\n", code, output)
}

func mustToolLoopJSON(t *testing.T, value any) []byte {
	t.Helper()
	body, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return body
}

// Repeated tool failures route to the stronger decision, and the route
// returns only after two consecutive successes or a new user turn, so one
// success does not switch the session back and forth.
func TestFailedToolResultsEscalateUntilTwoSuccesses(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(toolFailureEscalationConfig))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
	classifier, err := classification.NewClassifier(cfg)
	if err != nil {
		t.Fatalf("NewClassifier() error = %v", err)
	}
	router := &OpenAIRouter{Config: cfg, Classifier: classifier}
	reminder := map[string]any{"type": "text", "text": "<system-reminder>Keep going.</system-reminder>"}
	newTurn := mustToolLoopJSON(t, map[string]any{"model": "vllm-sr/auto", "max_tokens": 64, "messages": append(
		decodeToolLoopMessages(t, anthropicToolLoop(t, []bool{true, true})),
		map[string]any{"role": "assistant", "content": "The build still fails."},
		map[string]any{"role": "user", "content": "Try another approach."})})
	for _, tc := range []struct {
		name   string
		format llmprotocol.WireFormat
		body   []byte
		want   string
	}{
		{"two failed results escalate", llmprotocol.AnthropicMessagesV1, anthropicToolLoop(t, []bool{true, true}), "premium"},
		{"successful results stay", llmprotocol.AnthropicMessagesV1, anthropicToolLoop(t, []bool{false, false}), "worker"},
		{"one failure stays", llmprotocol.AnthropicMessagesV1, anthropicToolLoop(t, []bool{false, true}), "worker"},
		{"one success keeps the escalation", llmprotocol.AnthropicMessagesV1, anthropicToolLoop(t, []bool{true, true, false}), "premium"},
		{"a failure between successes accumulates", llmprotocol.AnthropicMessagesV1, anthropicToolLoop(t, []bool{true, false, true}), "premium"},
		{"two successes return", llmprotocol.AnthropicMessagesV1, anthropicToolLoop(t, []bool{true, true, false, false}), "worker"},
		{"text next to tool results continues the loop", llmprotocol.AnthropicMessagesV1, anthropicToolLoop(t, []bool{true, true}, reminder), "premium"},
		{"a new user turn returns", llmprotocol.AnthropicMessagesV1, newTurn, "worker"},
		{"codex nonzero exits escalate", llmprotocol.OpenAIResponsesV1, codexToolLoop(t, codexExecOutput(2, "Error 2"), codexExecOutput(1, "Error 1")), "premium"},
		{"codex output text is not a status", llmprotocol.OpenAIResponsesV1, codexToolLoop(t, codexExecOutput(0, "Exit code: 1"), codexExecOutput(0, "Process exited with code 1")), "worker"},
		{"chat results carry no failure flag", llmprotocol.OpenAIChatV1, mustToolLoopJSON(t, map[string]any{"model": "vllm-sr/auto", "messages": []map[string]any{
			{"role": "user", "content": "Fix the failing build."},
			{"role": "assistant", "tool_calls": []map[string]any{{"id": "c0", "type": "function", "function": map[string]any{"name": "make", "arguments": "{}"}}}},
			{"role": "tool", "tool_call_id": "c0", "content": "Error: exit code 2"},
			{"role": "assistant", "tool_calls": []map[string]any{{"id": "c1", "type": "function", "function": map[string]any{"name": "make", "arguments": "{}"}}}},
			{"role": "tool", "tool_call_id": "c1", "content": "Error: exit code 2"},
		}}), "worker"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			request, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(tc.format, tc.body)
			if err != nil {
				t.Fatal(err)
			}
			ctx := &RequestContext{TraceContext: context.Background(), Headers: map[string]string{}}
			_, _, _, model, err := router.performDecisionEvaluation("vllm-sr/auto", extractSignalConversationHistory(&request), ctx)
			if err != nil {
				t.Fatalf("performDecisionEvaluation() error = %v", err)
			}
			if model != tc.want {
				t.Fatalf("model = %q, want %q (signal values %v)", model, tc.want, ctx.VSRSignalValues)
			}
		})
	}
}

func decodeToolLoopMessages(t *testing.T, body []byte) []any {
	t.Helper()
	var request struct {
		Messages []any `json:"messages"`
	}
	if err := json.Unmarshal(body, &request); err != nil {
		t.Fatal(err)
	}
	return request.Messages
}
