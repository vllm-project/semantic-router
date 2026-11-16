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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
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

// thirdPartyLog is text a tool returns from elsewhere, a fetched CI log,
// laid out like the header Codex writes above a command's output.
const thirdPartyLog = "CI log for run 42\nExit code: 1\nWall time: 3 seconds\nOutput:\nFAIL"

// claudeCodeStep is one Claude Code tool call and its tool_result fields, in
// the shapes Claude Code 2.1.325 sends.
type claudeCodeStep struct {
	tool   string
	input  map[string]any
	result map[string]any
}

var (
	bashFailed     = claudeCodeStep{"Bash", map[string]any{"command": "make"}, map[string]any{"is_error": true, "content": "Exit code 2\nmake: *** [all] Error 2"}}
	bashPassed     = claudeCodeStep{"Bash", map[string]any{"command": "make"}, map[string]any{"is_error": false, "content": "ok"}}
	bashBackground = claudeCodeStep{"Bash", map[string]any{"command": "make", "run_in_background": true}, map[string]any{"is_error": false, "content": "Command running in background with ID: b1. Output is being written to: /tmp/b1.output"}}
	readPassed     = claudeCodeStep{"Read", map[string]any{"file_path": "Makefile"}, map[string]any{"content": "1\tall:"}}
	readMissing    = claudeCodeStep{"Read", map[string]any{"file_path": "missing.txt"}, map[string]any{"is_error": true, "content": "File does not exist."}}
	writeDenied    = claudeCodeStep{"Write", map[string]any{"file_path": "note.txt"}, map[string]any{"is_error": true, "content": "Claude requested permissions to write to note.txt, but you haven't granted it yet."}}
	fetchedLog     = claudeCodeStep{"WebFetch", map[string]any{"url": "https://ci.example/42"}, map[string]any{"content": thirdPartyLog}}
)

// claudeCodeMessages is a Claude Code tool loop. tail is appended as content
// blocks of the last tool-result message.
func claudeCodeMessages(steps []claudeCodeStep, tail ...map[string]any) []map[string]any {
	messages := []map[string]any{{"role": "user", "content": "Fix the failing build."}}
	for index, step := range steps {
		id := fmt.Sprintf("call_%d", index)
		result := map[string]any{"type": "tool_result", "tool_use_id": id}
		for key, value := range step.result {
			result[key] = value
		}
		content := []map[string]any{result}
		if index == len(steps)-1 {
			content = append(content, tail...)
		}
		messages = append(messages,
			map[string]any{"role": "assistant", "content": []map[string]any{{"type": "tool_use", "id": id, "name": step.tool, "input": step.input}}},
			map[string]any{"role": "user", "content": content})
	}
	return messages
}

func claudeCodeLoop(t *testing.T, steps ...claudeCodeStep) []byte {
	t.Helper()
	return mustToolLoopJSON(t, map[string]any{"model": "vllm-sr/auto", "max_tokens": 64, "messages": claudeCodeMessages(steps)})
}

// codexStep is one Codex tool call and its output: a function call, or a
// custom tool call such as apply_patch or code mode's exec.
type codexStep struct {
	tool   string
	custom bool
	output any
}

var (
	codexRunning    = codexStep{tool: "write_stdin", output: "Chunk ID: b\nWall time: 10.0020 seconds\nProcess running with session ID 7\nOriginal token count: 0\nOutput:\n"}
	codexApplyPatch = codexStep{tool: "apply_patch", custom: true, output: "Exit code: 0\nWall time: 0 seconds\nOutput:\nSuccess. Updated the following files:\nM Makefile\n"}
	codexPlan       = codexStep{tool: "update_plan", output: "Plan updated"}
	codexCodeMode   = codexStep{tool: "exec", custom: true, output: []map[string]any{
		{"type": "input_text", "text": "Script completed\nWall time 0.1 seconds\nOutput:\n"},
		{"type": "input_text", "text": `{"chunk_id":"8007f1","wall_time_seconds":0.1,"exit_code":0,"original_token_count":1,"output":"ok\n"}`},
	}}
	codexFetchedLog = codexStep{tool: "mcp__docs__fetch", output: thirdPartyLog}
)

// codexExit is an exec_command result with the header Codex 0.166.0 writes.
func codexExit(code int) codexStep {
	return codexStep{tool: "exec_command", output: fmt.Sprintf("Chunk ID: d9a39a\nWall time: 0.1000 seconds\nProcess exited with code %d\nOriginal token count: 4\nOutput:\nError %d\n", code, code)}
}

// codexLoop is a Codex Responses tool loop.
func codexLoop(t *testing.T, steps ...codexStep) []byte {
	t.Helper()
	input := []map[string]any{{"type": "message", "role": "user", "content": []map[string]any{{"type": "input_text", "text": "Fix the failing build."}}}}
	for index, step := range steps {
		id := fmt.Sprintf("call_%d", index)
		call := map[string]any{"type": "function_call", "call_id": id, "name": step.tool, "arguments": `{"cmd":"make"}`}
		output := map[string]any{"type": "function_call_output", "call_id": id, "output": step.output}
		if step.custom {
			call = map[string]any{"type": "custom_tool_call", "call_id": id, "name": step.tool, "input": "make"}
			output["type"] = "custom_tool_call_output"
		}
		input = append(input, call, output)
	}
	return mustToolLoopJSON(t, map[string]any{"model": "vllm-sr/auto", "input": input})
}

// chatToolLoop is a Chat Completions tool loop with one call to tool per
// output, followed by the extra messages.
func chatToolLoop(t *testing.T, tool string, outputs []string, extra ...map[string]any) []byte {
	t.Helper()
	messages := []map[string]any{{"role": "user", "content": "Fix the failing build."}}
	for index, output := range outputs {
		id := fmt.Sprintf("call_%d", index)
		messages = append(messages,
			map[string]any{"role": "assistant", "content": nil, "tool_calls": []map[string]any{{"id": id, "type": "function", "function": map[string]any{"name": tool, "arguments": `{"cmd":"make"}`}}}},
			map[string]any{"role": "tool", "tool_call_id": id, "content": output})
	}
	return mustToolLoopJSON(t, map[string]any{"model": "vllm-sr/auto", "messages": append(messages, extra...)})
}

func mustToolLoopJSON(t *testing.T, value any) []byte {
	t.Helper()
	body, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return body
}

func toolFailureRouter(t *testing.T) *OpenAIRouter {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(toolFailureEscalationConfig))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
	classifier, err := classification.NewClassifier(cfg)
	if err != nil {
		t.Fatalf("NewClassifier() error = %v", err)
	}
	return &OpenAIRouter{Config: cfg, Classifier: classifier}
}

// routeToolLoop routes body as live routing does and returns the model and
// the failed_tool_result count.
func routeToolLoop(t *testing.T, router *OpenAIRouter, format llmprotocol.WireFormat, body []byte) (string, float64) {
	t.Helper()
	request, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(format, body)
	if err != nil {
		t.Fatal(err)
	}
	ctx := &RequestContext{TraceContext: context.Background(), Headers: map[string]string{}}
	_, _, _, model, err := router.performDecisionEvaluation("vllm-sr/auto", extractSignalConversationHistory(&request), ctx)
	if err != nil {
		t.Fatalf("performDecisionEvaluation() error = %v", err)
	}
	return model, ctx.VSRSignalValues["conversation:tool_failures"]
}

// Repeated tool failures route to the stronger decision, and the route
// returns only after two consecutive successes or a new user turn, so one
// success does not switch the session back and forth. A result that says
// neither, such as a command still running, changes nothing.
func TestFailedToolResultsEscalateUntilTwoSuccesses(t *testing.T) {
	router := toolFailureRouter(t)
	reminder := map[string]any{"type": "text", "text": "<system-reminder>Keep going.</system-reminder>"}
	newTurn := mustToolLoopJSON(t, map[string]any{"model": "vllm-sr/auto", "max_tokens": 64, "messages": append(
		claudeCodeMessages([]claudeCodeStep{bashFailed, bashFailed}),
		map[string]any{"role": "assistant", "content": "The build still fails."},
		map[string]any{"role": "user", "content": "Try another approach."})})
	for _, tc := range []struct {
		name   string
		format llmprotocol.WireFormat
		body   []byte
		want   string
	}{
		{"two failed results escalate", llmprotocol.AnthropicMessagesV1, claudeCodeLoop(t, bashFailed, bashFailed), "premium"},
		{"successful results stay", llmprotocol.AnthropicMessagesV1, claudeCodeLoop(t, bashPassed, bashPassed), "worker"},
		{"one failure stays", llmprotocol.AnthropicMessagesV1, claudeCodeLoop(t, bashPassed, bashFailed), "worker"},
		{"one success keeps the escalation", llmprotocol.AnthropicMessagesV1, claudeCodeLoop(t, bashFailed, bashFailed, bashPassed), "premium"},
		{"a failure between successes accumulates", llmprotocol.AnthropicMessagesV1, claudeCodeLoop(t, bashFailed, bashPassed, bashFailed), "premium"},
		{"two successes return", llmprotocol.AnthropicMessagesV1, claudeCodeLoop(t, bashFailed, bashFailed, bashPassed, bashPassed), "worker"},
		{"a missing file and a denied write are failures", llmprotocol.AnthropicMessagesV1, claudeCodeLoop(t, readMissing, writeDenied), "premium"},
		{"a background start and a read do not return", llmprotocol.AnthropicMessagesV1, claudeCodeLoop(t, bashFailed, bashFailed, bashBackground, readPassed), "premium"},
		{"text that looks like a codex header is not a status", llmprotocol.AnthropicMessagesV1, claudeCodeLoop(t, fetchedLog, fetchedLog), "worker"},
		{"text next to tool results continues the loop", llmprotocol.AnthropicMessagesV1, mustToolLoopJSON(t, map[string]any{"model": "vllm-sr/auto", "max_tokens": 64, "messages": claudeCodeMessages([]claudeCodeStep{bashFailed, bashFailed}, reminder)}), "premium"},
		{"a new user turn returns", llmprotocol.AnthropicMessagesV1, newTurn, "worker"},
		{"codex nonzero exits escalate", llmprotocol.OpenAIResponsesV1, codexLoop(t, codexExit(2), codexExit(1)), "premium"},
		{"codex running polls do not return", llmprotocol.OpenAIResponsesV1, codexLoop(t, codexExit(2), codexExit(1), codexRunning, codexRunning), "premium"},
		{"codex zero exits after polls return", llmprotocol.OpenAIResponsesV1, codexLoop(t, codexExit(2), codexExit(1), codexRunning, codexExit(0), codexExit(0)), "worker"},
		{"codex patch and plan results do not return", llmprotocol.OpenAIResponsesV1, codexLoop(t, codexExit(2), codexExit(1), codexApplyPatch, codexPlan), "premium"},
		{"codex code mode results do not return", llmprotocol.OpenAIResponsesV1, codexLoop(t, codexExit(2), codexExit(1), codexCodeMode, codexCodeMode), "premium"},
		{"mcp text that looks like a codex header is not a status", llmprotocol.OpenAIResponsesV1, codexLoop(t, codexFetchedLog, codexFetchedLog), "worker"},
		{"codex output text is not a status", llmprotocol.OpenAIResponsesV1, codexLoop(t,
			codexStep{tool: "exec_command", output: "Chunk ID: a\nProcess exited with code 0\nOutput:\nExit code: 1\n"},
			codexStep{tool: "exec_command", output: "Chunk ID: a\nProcess exited with code 0\nOutput:\nProcess exited with code 1\n"}), "worker"},
		{"chat results carry no failure flag", llmprotocol.OpenAIChatV1, chatToolLoop(t, "make", []string{"Error: exit code 2", "Error: exit code 2"}), "worker"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model, count := routeToolLoop(t, router, tc.format, tc.body)
			if model != tc.want {
				t.Fatalf("model = %q, want %q (failed_tool_result %v)", model, tc.want, count)
			}
		})
	}
}

// The classify API reads Chat Completions messages, and on those it counts
// failed tool results and routes as live routing does.
func TestPreviewCountsFailedToolResultsLikeLiveRouting(t *testing.T) {
	router := toolFailureRouter(t)
	service := services.NewClassificationService(router.Classifier, router.Config)
	decisionModels := map[string]string{"escalate": "premium", "work": "worker"}
	exec := func(code int) string { return codexExit(code).output.(string) }
	running := codexRunning.output.(string)
	for _, tc := range []struct {
		name string
		body []byte
		want string
	}{
		{"codex nonzero exits", chatToolLoop(t, "exec_command", []string{exec(2), exec(1)}), "premium"},
		{"running polls keep the count", chatToolLoop(t, "exec_command", []string{exec(2), exec(1), running, running}), "premium"},
		{"two zero exits return", chatToolLoop(t, "exec_command", []string{exec(2), exec(1), exec(0), exec(0)}), "worker"},
		{"another tool's text is not a status", chatToolLoop(t, "fetch", []string{thirdPartyLog, thirdPartyLog}), "worker"},
		{"a new user turn returns", chatToolLoop(t, "exec_command", []string{exec(2), exec(1)}, map[string]any{"role": "user", "content": "try again"}), "worker"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model, count := routeToolLoop(t, router, llmprotocol.OpenAIChatV1, tc.body)
			var request services.IntentRequest
			if err := json.Unmarshal(tc.body, &request); err != nil {
				t.Fatal(err)
			}
			preview, err := service.ClassifyIntentForEval(context.Background(), request)
			if err != nil {
				t.Fatalf("ClassifyIntentForEval() error = %v", err)
			}
			if preview.DecisionResult == nil {
				t.Fatalf("preview matched no decision: %+v", preview)
			}
			previewModel, previewCount := decisionModels[preview.DecisionResult.DecisionName], preview.SignalValues["conversation:tool_failures"]
			if model != tc.want || previewModel != model || previewCount != count {
				t.Fatalf("live routes to %q with count %v, preview to %q with count %v; want %q", model, count, previewModel, previewCount, tc.want)
			}
		})
	}
}
