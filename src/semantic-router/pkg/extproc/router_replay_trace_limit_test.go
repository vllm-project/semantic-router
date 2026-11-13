package extproc

import (
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

func replayTraceRecord(t *testing.T, limits map[string]interface{}, messages []llmprotocol.Message) routerreplay.RoutingRecord {
	t.Helper()
	plugin := map[string]interface{}{"enabled": true, "prompt_capture": config.PromptCaptureRaw, "capture_request_body": true}
	for key, value := range limits {
		plugin[key] = value
	}
	cfg := withGlobalPlugins(t, &config.RouterConfig{RouterReplay: config.RouterReplayConfig{StoreBackend: "memory"}}, replayTestPlugin(plugin))
	router := replayRuntimeRouter(t, cfg)
	ctx := replayPolicyRequest(t)
	ctx.SemanticRequest.Messages = messages
	ctx.RouterReplayPluginConfig = cfg.EffectiveRouterReplayConfig(nil)
	router.startRouterReplay(ctx, "public-entry", "model", "ordinary")
	record, ok := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
	if !ok {
		t.Fatal("missing record")
	}
	return record
}

func replayUserTurn(text string) llmprotocol.Message {
	return llmprotocol.Message{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}}}
}

func replayTraceBytes(trace *routerreplay.ToolTrace) int {
	total := 0
	for _, step := range trace.Steps {
		total += len(step.Text) + len(step.Arguments) + len(step.RawArguments) + len(step.Output) + len(step.RawOutput)
	}
	return total
}

// Under raw, where no layer sets a trace limit, a record's request trace keeps
// each text field to 64 KiB, raw copies included, and its steps to 256 KiB in
// all, dropping the oldest steps first. A max_tool_trace_bytes a layer sets,
// 0 included, keeps the trace as it was before these defaults.
func TestReplayBoundsTheRequestTraceByDefault(t *testing.T) {
	long := strings.Repeat("Summarize the meeting agenda. ", 140_000)
	conversation := make([]llmprotocol.Message, 100)
	for i := range conversation {
		conversation[i] = replayUserTurn(fmt.Sprintf("Turn %03d. ", i) + strings.Repeat("Notes on the agenda. ", 400))
	}
	arguments := `{"query":"` + strings.Repeat("q", 100_000) + `"}`
	call := llmprotocol.Message{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{
		Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: "call-1", Name: "search", Arguments: arguments},
	}}}

	t.Run("a long single turn keeps the prompt's limit and 64 KiB of its turn", func(t *testing.T) {
		record := replayTraceRecord(t, nil, []llmprotocol.Message{replayUserTurn(long)})
		if len(record.Prompt) != 64<<10 || !record.PromptTruncated {
			t.Fatalf("prompt of %d bytes kept as %d", len(long), len(record.Prompt))
		}
		steps := record.ToolTrace.Steps
		if len(steps) != 1 || len(steps[0].Text) != 64<<10 || !steps[0].Truncated || record.ToolTrace.StepsTruncated {
			t.Fatalf("the user turn of %d bytes is not kept to 64 KiB: %d steps, %d bytes", len(long), len(steps), len(steps[0].Text))
		}
	})
	t.Run("a 100-turn conversation stays within 256 KiB, newest turns kept", func(t *testing.T) {
		record := replayTraceRecord(t, nil, conversation)
		trace := record.ToolTrace
		kept := len(trace.Steps)
		if replayTraceBytes(trace) > 256<<10 || !trace.StepsTruncated || trace.DroppedStepCount != 100-kept || kept == 0 {
			t.Fatalf("%d steps of %d bytes kept, %d dropped (truncated %v)", kept, replayTraceBytes(trace), trace.DroppedStepCount, trace.StepsTruncated)
		}
		if !strings.HasPrefix(trace.Steps[kept-1].Text, "Turn 099.") || !strings.HasPrefix(trace.Steps[0].Text, fmt.Sprintf("Turn %03d.", 100-kept)) {
			t.Fatalf("the oldest turns were not the ones dropped: %q ... %q", trace.Steps[0].Text[:10], trace.Steps[kept-1].Text[:10])
		}
	})
	t.Run("a tool call's raw copy is cut with it", func(t *testing.T) {
		record := replayTraceRecord(t, nil, []llmprotocol.Message{replayUserTurn("Search for it."), call})
		step := record.ToolTrace.Steps[1]
		if len(step.Arguments) != 64<<10 || len(step.RawArguments) != 64<<10 || !step.Truncated || !step.RawTruncated {
			t.Fatalf("arguments %d and raw %d bytes (truncated %v, raw %v), want 64 KiB each, flagged", len(step.Arguments), len(step.RawArguments), step.Truncated, step.RawTruncated)
		}
	})
	t.Run("an explicit limit keeps whole raw copies and every step", func(t *testing.T) {
		record := replayTraceRecord(t, map[string]interface{}{"max_tool_trace_bytes": 4096}, append(append([]llmprotocol.Message{}, conversation...), call))
		trace := record.ToolTrace
		if len(trace.Steps) != 100 || trace.DroppedStepCount != 1 {
			t.Fatalf("%d steps kept, %d dropped; want max_tool_trace_steps' 100 and the oldest one dropped", len(trace.Steps), trace.DroppedStepCount)
		}
		last := trace.Steps[99]
		if len(trace.Steps[0].Text) != 4096 || len(last.Arguments) != 4096 || last.RawArguments != arguments || last.RawTruncated {
			t.Fatalf("turns %d bytes, call %d bytes, raw copy %d bytes (cut %v)", len(trace.Steps[0].Text), len(last.Arguments), len(last.RawArguments), last.RawTruncated)
		}
	})
	t.Run("an explicit 0 keeps whole turns", func(t *testing.T) {
		record := replayTraceRecord(t, map[string]interface{}{"max_tool_trace_bytes": 0}, []llmprotocol.Message{replayUserTurn(long)})
		if text := record.ToolTrace.Steps[0].Text; text != strings.TrimSpace(long) || record.ToolTrace.Steps[0].Truncated {
			t.Fatalf("the user turn of %d bytes kept as %d", len(long), len(text))
		}
	})
	t.Run("an explicit total keeps the field default", func(t *testing.T) {
		record := replayTraceRecord(t, map[string]interface{}{"max_tool_trace_total_bytes": 65536}, conversation)
		trace := record.ToolTrace
		if replayTraceBytes(trace) > 64<<10 || len(trace.Steps) == 0 || trace.DroppedStepCount != 100-len(trace.Steps) {
			t.Fatalf("%d steps of %d bytes kept under a 64 KiB total", len(trace.Steps), replayTraceBytes(trace))
		}
	})
}
