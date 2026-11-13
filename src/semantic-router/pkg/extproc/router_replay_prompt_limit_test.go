package extproc

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
)

// The prompt a record keeps is cut to its limit, 64 KiB where neither
// max_prompt_bytes nor max_tool_trace_bytes is set, under raw capture and
// under verified capture once the prompt is read clean, with the whole
// prompt's size and SHA-256; none keeps no prompt. The request trace keeps its
// user turn to max_tool_trace_bytes alone.
func TestReplayCutsThePromptToMaxPromptBytesPerMode(t *testing.T) {
	prompt := strings.Repeat("Summarize the meeting agenda. ", 4_000) + "Thanks."
	sum := sha256.Sum256([]byte(prompt))
	for _, test := range []struct {
		name, mode string
		limits     map[string]interface{}
		kept       int
	}{
		{"none", config.PromptCaptureNone, nil, 0},
		{"raw", config.PromptCaptureRaw, nil, 64 << 10},
		{"verified", config.PromptCaptureVerified, nil, 64 << 10},
		{"verified with a trace limit", config.PromptCaptureVerified, map[string]interface{}{"max_tool_trace_bytes": 4096}, 4096},
	} {
		t.Run(test.name, func(t *testing.T) {
			cfg := &config.RouterConfig{
				RouterReplay:       config.RouterReplayConfig{StoreBackend: "memory"},
				IntelligentRouting: config.IntelligentRouting{Signals: config.Signals{PIIRules: []config.PIIRule{{Name: "personal_data"}}}},
			}
			withGlobalPlugins(t, cfg, replayTestPlugin(map[string]interface{}{"enabled": true, "prompt_capture": config.PromptCaptureVerified}))
			router := replayRuntimeRouter(t, cfg)
			router.Classifier = &classification.Classifier{Config: cfg}
			read := make(chan struct{}, 1)
			savedEvidence, savedReadable := replayPromptEvidence, replayPromptReadable
			t.Cleanup(func() { replayPromptEvidence, replayPromptReadable = savedEvidence, savedReadable })
			replayPromptReadable = func(*classification.Classifier, string, time.Duration, bool) bool { return true }
			replayPromptEvidence = func(_ *classification.Classifier, _ context.Context, text string) classification.PrivacyEvidence {
				read <- struct{}{}
				return classification.NewPrivacyEvidence("request", text, true, true)
			}
			ctx := replayPolicyRequest(t)
			ctx.SemanticRequest.Messages[0].Content[0].Text = prompt
			ctx.Routing.SelectRecipe(cfg.DefaultRecipe())
			plugin := map[string]interface{}{"enabled": true, "prompt_capture": test.mode, "capture_request_body": true}
			for key, value := range test.limits {
				plugin[key] = value
			}
			own := replayDecision(plugin)
			router.applyDecisionResultToContext(&decision.DecisionResult{Decision: own}, ctx)
			router.startRouterReplay(ctx, "public-entry", "model", own.Name)
			if test.mode == config.PromptCaptureVerified {
				select {
				case <-read:
				case <-time.After(5 * time.Second):
					t.Fatal("the verified prompt was never read")
				}
			}
			deadline := time.Now().Add(5 * time.Second)
			for len(replayPromptCheckSlots) > 0 && time.Now().Before(deadline) {
				time.Sleep(time.Millisecond)
			}
			record, ok := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
			if !ok {
				t.Fatal("missing record")
			}
			if len(record.Prompt) != test.kept || record.PromptTruncated != (test.kept > 0) {
				t.Fatalf("prompt of %d bytes kept as %d (truncated %v), want %d", len(prompt), len(record.Prompt), record.PromptTruncated, test.kept)
			}
			if test.kept > 0 && (record.PromptOriginalBytes != len(prompt) || record.PromptOriginalSHA256 != hex.EncodeToString(sum[:])) {
				t.Fatalf("the cut does not identify the prompt: %d bytes, sha256 %q", record.PromptOriginalBytes, record.PromptOriginalSHA256)
			}
			if test.mode == config.PromptCaptureRaw {
				if record.ToolTrace == nil || len(record.ToolTrace.Steps) == 0 || record.ToolTrace.Steps[0].Text != prompt {
					t.Fatalf("the request trace's user turn is not kept to max_tool_trace_bytes alone: %+v", record.ToolTrace)
				}
			}
		})
	}
}

// Where no layer sets max_prompt_bytes, a raw record keeps as much of the
// prompt as max_tool_trace_bytes allows, as before max_prompt_bytes existed;
// an explicit value wins, and 0 keeps the whole prompt.
func TestReplayPromptKeepsTheTraceLimitUntilMaxPromptBytesIsSet(t *testing.T) {
	prompt := strings.Repeat("Summarize the meeting agenda. ", 4_000) + "Thanks."
	for _, test := range []struct {
		name   string
		limits map[string]interface{}
		kept   int
	}{
		{"a trace limit", map[string]interface{}{"max_tool_trace_bytes": 4096}, 4096},
		{"an explicit value above it", map[string]interface{}{"max_tool_trace_bytes": 4096, "max_prompt_bytes": 8192}, 8192},
		{"an explicit 0", map[string]interface{}{"max_tool_trace_bytes": 4096, "max_prompt_bytes": 0}, len(prompt)},
	} {
		t.Run(test.name, func(t *testing.T) {
			plugin := map[string]interface{}{"enabled": true, "prompt_capture": config.PromptCaptureRaw, "capture_request_body": true}
			for key, value := range test.limits {
				plugin[key] = value
			}
			cfg := withGlobalPlugins(t, &config.RouterConfig{RouterReplay: config.RouterReplayConfig{StoreBackend: "memory"}}, replayTestPlugin(plugin))
			router := replayRuntimeRouter(t, cfg)
			ctx := replayPolicyRequest(t)
			ctx.SemanticRequest.Messages[0].Content[0].Text = prompt
			ctx.RouterReplayPluginConfig = cfg.EffectiveRouterReplayConfig(nil)
			router.startRouterReplay(ctx, "public-entry", "model", "ordinary")
			record, ok := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
			if !ok {
				t.Fatal("missing record")
			}
			if len(record.Prompt) != test.kept || record.PromptTruncated != (test.kept < len(prompt)) {
				t.Fatalf("prompt of %d bytes kept as %d (truncated %v), want %d", len(prompt), len(record.Prompt), record.PromptTruncated, test.kept)
			}
			if trace := record.ToolTrace; trace == nil || len(trace.Steps) == 0 || len(trace.Steps[0].Text) != 4096 {
				t.Fatalf("the request trace's user turn is not cut to max_tool_trace_bytes: %+v", trace)
			}
		})
	}
}
