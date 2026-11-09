package extproc

import (
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/historyreset"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

var combinedToolOutput = strings.Repeat("large recoverable tool output ", 200)

// combinedRecoveryConversation has one prior turn reset removes and a live
// turn whose tool result compression stores, so both writers run.
func combinedRecoveryConversation(oldQuestion string) *llmprotocol.Request {
	text := func(role llmprotocol.Role, value string) llmprotocol.Message {
		return llmprotocol.Message{Role: role, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: value}}}
	}
	return &llmprotocol.Request{Model: "model", Messages: []llmprotocol.Message{
		text(llmprotocol.RoleUser, oldQuestion),
		text(llmprotocol.RoleAssistant, "old answer"),
		text(llmprotocol.RoleUser, "live question"),
		{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{
			Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: "call-1", Name: "lookup", Arguments: "{}"},
		}}},
		{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{
			Kind: llmprotocol.ContentToolResult,
			ToolResult: &llmprotocol.ToolResult{CallID: "call-1", Content: []llmprotocol.Content{{
				Kind: llmprotocol.ContentText, Text: combinedToolOutput,
			}}},
		}}},
	}}
}

func combinedRecoveryDecision(compression map[string]interface{}, reset map[string]interface{}, compressionFailureMode string) *config.Decision {
	compressionConfig := map[string]interface{}{
		"enabled": true, "mode": "always",
		"targets": map[string]interface{}{"tool_outputs": map[string]interface{}{
			"mode": "recoverable", "min_tokens": 10, "target_tokens": 30,
		}},
		"recovery": compression,
	}
	if compressionFailureMode != "" {
		compressionConfig["failure_mode"] = compressionFailureMode
	}
	return &config.Decision{Name: "context", Plugins: []config.DecisionPlugin{
		{Type: config.DecisionPluginContextCompression, Configuration: config.MustStructuredPayload(compressionConfig)},
		{Type: config.DecisionPluginHistoryReset, Configuration: config.MustStructuredPayload(map[string]interface{}{
			"enabled": true,
			"trigger": map[string]interface{}{
				"signal": "topic_boundary", "min_confidence": 0.9, "accepted_versions": []string{"v1"},
			},
			"recovery": reset,
		})},
	}}
}

func recoveryBounds(maxBytes int, extra map[string]interface{}) map[string]interface{} {
	bounds := map[string]interface{}{"enabled": true, "store": "redis", "max_bytes_per_request": maxBytes}
	for key, value := range extra {
		bounds[key] = value
	}
	return bounds
}

type combinedRun struct {
	ctx     *RequestContext
	request *llmprotocol.Request
	stored  map[string]contextcompression.RecoveryEntry
	err     error
}

func runCombinedRecovery(t *testing.T, router *OpenAIRouter, decision *config.Decision, oldQuestion string) combinedRun {
	t.Helper()
	request := combinedRecoveryConversation(oldQuestion)
	ctx := recoverableResetContext(t, router, decision, request)
	router.prepareContextHistorySteps(ctx, request)
	err := router.applyContextTransformationPlan(ctx, request)
	run := combinedRun{ctx: ctx, request: request, err: err}
	if stub, ok := router.ContextRecoveryBackend.(*recoveryStoreStub); ok {
		run.stored = stub.entries
	}
	return run
}

func (run combinedRun) toolOutputPreserved() bool {
	last := run.request.Messages[len(run.request.Messages)-1]
	return len(last.Content) == 1 && last.Content[0].ToolResult != nil &&
		last.Content[0].ToolResult.Content[0].Text == combinedToolOutput
}

func (run combinedRun) assertAdvertisedKeysWereStored(t *testing.T) {
	t.Helper()
	for _, key := range run.ctx.ContextRecoveryKeys {
		found := false
		for _, entry := range run.stored {
			found = found || entry.Key == key
		}
		if !found {
			t.Fatal("the retrieval tool advertises a key that was never stored")
		}
	}
}

func TestCompressionWriteStopsAtTheResetBound(t *testing.T) {
	router := recoverableResetRouter()
	run := runCombinedRecovery(t, router, combinedRecoveryDecision(
		recoveryBounds(8192, nil), recoveryBounds(4096, nil), ""), "old question")
	if run.err != nil {
		t.Fatalf("fail_open compression must preserve the request: %v", run.err)
	}
	if !run.toolOutputPreserved() {
		t.Fatal("compression stored content past the merged bound")
	}
	if len(run.stored) != 1 || len(run.ctx.ContextRecoveryKeys) != 1 {
		t.Fatalf("stored %d entries and advertised %d keys, want only reset's", len(run.stored), len(run.ctx.ContextRecoveryKeys))
	}
	run.assertAdvertisedKeysWereStored(t)
}

func TestResetWriteStopsAtTheCompressionBound(t *testing.T) {
	router := recoverableResetRouter()
	run := runCombinedRecovery(t, router, combinedRecoveryDecision(
		recoveryBounds(4096, nil), recoveryBounds(8192, nil), ""), strings.Repeat("old question ", 400))
	if run.ctx.HistoryResetDiagnostics.Reason != historyreset.ReasonRecoveryLimitExceeded {
		t.Fatalf("reset ignored the merged bound: %+v", run.ctx.HistoryResetDiagnostics)
	}
	if !strings.Contains(run.request.Messages[0].Content[0].Text, "old question") {
		t.Fatal("reset removed history it could not store")
	}
	run.assertAdvertisedKeysWereStored(t)
}

// Reset runs before compression in the context stage, so compression gets
// only what reset left of the budget.
func TestCompressionGetsOnlyTheRemainderOfTheRequestBudget(t *testing.T) {
	limit := len(combinedToolOutput) + 200
	oldQuestion := strings.Repeat("old question ", 30)
	for _, failureMode := range []string{"fail_open", "fail_closed"} {
		t.Run(failureMode, func(t *testing.T) {
			router := recoverableResetRouter()
			run := runCombinedRecovery(t, router, combinedRecoveryDecision(
				recoveryBounds(limit, nil), recoveryBounds(limit, nil), failureMode), oldQuestion)
			if run.ctx.HistoryResetDiagnostics.RecoveryStatus != historyreset.RecoveryStored {
				t.Fatalf("reset did not store first: %+v", run.ctx.HistoryResetDiagnostics)
			}
			if len(run.stored) != 1 {
				t.Fatalf("the backing store holds %d entries, want only reset's", len(run.stored))
			}
			for _, entry := range run.stored {
				if entry.Content == combinedToolOutput {
					t.Fatal("the over-budget compression entry reached the backing store")
				}
			}
			run.assertAdvertisedKeysWereStored(t)
			if failureMode == "fail_open" {
				if run.err != nil || !run.toolOutputPreserved() {
					t.Fatalf("fail_open must forward the uncompressed output: err=%v", run.err)
				}
			} else if run.err == nil {
				t.Fatal("fail_closed compression accepted a write past the request budget")
			}
		})
	}
}

func TestRequestsDoNotShareARecoveryBudget(t *testing.T) {
	limit := len(combinedToolOutput) + 200
	router := recoverableResetRouter()
	decision := combinedRecoveryDecision(recoveryBounds(limit, nil), recoveryBounds(limit, nil), "")
	first := runCombinedRecovery(t, router, decision, "old question")
	second := runCombinedRecovery(t, router, decision, "another old question")
	if first.ctx.ContextRecoveryStore == second.ctx.ContextRecoveryStore {
		t.Fatal("two requests share one budgeted store")
	}
	if first.ctx.HistoryResetDiagnostics.RecoveryStatus != historyreset.RecoveryStored ||
		second.ctx.HistoryResetDiagnostics.RecoveryStatus != historyreset.RecoveryStored {
		t.Fatal("a later request inherited an earlier request's spent budget")
	}
}

func miniredisRecoveryRouter(t *testing.T) (*OpenAIRouter, *miniredis.Miniredis) {
	t.Helper()
	server := miniredis.RunT(t)
	port, err := strconv.Atoi(server.Port())
	if err != nil {
		t.Fatal(err)
	}
	redis := &config.RedisConfig{}
	redis.Connection.Host = server.Host()
	redis.Connection.Port = port
	cfg := &config.RouterConfig{}
	cfg.Redis = redis
	return &OpenAIRouter{Config: cfg}, server
}

func TestCompressionEntriesUseTheMergedTTL(t *testing.T) {
	router, server := miniredisRecoveryRouter(t)
	run := runCombinedRecovery(t, router, combinedRecoveryDecision(
		recoveryBounds(1<<20, map[string]interface{}{"ttl_seconds": 900}),
		recoveryBounds(1<<20, map[string]interface{}{"ttl_seconds": 60}), ""), "old question")
	if run.err != nil || len(run.ctx.ContextRecoveryKeys) != 2 {
		t.Fatalf("expected both writers to store: err=%v keys=%d", run.err, len(run.ctx.ContextRecoveryKeys))
	}
	entries := 0
	for _, key := range server.Keys() {
		if !strings.HasPrefix(key, "vsr:context-recovery:v1:entry:") {
			continue
		}
		entries++
		if ttl := server.TTL(key); ttl <= 0 || ttl > 60*time.Second {
			t.Fatalf("entry TTL %s, want at most reset's 60s", ttl)
		}
	}
	if entries != 2 {
		t.Fatalf("found %d stored entries, want 2", entries)
	}
}

// The request is streaming, so reset never runs; binding alone must build the
// store, and with the merged total.
func TestRecoveryStoreIsBuiltWithTheMergedTotal(t *testing.T) {
	router, _ := miniredisRecoveryRouter(t)
	decision := combinedRecoveryDecision(
		recoveryBounds(1<<20, map[string]interface{}{"max_total_bytes": 64 << 20}),
		recoveryBounds(1<<20, map[string]interface{}{"max_total_bytes": 2048}), "")
	ctx := &RequestContext{
		RequestID: "streaming", VSRSelectedDecision: decision, VSRSelectedDecisionName: "context",
		Headers: map[string]string{"x-authz-user-id": "user-1"}, ExpectStreamingResponse: true,
		SemanticRequest: combinedRecoveryConversation("old question"),
	}
	router.bindContextPolicy(ctx)
	if router.ContextRecoveryBackend == nil {
		t.Fatal("binding did not build the shared store")
	}
	_, err := contextcompression.StoreRecoveryContent(t.Context(), router.ContextRecoveryBackend,
		"scope", strings.Repeat("x", 4096), time.Minute)
	if err == nil {
		t.Fatal("the shared store accepted more than the merged total bound")
	}
}
