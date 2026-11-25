package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// writeStalledAnthropicStream starts an answer, as Anthropic does with the
// prompt's usage in message_start, sends some text and then waits, as a long
// answer does, until the client goes away, or with cut ends the stream there.
func writeStalledAnthropicStream(w http.ResponseWriter, r *http.Request, cut bool) {
	w.Header().Set("Content-Type", "text/event-stream")
	for _, event := range []map[string]any{
		{"type": "message_start", "message": map[string]any{
			"id": "msg-slow", "type": "message", "role": "assistant", "model": "slow-model", "content": []any{},
			"stop_reason": nil, "stop_sequence": nil,
			"usage": map[string]any{"input_tokens": 200, "output_tokens": 1, "cache_read_input_tokens": 1500, "cache_creation_input_tokens": 300},
		}},
		{"type": "content_block_start", "index": 0, "content_block": map[string]any{"type": "text", "text": ""}},
		{"type": "content_block_delta", "index": 0, "delta": map[string]any{"type": "text_delta", "text": "partial answer"}},
	} {
		data, _ := json.Marshal(event)
		_, _ = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event["type"], data)
	}
	w.(http.Flusher).Flush()
	if cut {
		return
	}
	select {
	case <-r.Context().Done():
	case <-time.After(10 * time.Second):
	}
}

// toolTurn is a Claude Code request whose history holds a tool call the client
// ran and its result.
func toolTurn(text string) string {
	return `{"model":"vllm-sr/auto","max_tokens":64,"stream":true,` + claudeCodeThinking +
		`,"tools":[{"name":"Bash","description":"Run a shell command","input_schema":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}],` +
		`"messages":[{"role":"user","content":"` + text + `"},` +
		`{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"Bash","input":{"command":"ls"}}]},` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"a.txt b.txt"}]}]}`
}

// A tool result in the history is a tool the client already ran: resending
// the request to the next candidate runs nothing again, so it falls back as a
// request without one does.
func TestFallbackResendsAConversationWithAToolResult(t *testing.T) {
	h := newServedModelHarness(t)
	call := h.send(t, "/v1/messages", toolTurn("HANDOFFMSG list the files"))
	require.Equal(t, http.StatusOK, call.status, "%s", call.body)
	require.Equal(t, []string{"fail-model", "plain-msg"}, call.called)
	require.Contains(t, string(call.body), "ok from plain-msg")
	require.Equal(t, "plain-msg", call.record.SelectedModel)
	outcomes := fallbackOutcomes(call.record)
	require.Len(t, outcomes, 1)
	require.Equal(t, "completed", outcomes[0].Verdict)
}

// A streamed answer the client stops keeps the usage the stream reported
// before it stopped, such as the prompt Anthropic counts in message_start,
// priced and marked partial instead of left unpriced.
func TestAnInterruptedStreamKeepsTheUsageItReported(t *testing.T) {
	h := newServedModelHarness(t)
	record := h.sendAndCancel(t, "/v1/messages",
		`{"model":"vllm-sr/auto","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"SLOWSTREAM"}]}`)
	require.Equal(t, routerreplay.LifecycleAborted, record.LifecycleState)
	require.Equal(t, "client_canceled", record.TerminalReason)
	require.NotNil(t, record.PromptTokens, "the prompt the stream counted was dropped")
	require.Equal(t, 2000, *record.PromptTokens)
	require.Equal(t, 1500, *record.CachedPromptTokens)
	require.NotNil(t, record.ActualCost)
	require.Positive(t, *record.ActualCost)
	require.NotNil(t, record.CostAccounting)
	require.True(t, record.CostAccounting.PartialUsage)
	require.Empty(t, record.CostAccounting.UnpricedReason)

	cut := h.send(t, "/v1/messages",
		`{"model":"vllm-sr/auto","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"CUTSTREAM"}]}`)
	require.Equal(t, routerreplay.LifecycleFailed, cut.record.LifecycleState, cut.record.TerminalReason)
	require.NotNil(t, cut.record.PromptTokens, "the prompt a stream cut by its backend counted was dropped")
	require.Equal(t, 2000, *cut.record.PromptTokens)
	require.NotNil(t, cut.record.ActualCost)
	require.True(t, cut.record.CostAccounting.PartialUsage)
}

// Under a run budget a stream the client stops is charged the price of the
// usage it reported, not its whole reservation, and Replay's coverage counts
// the record priced, as the run view and vllm-sr usage read it.
func TestAnInterruptedStreamSettlesItsBudgetAtThePartialPrice(t *testing.T) {
	h := newServedModelHarness(t, func(document string) string {
		return strings.Replace(document, "  services:\n", "  services:\n    budgets: {currency: USD, run: {limit: 5}}\n", 1)
	})
	record := h.sendAndCancel(t, "/v1/messages",
		`{"model":"vllm-sr/auto","max_tokens":4000,"stream":true,"messages":[{"role":"user","content":"SLOWSTREAM"}]}`,
		routing.HeaderField{Name: "x-claude-code-session-id", Value: t.Name()})
	require.NotNil(t, record.ActualCost)
	require.True(t, record.CostAccounting.PartialUsage)
	budget := record.CostAccounting.Budget
	require.NotNil(t, budget, "the record carries no budget")
	require.Equal(t, store.BudgetSettledPartialUsage, budget.Settlement)
	require.NotNil(t, budget.Charged)
	require.InDelta(t, *record.ActualCost, *budget.Charged, 1e-12)
	require.Greater(t, budget.Estimate, *budget.Charged, "the reservation was charged whole")
	require.InDelta(t, 5-*record.ActualCost, budget.Remaining, 1e-9)

	coverage := buildRouterReplayAggregateCostSummary([]routerreplay.RoutingRecord{record}).Coverage
	require.Equal(t, 1, coverage.PricedRecords, "%+v", coverage)
	require.Zero(t, coverage.PartlyPricedRecords+coverage.UnpricedRecords)
}

// sendAndCancel starts a streamed request, reads until the answer's text
// arrives and then goes away, as a client stopped mid-answer does.
func (h *servedModelHarness) sendAndCancel(t *testing.T, path, body string, header ...routing.HeaderField) routerreplay.RoutingRecord {
	t.Helper()
	request := append(routing.Header{
		{Name: ":method", Value: "POST"},
		{Name: ":path", Value: path},
		{Name: ":authority", Value: "router"},
		{Name: ":scheme", Value: "http"},
		{Name: "content-type", Value: "application/json"},
		{Name: "anthropic-version", Value: "2023-06-01"},
	}, header...)
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	plan, err := h.engine.Plan(ctx, &routing.Request{Header: request, Body: []byte(body)})
	require.NoError(t, err)
	require.Nil(t, plan.Immediate)
	result, err := h.set.Execute(ctx, plan.Call, looperListener)
	require.NoError(t, err)
	answer, err := h.engine.Respond(ctx, plan, &routing.UpstreamResponse{
		Status: result.Response.StatusCode, Header: routingHeaderOf(result.Response.Header), Body: result.Response.Body,
	})
	require.NoError(t, err)
	require.NotNil(t, answer.Stream)
	var received strings.Builder
	for !strings.Contains(received.String(), "partial answer") {
		chunk, err := answer.Stream.Next()
		received.Write(chunk)
		require.NoError(t, err, received.String())
	}
	cancel()
	_ = result.Response.Body.Close()
	plan.Finish(context.Canceled)
	h.backend.takeCalls()
	id := answer.Header.Get("x-vsr-replay-id")
	require.NotEmpty(t, id)
	record, found := h.router.findRouterReplayRecord(id)
	require.True(t, found)
	return record
}
