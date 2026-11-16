package extproc

import (
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func answeredWith(stop llmprotocol.StopReason) []routerreplay.Outcome {
	return []routerreplay.Outcome{{
		Source: primaryResponseOutcomeSource, Target: "model", Verdict: "completed",
		Metadata: map[string]string{primaryResponseStopReason: string(stop)},
	}}
}

// newDisplayRunRouter holds a Claude Code run kept without tool traces: the
// main agent asks for tools on big, side requests go to small, the Router
// refuses three calls itself, a fallback and a Looper call answer on big
// after a candidate failed or a budget refused a hop, and the last call
// answers. A session of only connect checks sits beside it.
func newDisplayRunRouter(t *testing.T) (*OpenAIRouter, map[string]string) {
	t.Helper()
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(30, 0))
	recorder.SetCapturePolicy(true, true, 0)
	router := &OpenAIRouter{ReplayStoreShared: true, ReplayRecorder: recorder}
	start := time.Date(2026, 11, 16, 9, 0, 0, 0, time.UTC)
	usd := "USD"
	cost := func(value float64) *float64 { return &value }
	tokens := func(value int) *int { return &value }
	claude := &store.AgentIdentity{Harness: store.HarnessClaudeCode}
	completed := func(id string, at int, model string, outcomes []routerreplay.Outcome) routerreplay.RoutingRecord {
		return routerreplay.RoutingRecord{
			ID: id, RequestID: "req-" + id, SessionID: "cc-display", Timestamp: start.Add(time.Duration(at) * time.Second),
			OriginalModel: "vllm-sr/auto", SelectedModel: model, Agent: claude,
			LifecycleState: routerreplay.LifecycleCompleted, ResponseStatus: 200,
			PromptTokens: tokens(100), CompletionTokens: tokens(20), TotalTokens: tokens(120), ActualCost: cost(0.001), Currency: &usd,
			Outcomes: outcomes,
		}
	}
	refused := func(id string, at, status int, reason string) routerreplay.RoutingRecord {
		return routerreplay.RoutingRecord{
			ID: id, SessionID: "cc-display", Timestamp: start.Add(time.Duration(at) * time.Second), Agent: claude,
			OriginalModel: "vllm-sr/auto", LifecycleState: routerreplay.LifecycleFailed, ResponseStatus: status, TerminalReason: reason,
		}
	}
	hop := func(id string, at int, model string) routerreplay.RoutingRecord {
		return routerreplay.RoutingRecord{
			ID: id, ParentRequestID: "req-looper", SessionID: "cc-display", Timestamp: start.Add(time.Duration(at) * time.Second),
			SelectedModel: model, Agent: claude, LifecycleState: routerreplay.LifecycleCompleted, ResponseStatus: 200,
			CostAccounting: &store.CostAccounting{UnpricedReason: store.UnpricedCountedInParent},
		}
	}

	auxiliary := completed("aux", 1, "small", nil)
	auxiliary.RequestKind, auxiliary.Signals = store.RequestKindAuxiliary, routerreplay.Signal{Agent: []string{"housekeeping"}}
	auxiliary.ActualCost, auxiliary.Currency = nil, nil
	auxiliary.CostAccounting = &store.CostAccounting{UnpricedReason: store.UnpricedPricingNotConfigured}
	compaction := completed("compaction", 3, "small", nil)
	compaction.RequestKind = store.RequestKindCompaction
	block := refused("block", 4, 400, replayTerminalRejected)
	// The same block on a stream: an error event under the 200 it began with.
	streamBlock := refused("stream-block", 4, 200, replayTerminalRejected)
	block.Decision, block.Signals = "block_kw", routerreplay.Signal{Keyword: []string{"kw_block"}}
	guarded := completed("guarded", 5, "big", []routerreplay.Outcome{{Source: "router", Target: "response_guard", Verdict: "matched"}})
	guarded.LifecycleState, guarded.ResponseStatus, guarded.TerminalReason = routerreplay.LifecycleFailed, 400, replayTerminalRejected
	// A streamed answer the guard refused after its 200 began.
	streamGuarded := completed("guarded-stream", 5, "big", nil)
	streamGuarded.LifecycleState, streamGuarded.TerminalReason = routerreplay.LifecycleFailed, replayTerminalRejected
	fellBack := completed("fallback", 7, "fbbad", []routerreplay.Outcome{{
		Source: fallbackOutcomeSource, Target: "request", TargetRef: "big", Verdict: "completed", Reason: "succeeded",
		Metadata: map[string]string{"attempts": "2", "initial_model": "fbbad", "fallback_model": "big"},
	}})
	looper := completed("looper", 8, "big", nil)
	looper.ActualCost, looper.Currency = nil, nil
	looper.CostAccounting = &store.CostAccounting{RouterCalls: []store.RouterCallCost{
		{Kind: store.RouterCallKindLooper, Calls: 2, Cost: cost(0.002), Currency: "USD"},
	}}
	refusedHop := hop("hop-refused", 8, "big")
	refusedHop.LifecycleState, refusedHop.ResponseStatus, refusedHop.TerminalReason = routerreplay.LifecycleFailed, 429, budgetExceededCode
	// Claude Code's quota probe, which a provider answers 429 through a proxy,
	// with the failed outcome the Router's fallback audit appends.
	quota := refused("quota", 10, 429, "upstream_error_response")
	quota.RequestKind, quota.SelectedModel = store.RequestKindAuxiliary, "small"
	quota.Outcomes = []routerreplay.Outcome{{Source: fallbackOutcomeSource, Target: "request", Verdict: "failed", Reason: "exhausted"}}
	probe := completed("probe", 20, "big", nil)
	probe.SessionID, probe.RequestKind = "connect-check", store.RequestKindProbe
	// The quota check claude --continue sends under a session of its own.
	final := completed("final", 9, "big", answeredWith(llmprotocol.StopEndTurn))
	final.CostAccounting = &store.CostAccounting{PricingSource: store.PricingSourceCatalogList}
	startQuota := refused("start-quota", 21, 429, "upstream_error_response")
	startQuota.SessionID, startQuota.RequestKind = "quota-only", store.RequestKindAuxiliary

	ids := make(map[string]string)
	for _, record := range []routerreplay.RoutingRecord{
		completed("main-1", 0, "big", answeredWith(llmprotocol.StopToolCall)),
		auxiliary,
		completed("main-2", 2, "big", answeredWith(llmprotocol.StopToolCall)),
		compaction, block, streamBlock, guarded, streamGuarded,
		refused("budget", 6, 429, budgetExceededCode),
		fellBack, looper, hop("hop-small", 8, "small"), hop("hop-big", 8, "big"), refusedHop,
		final, quota, probe, startQuota,
	} {
		id, err := router.ReplayRecorder.AddRecord(record)
		require.NoError(t, err)
		ids[record.ID] = id
	}
	return router, ids
}

// A run shows what happened on each call: a refusal by the Router is a
// block, a side request is no switch of its agent, an agent rule is a
// matched signal, a model counts the calls it answered, and a run kept
// without tool traces reads its final answer from the stop reason.
func TestReplayRunShowsWhatHappened(t *testing.T) {
	router, ids := newDisplayRunRouter(t)
	var run routerreplay.RunResponse
	require.Equal(t, 200, replayAPIJSON(t, router, "/api/v1/observability/replays/run?session_id=cc-display", &run))

	markers := make(map[string][]string)
	for _, call := range run.Calls {
		markers[call.RecordID] = call.Markers
	}
	for _, id := range []string{"block", "stream-block", "guarded", "guarded-stream", "budget"} {
		require.Equal(t, []string{routerreplay.RunMarkerBlock}, markers[ids[id]], id)
	}
	require.Equal(t, []string{routerreplay.RunMarkerFallback}, markers[ids["fallback"]])
	require.Equal(t, []string{routerreplay.RunMarkerFailure}, markers[ids["quota"]], "the call list keeps a failed side request")
	require.Equal(t, map[string]int{routerreplay.RunMarkerBlock: 6, routerreplay.RunMarkerFallback: 1}, run.Run.Events,
		"a budget-refused Looper hop is a block too, no side request switches the main agent's model, and a failed side request is no failure of the run")

	byID := make(map[string]routerreplay.RunCall)
	for _, call := range run.Calls {
		byID[call.RecordID] = call
	}
	require.Equal(t, []string{"agent:housekeeping"}, byID[ids["aux"]].Signals)
	require.Equal(t, "big", byID[ids["fallback"]].Model, "the fallback's answer came from big")
	require.Equal(t, []routerreplay.AggregateValue{{Name: "big", Value: 7}, {Name: "small", Value: 3}}, run.Run.Models)
	require.Equal(t, []string{"big"}, run.Run.Agents[0].Models, "the side requests on small are no model the main agent works on")

	require.Equal(t, 1, run.Run.Cost.UnpricedRecords, "only the side request without a price is unpriced")
	require.Equal(t, routerreplay.RunStatusAnswered, run.Run.Outcome.Status)
	require.Equal(t, ids["final"], run.Run.Outcome.FinalAnswerRecordID)
	require.False(t, byID[ids["main-2"]].FinalAnswer, "a turn that stopped for a tool is no answer")
	require.Equal(t, store.PricingSourceCatalogList, byID[ids["final"]].PricingSource, "a cost at the catalog's list price says so")
	require.Empty(t, byID[ids["main-2"]].PricingSource)
	require.Empty(t, run.Run.Outcome.Reported, "the Router's own fallback outcome is no reported outcome")
	require.Empty(t, byID[ids["quota"]].Outcomes)

	var runs routerreplay.RunListResponse
	require.Equal(t, 200, replayAPIJSON(t, router, "/api/v1/observability/replays/runs", &runs))
	require.Equal(t, 1, runs.Total, "a session of only connect checks or only a quota check is no run")
	require.Equal(t, "cc-display", runs.Data[0].SessionID)
	require.Equal(t, routerreplay.RunStatusAnswered, runs.Data[0].Outcome.Status)

	var aggregate routerreplay.AggregateResponse
	require.Equal(t, 200, replayAPIJSON(t, router, "/api/v1/observability/replays/aggregate", &aggregate))
	coverage := aggregate.Summary.Coverage
	require.Equal(t, 5, coverage.RefusedRecords, "the streamed block is refused like the buffered one, and the quota checks cost nothing")
	require.Equal(t, 1, coverage.UnpricedRecords)
	require.Equal(t, 3, coverage.LooperHopRecords)
}

// A run's calls name every signal type a record keeps, so no new signal
// type reads as none matched.
func TestReplaySignalGroupsCoverEverySignalType(t *testing.T) {
	signals := routerreplay.Signal{}
	value := reflect.ValueOf(&signals).Elem()
	for index := range value.NumField() {
		value.Field(index).Set(reflect.ValueOf([]string{"x"}))
	}
	groups := replaySignalGroups(signals)
	for index := range value.NumField() {
		name, _, _ := strings.Cut(value.Type().Field(index).Tag.Get("json"), ",")
		require.Equal(t, []string{"x"}, groups[name], name)
	}
}

// The digest of an answer keeps the stop reason it ended with under any
// capture, buffered or streamed, so a run tells a final answer from a turn
// that asked for a tool without a tool trace.
func TestPrimaryResponseKeepsItsStopReason(t *testing.T) {
	toolCall := []byte(`{"id":"r1","object":"chat.completion","model":"small","choices":[{"index":0,"message":{"role":"assistant","content":"Let me check.","tool_calls":[{"id":"call_1","type":"function","function":{"name":"clock","arguments":"{}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":20,"completion_tokens":5,"total_tokens":25}}`)
	for name, test := range map[string]struct {
		reply func(*testing.T, *OpenAIRouter, *RequestContext)
		want  llmprotocol.StopReason
	}{
		"answer": {func(t *testing.T, r *OpenAIRouter, ctx *RequestContext) {
			runBufferedReply(t, r, ctx, 200, chatAnswer("done"), "")
		}, llmprotocol.StopEndTurn},
		"tool call with text": {func(t *testing.T, r *OpenAIRouter, ctx *RequestContext) {
			runBufferedReply(t, r, ctx, 200, toolCall, "")
		}, llmprotocol.StopToolCall},
		"streamed answer": {func(t *testing.T, r *OpenAIRouter, ctx *RequestContext) {
			ctx.ExpectStreamingResponse = true
			chatStream(t, r, ctx, []string{"do", "ne"}, true)
		}, llmprotocol.StopEndTurn},
	} {
		t.Run(name, func(t *testing.T) {
			router, ctx := startSessionReplay(t, "none", map[string]string{}, func(*RequestContext) {})
			test.reply(t, router, ctx)
			record, ok := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
			require.True(t, ok)
			stops := []string{}
			for _, outcome := range record.Outcomes {
				if outcome.Source == primaryResponseOutcomeSource {
					stops = append(stops, outcome.Metadata[primaryResponseStopReason])
				}
			}
			require.Equal(t, []string{string(test.want)}, stops)
			require.Equal(t, test.want == llmprotocol.StopEndTurn, replayEndsWithFinalAnswer(record))
		})
	}
}

// A call no provider answered, whose upstream connection was reset before any
// response, cost nothing: it is counted apart, not as unpriced. A provider's
// own error reply, and a stream reset after its 200 began, got a response and
// stay unpriced.
func TestReplayCostCountsACallNoProviderAnsweredApart(t *testing.T) {
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(30, 0))
	recorder.SetCapturePolicy(true, true, 0)
	router := &OpenAIRouter{ReplayStoreShared: true, ReplayRecorder: recorder}
	start := time.Date(2026, 11, 17, 9, 0, 0, 0, time.UTC)
	usd, price := "USD", 0.002
	failed := func(id string, at, status int, reason string) routerreplay.RoutingRecord {
		return routerreplay.RoutingRecord{
			ID: id, SessionID: "cc-down", Timestamp: start.Add(time.Duration(at) * time.Second), OriginalModel: "vllm-sr/auto",
			SelectedModel: "big", Agent: &store.AgentIdentity{Harness: store.HarnessClaudeCode},
			LifecycleState: routerreplay.LifecycleFailed, ResponseStatus: status, TerminalReason: reason,
			CostAccounting: &store.CostAccounting{PricingSource: store.PricingSourceConfigured},
		}
	}
	priced := failed("priced", 0, 200, "response_complete")
	priced.LifecycleState, priced.ActualCost, priced.Currency = routerreplay.LifecycleCompleted, &price, &usd
	for _, record := range []routerreplay.RoutingRecord{
		priced,
		failed("reset", 1, 503, "upstream_reset"),
		failed("provider-error", 2, 502, "upstream_error_response"),
		failed("stream-reset", 3, 200, "upstream_reset"),
	} {
		_, err := router.ReplayRecorder.AddRecord(record)
		require.NoError(t, err)
	}

	var aggregate routerreplay.AggregateResponse
	require.Equal(t, 200, replayAPIJSON(t, router, "/api/v1/observability/replays/aggregate", &aggregate))
	coverage := aggregate.Summary.Coverage
	require.Equal(t, 1, coverage.UnansweredRecords, "the reset before any response")
	require.Equal(t, 1, coverage.PricedRecords)
	require.Equal(t, 2, coverage.UnpricedRecords, "the provider's error reply and the reset stream")
	var run routerreplay.RunResponse
	require.Equal(t, 200, replayAPIJSON(t, router, "/api/v1/observability/replays/run?session_id=cc-down", &run))
	require.Equal(t, 2, run.Run.Cost.UnpricedRecords)
}

// Claude Code sends its title request with the first prompt: a title that
// starts a moment before the first turn still names the run, not the first
// prompt, and a later /rename names it after both.
func TestReplayRunNameTakesTheTitleOverTheFirstPrompt(t *testing.T) {
	start := time.Date(2026, 11, 17, 9, 0, 0, 0, time.UTC)
	record := func(id string, at time.Duration, kind, name string) routerreplay.RoutingRecord {
		return routerreplay.RoutingRecord{
			ID: id, SessionID: "cc-named", Timestamp: start.Add(at), RequestKind: kind, SessionName: name,
			Agent: &store.AgentIdentity{Harness: store.HarnessClaudeCode}, LifecycleState: routerreplay.LifecycleCompleted,
		}
	}
	prompted := func(id string, at time.Duration) routerreplay.RoutingRecord {
		first := record(id, at, "", "Use the Explore subagent to find where add is defined")
		first.SessionNameFromPrompt = true
		return first
	}
	for name, test := range map[string]struct {
		records []routerreplay.RoutingRecord
		want    string
	}{
		"title before the first turn": {[]routerreplay.RoutingRecord{
			record("title", 0, store.RequestKindSessionTitle, "Calc add bug"),
			prompted("first", 100*time.Millisecond),
		}, "Calc add bug"},
		"headless run": {[]routerreplay.RoutingRecord{
			prompted("first", 0),
			record("second", time.Second, "", ""),
		}, "Use the Explore subagent to find where add is defined"},
		"renamed later": {[]routerreplay.RoutingRecord{
			record("title", 0, store.RequestKindSessionTitle, "Calc add bug"),
			prompted("first", 100*time.Millisecond),
			record("renamed", time.Minute, "", "Fix add in calc.py"),
		}, "Fix add in calc.py"},
	} {
		t.Run(name, func(t *testing.T) {
			recorder := routerreplay.NewRecorder(store.NewMemoryStore(30, 0))
			router := &OpenAIRouter{ReplayStoreShared: true, ReplayRecorder: recorder}
			for _, entry := range test.records {
				_, err := recorder.AddRecord(entry)
				require.NoError(t, err)
			}
			var run routerreplay.RunResponse
			require.Equal(t, 200, replayAPIJSON(t, router, "/api/v1/observability/replays/run?session_id=cc-named", &run))
			require.Equal(t, test.want, run.Run.Name)
		})
	}
}
