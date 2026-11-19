package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/spendbudget"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/upstream"
)

// spendRouteConfig is a priced Router with a Looper escalation (small-model,
// then big-model), a Looper panel that calls three models at once, a
// fallback chain whose models both fail, a model whose
// answer cannot be read, a Looper panel of three Messages models, and a
// plain route. Every Chat answer reports 2000 input tokens, 1500 of them
// cache reads and 300 cache writes, and 100 output tokens: far more than a
// short request's estimate. Every Messages answer reports 10000 one-hour
// cache writes and nothing else.
const spendRouteConfig = `
version: v0.5
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
    identity: {trust_headers: true}
providers:
  defaults:
    model: main-model
  models:
    - name: main-model
      provider_model_id: main-model
      api_format: openai
      max_output_tokens: 1000
      pricing: {currency: USD, prompt_per_1m: 3.0, cached_input_per_1m: 0.3, cache_write_per_1m: 3.75, completion_per_1m: 15.0}
      backend_refs: [{name: fake, endpoint: BACKEND, protocol: http, provider: vllm}]
    - name: small-model
      provider_model_id: small-model
      api_format: openai
      max_output_tokens: 1000
      param_size: 1B
      pricing: {currency: USD, prompt_per_1m: 0.5, cached_input_per_1m: 0.05, cache_write_per_1m: 0.6, completion_per_1m: 2.0}
      backend_refs: [{name: fake, endpoint: BACKEND, protocol: http, provider: vllm}]
    - name: big-model
      provider_model_id: big-model
      api_format: openai
      max_output_tokens: 1000
      param_size: 70B
      pricing: {currency: USD, prompt_per_1m: 2.0, cached_input_per_1m: 0.2, cache_write_per_1m: 2.5, completion_per_1m: 8.0}
      backend_refs: [{name: fake, endpoint: BACKEND, protocol: http, provider: vllm}]
    - name: fail-a
      provider_model_id: fail-a
      api_format: openai
      max_output_tokens: 1000
      pricing: {currency: USD, prompt_per_1m: 4.0, completion_per_1m: 20.0}
      backend_refs: [{name: fake, endpoint: BACKEND, protocol: http, provider: vllm}]
    - name: fail-b
      provider_model_id: fail-b
      api_format: openai
      max_output_tokens: 1000
      pricing: {currency: USD, prompt_per_1m: 4.0, completion_per_1m: 20.0}
      backend_refs: [{name: fake, endpoint: BACKEND, protocol: http, provider: vllm}]
    - name: tier-a
      provider_model_id: tier-a
      api_format: anthropic
      max_output_tokens: 1000
      pricing: {currency: USD, prompt_per_1m: 3.0, cache_write_per_1m: 3.75, cache_write_1h_per_1m: 6.0, completion_per_1m: 15.0}
      backend_refs: [{name: fake, endpoint: BACKEND, protocol: http, provider: vllm}]
    - name: tier-b
      provider_model_id: tier-b
      api_format: anthropic
      max_output_tokens: 1000
      pricing: {currency: USD, prompt_per_1m: 3.0, cache_write_per_1m: 3.75, cache_write_1h_per_1m: 6.0, completion_per_1m: 15.0}
      backend_refs: [{name: fake, endpoint: BACKEND, protocol: http, provider: vllm}]
    - name: tier-c
      provider_model_id: tier-c
      api_format: anthropic
      max_output_tokens: 1000
      pricing: {currency: USD, prompt_per_1m: 3.0, cache_write_per_1m: 3.75, cache_write_1h_per_1m: 6.0, completion_per_1m: 15.0}
      backend_refs: [{name: fake, endpoint: BACKEND, protocol: http, provider: vllm}]
    - name: broken-model
      provider_model_id: broken-model
      api_format: openai
      max_output_tokens: 1000
      pricing: {currency: USD, prompt_per_1m: 1.0, completion_per_1m: 5.0}
      backend_refs: [{name: fake, endpoint: BACKEND, protocol: http, provider: vllm}]
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: default
    default: true
recipes:
  - name: default
    signals:
      keywords:
        - {name: looper_kw, operator: OR, keywords: ["LOOPER"], case_sensitive: true}
        - {name: panel_kw, operator: OR, keywords: ["PANEL"], case_sensitive: true}
        - {name: failover_kw, operator: OR, keywords: ["FAILOVER"], case_sensitive: true}
        - {name: fail_kw, operator: OR, keywords: ["FAILONLY"], case_sensitive: true}
        - {name: broken_kw, operator: OR, keywords: ["BROKEN"], case_sensitive: true}
        - {name: tiered_kw, operator: OR, keywords: ["TIERED"], case_sensitive: true}
    decisions:
      - name: looper
        priority: 200
        rules: {operator: AND, conditions: [{type: keyword, name: looper_kw}]}
        modelRefs: [{model: small-model}, {model: big-model}]
        algorithm:
          type: escalate
          budget: {deadline: 30s, max_calls: 2}
          on_error: last_attempt
          escalate:
            order: size
            accept:
              confidence: {method: avg_logprob, gte: 0.8}
      - name: panel
        priority: 200
        rules: {operator: AND, conditions: [{type: keyword, name: panel_kw}]}
        modelRefs: [{model: small-model}, {model: big-model}, {model: main-model}]
        algorithm:
          type: ratings
          on_error: skip
          ratings: {max_concurrent: 3}
      - name: tiered
        priority: 200
        rules: {operator: AND, conditions: [{type: keyword, name: tiered_kw}]}
        modelRefs: [{model: tier-a}, {model: tier-b}, {model: tier-c}]
        algorithm:
          type: ratings
          on_error: skip
          ratings: {max_concurrent: 3}
      - name: failover
        priority: 200
        rules: {operator: AND, conditions: [{type: keyword, name: failover_kw}]}
        modelRefs: [{model: fail-a}, {model: fail-b}]
        fallback: {enabled: true, max_attempts: 2}
      - name: fail
        priority: 200
        rules: {operator: AND, conditions: [{type: keyword, name: fail_kw}]}
        modelRefs: [{model: fail-a}]
      - name: broken
        priority: 200
        rules: {operator: AND, conditions: [{type: keyword, name: broken_kw}]}
        modelRefs: [{model: broken-model}]
      - name: default
        priority: 1
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: main-model}]
global:
  services:
    response_api: {enabled: true, store_backend: memory}
    router_replay: {store_backend: memory}
BUDGETS
  plugins:
    - type: router_replay
      configuration: {enabled: true}
`

// spendBackend answers as the vx mock does: fail-a and fail-b with 503,
// broken-model with a body no codec reads, small-model with low
// log probabilities to a LOOPER question, so escalation moves on.
func spendBackend(t *testing.T) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Model    string          `json:"model"`
			Logprobs bool            `json:"logprobs"`
			Messages json.RawMessage `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		w.Header().Set("content-type", "application/json")
		if strings.HasSuffix(r.URL.Path, "/messages") {
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id": "msg_1", "type": "message", "role": "assistant", "model": req.Model,
				"content":     []any{map[string]any{"type": "text", "text": "ok from " + req.Model}},
				"stop_reason": "end_turn",
				"usage": map[string]any{
					"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 10000,
					"cache_creation": map[string]any{"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 10000},
				},
			})
			return
		}
		switch req.Model {
		case "fail-a", "fail-b":
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = io.WriteString(w, `{"error":{"message":"overloaded","type":"server_error"}}`)
			return
		case "broken-model":
			_, _ = io.WriteString(w, `not an answer`)
			return
		}
		logprob := -0.01
		if req.Model == "small-model" && strings.Contains(string(req.Messages), "LOOPER") {
			logprob = -2.5
		}
		choice := map[string]any{"index": 0, "finish_reason": "stop", "message": map[string]any{"role": "assistant", "content": "ok from " + req.Model}}
		if req.Logprobs {
			choice["logprobs"] = map[string]any{"content": []map[string]any{{"token": "ok", "logprob": logprob, "bytes": nil, "top_logprobs": []any{}}}}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": req.Model, "choices": []any{choice},
			"usage": map[string]any{
				"prompt_tokens": 2000, "completion_tokens": 100, "total_tokens": 2100,
				"prompt_tokens_details": map[string]any{"cached_tokens": 1500, "cache_write_tokens": 300},
			},
		})
	}))
	t.Cleanup(server.Close)
	return server
}

type spendRoute struct {
	t      *testing.T
	router *OpenAIRouter
	engine routing.Engine
	set    *upstream.Set
	calls  atomic.Int64
}

func newSpendRoute(t *testing.T, budgets string) *spendRoute {
	t.Helper()
	backend := spendBackend(t)
	yaml := strings.NewReplacer("BACKEND", strings.TrimPrefix(backend.URL, "http://"), "BUDGETS", budgets).Replace(spendRouteConfig)
	cfg, err := config.ParseYAMLBytes([]byte(yaml))
	require.NoError(t, err)
	router := newLooperRouter(t, cfg)
	opts := routing.DefaultOptions
	opts.ExecutesFallback = true
	return &spendRoute{t: t, router: router, engine: routing.NewEngine(router, opts), set: router.upstreamSet()}
}

// send serves one client request as the standalone gateway does and returns
// the client's response.
func (h *spendRoute) send(path, body string, requestHeaders map[string]string) *routing.Response {
	h.t.Helper()
	header := routing.Header{
		{Name: ":method", Value: "POST"},
		{Name: ":path", Value: path},
		{Name: ":authority", Value: "router"},
		{Name: ":scheme", Value: "http"},
		{Name: "content-type", Value: "application/json"},
		{Name: "x-request-id", Value: fmt.Sprintf("%s-%d", h.t.Name(), h.calls.Add(1))},
	}
	for name, value := range requestHeaders {
		header = append(header, routing.HeaderField{Name: name, Value: value})
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	plan, err := h.engine.Plan(ctx, &routing.Request{Header: header, Body: []byte(body)})
	require.NoError(h.t, err)
	defer plan.Finish(nil)
	if plan.Immediate != nil {
		return plan.Immediate
	}
	result, err := h.set.Execute(ctx, plan.Call, "http-8899")
	require.NoError(h.t, err)
	if result.Immediate != nil {
		return result.Immediate
	}
	defer result.Response.Body.Close()
	out, err := h.engine.Respond(ctx, plan, &routing.UpstreamResponse{
		Status: result.Response.StatusCode, Header: routingHeaderOf(result.Response.Header), Body: result.Response.Body,
	})
	require.NoError(h.t, err)
	if out.Stream != nil {
		var streamed []byte
		for {
			chunk, err := out.Stream.Next()
			streamed = append(streamed, chunk...)
			if err != nil {
				break
			}
		}
		out.Body, out.Stream = streamed, nil
	}
	return out
}

func (h *spendRoute) chat(text, session string) *routing.Response {
	h.t.Helper()
	body := fmt.Sprintf(`{"model":"vllm-sr/auto","max_tokens":100,"messages":[{"role":"user","content":%q}]}`, text)
	return h.send("/v1/chat/completions", body, map[string]string{
		"x-claude-code-session-id": h.t.Name() + "/" + session, headers.AuthzUserID: "u-" + session,
	})
}

func (h *spendRoute) record(response *routing.Response) routerreplay.RoutingRecord {
	h.t.Helper()
	id := response.Header.Get(headers.RouterReplayID)
	require.NotEmpty(h.t, id, "the response names no replay record")
	recorders := []*routerreplay.Recorder{h.router.ReplayRecorder}
	for _, recorder := range h.router.ReplayRecorders {
		recorders = append(recorders, recorder)
	}
	for _, recorder := range recorders {
		if recorder == nil {
			continue
		}
		if record, ok := recorder.GetRecord(id); ok {
			return record
		}
	}
	h.t.Fatalf("no recorder holds record %s", id)
	return routerreplay.RoutingRecord{}
}

func headerAmount(t *testing.T, response *routing.Response, name string) float64 {
	t.Helper()
	value, err := strconv.ParseFloat(response.Header.Get(name), 64)
	require.NoError(t, err, "%s is %q", name, response.Header.Get(name))
	return value
}

// A Looper answer is charged what its record and run price its calls at,
// each call once, whether the calls run one after another or at once, and
// reports the budget and that cost.
func TestALooperRunSpendsWhatItsRecordPrices(t *testing.T) {
	h := newSpendRoute(t, "    budgets: {currency: USD, run: {limit: 1}}")
	spent := 0.0
	for _, text := range []string{"LOOPER question", "PANEL question", "LOOPER again", "PANEL again"} {
		response := h.chat(text, "loop")
		require.Equal(t, http.StatusOK, response.Status, string(response.Body))
		cost := headerAmount(t, response, headers.VSRCost)
		require.Equal(t, "USD", response.Header.Get(headers.VSRCostCurrency))
		record := h.record(response)
		require.NotNil(t, record.CostAccounting)
		require.Len(t, record.CostAccounting.RouterCalls, 1)
		looper := record.CostAccounting.RouterCalls[0]
		require.GreaterOrEqual(t, looper.Calls, 2, "escalation and the panel call more than one model")
		require.Zero(t, looper.UnpricedCalls)
		require.NotNil(t, looper.Cost)
		require.InDelta(t, *looper.Cost, cost, 1e-12, "the answer's cost is its record's")
		spent += cost

		require.Equal(t, "run", response.Header.Get(headers.VSRBudgetScope))
		require.InDelta(t, 1-spent, headerAmount(t, response, headers.VSRBudgetRemaining), 1e-6)
		budget := record.CostAccounting.Budget
		require.NotNil(t, budget, "the run's record has no budget")
		require.NotNil(t, budget.Charged)
		require.InDelta(t, cost, *budget.Charged, 1e-9, "the record charges its calls at their price")
	}
	limits := h.router.spendLimits(&RequestContext{Headers: map[string]string{
		"x-claude-code-session-id": t.Name() + "/loop", headers.AuthzUserID: "u-loop",
	}})
	status, _ := spendbudget.Process().Tightest(limits)
	require.InDelta(t, 1-spent, status.Remaining, 1e-9, "the budget charged other than the run's calls cost")
}

// A Looper hop is priced at what its budget charge settled, so a tier the
// Chat projection of its usage has no field for, such as one-hour cache
// writes, counts in the answer's cost and its record as in the budget.
func TestALooperHopCostsWhatItsBudgetChargeSettled(t *testing.T) {
	h := newSpendRoute(t, "    budgets: {currency: USD, run: {limit: 1}}")
	response := h.chat("TIERED question", "tier")
	require.Equal(t, http.StatusOK, response.Status, string(response.Body))
	// Three hops of 10000 one-hour cache writes at 6 per million.
	require.InDelta(t, 0.18, headerAmount(t, response, headers.VSRCost), 1e-9)
	record := h.record(response)
	require.Len(t, record.CostAccounting.RouterCalls, 1)
	looper := record.CostAccounting.RouterCalls[0]
	require.Equal(t, 3, looper.Calls)
	require.NotNil(t, looper.Cost)
	require.InDelta(t, 0.18, *looper.Cost, 1e-9)
	require.NotNil(t, record.CostAccounting.Budget.Charged)
	require.InDelta(t, 0.18, *record.CostAccounting.Budget.Charged, 1e-9)
	require.InDelta(t, 0.82, headerAmount(t, response, headers.VSRBudgetRemaining), 1e-6)
}

// A Looper run a budget stops answers as a refused call does alone: 429
// budget_exceeded with the budget's headers, a record of the refusal under
// the model the client asked for, and no retry.
func TestALooperRunABudgetStopsAnswersTheRefusal(t *testing.T) {
	h := newSpendRoute(t, "    budgets: {currency: USD, run: {limit: 0.0001}}")
	response := h.chat("LOOPER question", "tiny")
	require.Equal(t, http.StatusTooManyRequests, response.Status, string(response.Body))
	var body struct {
		Error struct{ Code, Type, Message string } `json:"error"`
	}
	require.NoError(t, json.Unmarshal(response.Body, &body))
	require.Equal(t, budgetExceededCode, body.Error.Code)
	require.Equal(t, "rate_limit_error", body.Error.Type)
	require.True(t, strings.HasPrefix(body.Error.Message, "out of budget: the run spend budget of 0.000100"), body.Error.Message)
	require.Equal(t, "run", response.Header.Get(headers.VSRBudgetScope))
	require.Equal(t, "0.000100", response.Header.Get(headers.VSRBudgetRemaining))
	require.Equal(t, "false", response.Header.Get(headers.XShouldRetry))
	require.Empty(t, response.Header.Get("retry-after"), "a run budget has no window to wait for")

	record := h.record(response)
	require.Equal(t, budgetExceededCode, record.TerminalReason)
	require.Equal(t, "vllm-sr/auto", record.OriginalModel)
	require.NotNil(t, record.CostAccounting)
	require.NotNil(t, record.CostAccounting.Budget)
	require.Equal(t, "run", record.CostAccounting.Budget.Scope)
	require.Positive(t, record.CostAccounting.Budget.Estimate, "the refused call's estimate")
}

// A Looper call a budget refused reads as that refusal in the run's attempt
// trace, not as an upstream error, and is no call in its record's cost: an
// escalation that keeps its first answer when the second model's call is
// refused records one priced call.
func TestARefusedLooperCallReadsAsTheRefusal(t *testing.T) {
	h := newSpendRoute(t, "    budgets: {currency: USD, run: {limit: 0.001}}")
	response := h.chat("LOOPER question", "refused-hop")
	require.Equal(t, http.StatusOK, response.Status, string(response.Body))
	record := h.record(response)
	require.NotNil(t, record.RouteDiagnostics)
	require.NotNil(t, record.RouteDiagnostics.Looper)
	reasons := map[string]string{}
	for _, attempt := range record.RouteDiagnostics.Looper.Attempts {
		reasons[attempt.Model] = attempt.Reason
	}
	require.Equal(t, "budget_exceeded", reasons["big-model"], "attempts %+v", record.RouteDiagnostics.Looper.Attempts)
	require.NotEqual(t, "budget_exceeded", reasons["small-model"])
	require.NotNil(t, record.CostAccounting)
	require.Len(t, record.CostAccounting.RouterCalls, 1)
	looper := record.CostAccounting.RouterCalls[0]
	require.Equal(t, 1, looper.Calls, "the refused call is no call")
	require.Zero(t, looper.UnpricedCalls, "the refused call is not an unpriced one")
	require.NotNil(t, looper.Cost)
	require.InDelta(t, *looper.Cost, headerAmount(t, response, headers.VSRCost), 1e-12)
}

// The Router's own answers report the budget too, after the failed calls'
// reservations end: a chain whose candidates all failed, an answer it could
// not read, and an upstream error it passes on.
func TestRouterAnswersReportTheBudgetAfterFailedCalls(t *testing.T) {
	h := newSpendRoute(t, "    budgets: {currency: USD, run: {limit: 1}}")

	exhausted := h.chat("FAILOVER question", "chain")
	require.Equal(t, http.StatusServiceUnavailable, exhausted.Status, string(exhausted.Body))
	require.Equal(t, "run", exhausted.Header.Get(headers.VSRBudgetScope))
	require.Equal(t, "1.000000", exhausted.Header.Get(headers.VSRBudgetRemaining), "no failed call is charged")

	failed := h.chat("FAILONLY question", "error")
	require.Equal(t, http.StatusServiceUnavailable, failed.Status, string(failed.Body))
	require.Equal(t, "1.000000", failed.Header.Get(headers.VSRBudgetRemaining), "the failed call's reservation still counts")

	broken := h.chat("BROKEN question", "broken")
	require.Equal(t, http.StatusBadGateway, broken.Status, string(broken.Body))
	require.Equal(t, "run", broken.Header.Get(headers.VSRBudgetScope))
	remaining := headerAmount(t, broken, headers.VSRBudgetRemaining)
	require.Less(t, remaining, 1.0, "an answered call is charged its estimate")
	record := h.record(broken)
	require.NotNil(t, record.CostAccounting)
	require.NotNil(t, record.CostAccounting.Budget)
	require.InDelta(t, remaining, record.CostAccounting.Budget.Remaining, 1e-6)
}

// A refusal after routing records the model the client asked for, not the
// one routing selected.
func TestARefusalAfterRoutingRecordsTheRequestedModel(t *testing.T) {
	h := newSpendRoute(t, "    budgets: {currency: USD, run: {limit: 0.0001}}")
	response := h.chat("plain question", "after")
	require.Equal(t, http.StatusTooManyRequests, response.Status, string(response.Body))
	record := h.record(response)
	require.Equal(t, budgetExceededCode, record.TerminalReason)
	require.Equal(t, "vllm-sr/auto", record.OriginalModel)
	require.Equal(t, "main-model", record.SelectedModel)
}

// An out-of-budget refusal tells each client protocol not to retry, with
// Retry-After at the window's start, and the Responses API in the form Codex
// ends a turn on; a refusal only calls in flight cause asks for a short
// retry.
func TestBudgetRefusalsTellClientsWhetherToRetry(t *testing.T) {
	h := newSpendRoute(t, `    budgets:
      currency: USD
      users: [{name: daily, match: {user: "*"}, limit: 0.0001, window: day}]`)
	user := map[string]string{headers.AuthzUserID: "u-" + t.Name()}
	_, reset := spendbudget.WindowDay.Bounds(time.Now())
	wait := time.Until(reset).Seconds()

	chat := h.send("/v1/chat/completions", `{"model":"vllm-sr/auto","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`, user)
	responses := h.send("/v1/responses", `{"model":"vllm-sr/auto","max_output_tokens":100,"input":"hi","store":false}`, user)
	messages := h.send("/v1/messages", `{"model":"vllm-sr/auto","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`, user)
	for name, response := range map[string]*routing.Response{"chat": chat, "responses": responses, "messages": messages} {
		require.Equal(t, http.StatusTooManyRequests, response.Status, "%s: %s", name, response.Body)
		require.Equal(t, "false", response.Header.Get(headers.XShouldRetry), name)
		retryAfter := headerAmount(t, response, "retry-after")
		require.LessOrEqual(t, math.Abs(retryAfter-wait), 2.0, "%s: Retry-After %v, want the %v s to the window's start", name, retryAfter, wait)
		require.Contains(t, string(response.Body), "out of budget until "+reset.Format(time.RFC3339), name)
	}
	var codex struct {
		Error struct {
			Type     string `json:"type"`
			Code     string `json:"code"`
			ResetsAt int64  `json:"resets_at"`
		} `json:"error"`
	}
	require.NoError(t, json.Unmarshal(responses.Body, &codex))
	require.Equal(t, "usage_limit_reached", codex.Error.Type)
	require.Equal(t, budgetExceededCode, codex.Error.Code)
	require.Equal(t, reset.Unix(), codex.Error.ResetsAt)
	require.True(t, strings.HasPrefix(responses.Header.Get(headers.XCodexPromoMessage), "Out of budget until "))
	require.Contains(t, string(chat.Body), `"type":"rate_limit_error"`)
	require.Contains(t, string(messages.Body), `"type":"rate_limit_error"`)
	require.Equal(t, budgetExceededCode, messages.Header.Get(headers.AnthropicOverageDisabledReason),
		"Claude Code on a login shows a Messages refusal's message as it is only with this header")
	require.Empty(t, chat.Header.Get(headers.AnthropicOverageDisabledReason))
	require.Empty(t, responses.Header.Get(headers.AnthropicOverageDisabledReason))

	// A user whose spend leaves room, but whose calls in flight hold it.
	held := map[string]string{headers.AuthzUserID: "u-held-" + t.Name()}
	limits := h.router.spendLimits(&RequestContext{Headers: held})
	hold, refusal := spendbudget.Process().Reserve(limits, 0.0001)
	require.Nil(t, refusal)
	defer hold.Release()
	busy := h.send("/v1/chat/completions", `{"model":"vllm-sr/auto","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`, held)
	require.Equal(t, http.StatusTooManyRequests, busy.Status, string(busy.Body))
	require.Equal(t, "true", busy.Header.Get(headers.XShouldRetry))
	require.Equal(t, strconv.Itoa(int(inFlightRetryAfter.Seconds())), busy.Header.Get("retry-after"))
	require.Contains(t, string(busy.Body), "has 0.000000 left (0.000000 spent, 0.000100 held by calls in flight)")
	require.Contains(t, string(busy.Body), "retry once they finish")
	busyMessages := h.send("/v1/messages", `{"model":"vllm-sr/auto","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`, held)
	require.Equal(t, http.StatusTooManyRequests, busyMessages.Status, string(busyMessages.Body))
	require.Empty(t, busyMessages.Header.Get(headers.AnthropicOverageDisabledReason), "a refusal a retry may clear is no spend limit")
}

// An out-of-budget refusal while another call of the run is in flight says
// what the run spent and what that call holds apart from what is left, on
// every client protocol, so the limit less what is left is not read as the
// spend.
func TestAnOutOfBudgetRefusalSaysWhatCallsInFlightHold(t *testing.T) {
	h := newSpendRoute(t, "    budgets: {currency: USD, run: {limit: 0.005}}")
	session := map[string]string{"x-claude-code-session-id": t.Name(), headers.AuthzUserID: "u-" + t.Name()}
	first := h.send("/v1/chat/completions", `{"model":"vllm-sr/auto","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`, session)
	require.Equal(t, http.StatusOK, first.Status, string(first.Body))
	spent := headerAmount(t, first, headers.VSRCost)
	hold, refusal := spendbudget.Process().Reserve(h.router.spendLimits(&RequestContext{Headers: session}), 0.001)
	require.Nil(t, refusal)
	defer hold.Release()

	left := fmt.Sprintf("%.6f", 0.005-spent-0.001)
	want := fmt.Sprintf("out of budget: the run spend budget of 0.005000 has %s left (%.6f spent, 0.001000 held by calls in flight), and this call is estimated at ", left, spent)
	chat := h.send("/v1/chat/completions", `{"model":"vllm-sr/auto","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`, session)
	responses := h.send("/v1/responses", `{"model":"vllm-sr/auto","max_output_tokens":100,"input":"hi","store":false}`, session)
	messages := h.send("/v1/messages", `{"model":"vllm-sr/auto","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`, session)
	for name, response := range map[string]*routing.Response{"chat": chat, "responses": responses, "messages": messages} {
		require.Equal(t, http.StatusTooManyRequests, response.Status, "%s: %s", name, response.Body)
		require.Equal(t, "false", response.Header.Get(headers.XShouldRetry), name)
		require.Equal(t, left, response.Header.Get(headers.VSRBudgetRemaining), "%s: what is left is what the message says", name)
		require.Contains(t, string(response.Body), want, name)
	}
	require.True(t, strings.HasPrefix(responses.Header.Get(headers.XCodexPromoMessage), "O"+want[1:]), responses.Header.Get(headers.XCodexPromoMessage))
	require.Contains(t, string(responses.Body), `"type":"usage_limit_reached"`)
	require.Equal(t, budgetExceededCode, messages.Header.Get(headers.AnthropicOverageDisabledReason))
}
