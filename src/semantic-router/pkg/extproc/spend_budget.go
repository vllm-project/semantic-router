package extproc

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/spendbudget"
)

// The codes a spend budget refusal answers with.
const (
	budgetExceededCode = "budget_exceeded"
	budgetUnpricedCode = "budget_unpriced"
)

// sessionlessRunIdle is how long the run of a request without a session
// outlives its last call: the request is the whole run.
const sessionlessRunIdle = time.Minute

// inFlightRetryAfter is the retry a refusal only calls in flight cause asks
// for: long enough for a short call to settle, short enough to look alive.
const inFlightRetryAfter = 5 * time.Second

// requestSpend is a request's place in the spend budgets: the budgets it
// belongs to, the reservation of its call in flight, and what it spent.
type requestSpend struct {
	limits   []spendbudget.Limit
	currency string
	// model is the model the client asked for, which the record of a refusal
	// names as the request's own.
	model    string
	hold     *spendbudget.Hold
	estimate float64
	charged  float64
	// metered counts the Router model calls already charged.
	metered  int
	unpriced int
	// report is what Replay records, once a call reserved, settled or was
	// refused.
	report *store.BudgetAccounting
	// client is the request a Looper hop serves, which counts what the hop
	// spends.
	client *requestSpend
	// hops is what the request's Looper hops spent, written from their
	// goroutines, and the refusal that stopped one.
	hops struct {
		sync.Mutex
		charged  float64
		unpriced int
		refusal  *budgetRefusal
	}
}

// budgetRefusal is a call a spend budget does not admit.
type budgetRefusal struct {
	code     string
	message  string
	status   spendbudget.Status
	estimate float64
	// inFlight is a refusal only the calls in flight cause.
	inFlight bool
}

// clientSpendKey marks a client request's context with its spend, so the
// Looper hops it makes in process spend the same budgets and count to it.
type clientSpendKey struct{}

// requestSpend returns the request's spend state, nil when no budget
// applies.
func (r *OpenAIRouter) requestSpend(ctx *RequestContext) *requestSpend {
	if ctx == nil || r == nil || r.Config == nil || !r.Config.Budgets.Enabled() {
		return nil
	}
	if ctx.spend != nil {
		return ctx.spend
	}
	parent := ctx.TraceContext
	if parent == nil {
		parent = context.Background()
	}
	if client, ok := parent.Value(clientSpendKey{}).(*requestSpend); ok && ctx.LooperRequest {
		if len(client.limits) == 0 {
			return nil
		}
		ctx.spend = &requestSpend{limits: client.limits, currency: client.currency, client: client}
		return ctx.spend
	}
	spend := &requestSpend{limits: r.spendLimits(ctx), currency: r.Config.Budgets.EffectiveCurrency(), model: ctx.RequestModel}
	// Keep even an empty scope on the context: Looper hops must inherit it
	// instead of resolving a new scope from their own identity.
	ctx.TraceContext = context.WithValue(parent, clientSpendKey{}, spend)
	if len(spend.limits) == 0 {
		return nil
	}
	ctx.spend = spend
	return ctx.spend
}

// spendLimits are the budgets a client request belongs to: its run, keyed by
// its user and the session Replay records, and each user rule it matches.
func (r *OpenAIRouter) spendLimits(ctx *RequestContext) []spendbudget.Limit {
	budgets := r.Config.Budgets
	identity := r.buildRateLimitContext(ctx, "")
	var limits []spendbudget.Limit
	if run := budgets.Run; run != nil {
		session, _ := protectionSessionID(ctx, r.Config.RouterLearning.Protection)
		idle := run.IdleTimeout()
		if session == "" {
			session, idle = fmt.Sprintf("request %s %p", ctx.RequestID, ctx), sessionlessRunIdle
		}
		limits = append(limits, spendbudget.Limit{
			Key:   strings.Join([]string{"run", identity.UserID, session}, "\x00"),
			Scope: "run", Amount: run.Limit, Idle: idle,
		})
	}
	for _, rule := range budgets.Users {
		if rule.Match.Matches(identity.UserID, identity.Groups) {
			limits = append(limits, spendbudget.Limit{
				Key:   strings.Join([]string{"user", rule.Name, identity.UserID}, "\x00"),
				Scope: "user/" + rule.Name, Amount: rule.Limit, Window: spendbudget.Window(rule.Window),
			})
		}
	}
	return limits
}

// refuseExhaustedSpend refuses, before routing, a request whose budget has
// nothing left, so a refused retry does not pay for its signals' model calls.
func (r *OpenAIRouter) refuseExhaustedSpend(ctx *RequestContext) *budgetRefusal {
	spend := r.requestSpend(ctx)
	if spend == nil {
		return nil
	}
	if refusal := spendbudget.Process().Check(spend.limits); refusal != nil {
		return spend.refused(refusal)
	}
	return nil
}

// reserveDispatchSpend reserves the estimated cost of sending request to
// model before it leaves. A reservation the request still holds, the primary
// a fallback replaces, ends first: that call failed.
func (r *OpenAIRouter) reserveDispatchSpend(ctx *RequestContext, model string, request *llmprotocol.Request) *budgetRefusal {
	spend := r.requestSpend(ctx)
	if spend == nil {
		return nil
	}
	if spend.hold != nil {
		spend.settle(0, store.BudgetSettledReleased)
	}
	estimate, priced := r.dispatchSpendEstimate(model, request, spend.currency)
	if !priced {
		if r.Config.Budgets.RefusesUnpriced() {
			status, _ := spendbudget.Process().Tightest(spend.limits)
			spend.setReport(status, 0)
			return &budgetRefusal{
				code: budgetUnpricedCode, status: status,
				message: fmt.Sprintf("model %q has no price in %s, and the spend budget refuses calls it cannot price", model, spend.currency),
			}
		}
		spend.add(0, 1)
	}
	hold, refusal := spendbudget.Process().Reserve(spend.limits, estimate)
	if refusal != nil {
		return spend.refused(refusal)
	}
	spend.hold, spend.estimate = hold, estimate
	status, _ := spendbudget.Process().Tightest(spend.limits)
	spend.setReport(status, estimate)
	return nil
}

// dispatchSpendEstimate bounds the cost of a call: its estimated input at the
// model's highest input rate, and its output at the request's limit, else
// the model's, else the budgets' reserve. It is false when the model has no
// price in currency.
func (r *OpenAIRouter) dispatchSpendEstimate(model string, request *llmprotocol.Request, currency string) (float64, bool) {
	pricing, ok := r.Config.GetFullModelPricing(model)
	if !ok || normalizeReplayCurrency(pricing.Currency) != currency {
		return 0, false
	}
	output := r.Config.Budgets.EffectiveOutputReserveTokens()
	limit := r.Config.ModelConfig[model].MaxOutputTokens
	if limit > 0 {
		output = limit
	}
	if request != nil && request.Sampling.MaxOutputTokens != nil && *request.Sampling.MaxOutputTokens > 0 {
		output = int(min(*request.Sampling.MaxOutputTokens, int64(math.MaxInt32)))
		if limit > 0 {
			output = min(output, limit)
		}
	}
	inputRate := max(pricing.PromptPer1M, effectiveCacheWriteRate(pricing), effectiveCacheWrite1hRate(pricing))
	input := llmprotocol.EstimateInput(request).Tokens
	return (float64(input)*inputRate + float64(output)*max(pricing.CompletionPer1M, 0)) / 1e6, true
}

// settleRequestSpend replaces the reservation with the priced usage, or with
// the estimate when the usage has no price in the budgets' currency, and
// charges the Router model calls made so far. replayUsage carries the report.
func (r *OpenAIRouter) settleRequestSpend(ctx *RequestContext, replayUsage *routerreplay.UsageCost) {
	spend := ctx.spend
	if spend == nil || spend.hold == nil {
		return
	}
	r.chargeRouterModelCalls(ctx, spend)
	if ctx.RequestCostPriced && normalizeReplayCurrency(ctx.RequestCostCurrency) == spend.currency {
		spend.settle(ctx.RequestCost, store.BudgetSettledUsage)
	} else {
		spend.settle(spend.estimate, store.BudgetSettledEstimate)
	}
	if replayUsage.CostAccounting == nil {
		replayUsage.CostAccounting = &store.CostAccounting{}
	}
	report := *spend.report
	replayUsage.CostAccounting.Budget = &report
	if replayUsage.TotalTokens == nil && replayUsage.ProviderCache == nil {
		// The usage write skips a record without usage.
		r.refreshReplayBudget(ctx)
	}
}

// settleHopSpend charges a Looper hop's answered call its priced usage, as
// a client request's call is charged, and returns that price and its
// currency, which the Looper records as the call's cost. response is the
// decoded answer, or nil to decode body.
func (r *OpenAIRouter) settleHopSpend(ctx *RequestContext, response *llmprotocol.Response, body []byte) (float64, string, bool) {
	cost, currency, priced := r.hopCost(ctx, response, body)
	spend := ctx.spend
	if spend == nil || spend.hold == nil || !priced || currency != spend.currency {
		// The hop's end charges the estimate.
		return cost, currency, priced
	}
	spend.settle(cost, store.BudgetSettledUsage)
	r.refreshReplayBudget(ctx)
	return cost, currency, priced
}

// hopCost prices a Looper hop's answer from its full usage, with the tiers,
// such as one-hour cache writes, that its Chat projection has no field for.
func (r *OpenAIRouter) hopCost(ctx *RequestContext, response *llmprotocol.Response, body []byte) (float64, string, bool) {
	if r.Config == nil {
		return 0, "", false
	}
	if response == nil {
		engine, err := r.protocolEngineForBackend(ctx)
		if err != nil {
			return 0, "", false
		}
		source, _ := responseWireFormats(ctx)
		decoded, _, _, err := engine.DecodeResponse(source, body)
		if err != nil {
			return 0, "", false
		}
		response = &decoded
	}
	usage := responseUsageFromSemanticUsage(response.Usage)
	pricing, priced := r.Config.GetFullModelPricing(ctx.RequestModel)
	if usage.invalid || !priced || !responseUsageHasPricableBreakdown(usage) {
		return 0, "", false
	}
	return costForResponseUsage(usage, pricing), normalizeReplayCurrency(pricing.Currency), true
}

// releaseFailedSpend ends the reservation of a call no provider answered as
// the reply that says so is built, so the reply reports the budget without
// it.
func (r *OpenAIRouter) releaseFailedSpend(ctx *RequestContext) {
	if ctx == nil || ctx.spend == nil || ctx.spend.hold == nil {
		return
	}
	ctx.spend.settle(0, store.BudgetSettledReleased)
	r.refreshReplayBudget(ctx)
}

// finishRequestSpend ends the request's part in the budgets: a call still
// reserved is charged the usage a stream cut short reported, when it was
// priced, else its estimate when a provider answered it, since its usage
// never arrived, and released otherwise; Router model calls made since the
// last charge are charged. A request whose Looper hops spent is reported
// once, with what they spent.
func (r *OpenAIRouter) finishRequestSpend(ctx *RequestContext) {
	if ctx == nil || ctx.spend == nil {
		return
	}
	spend := ctx.spend
	changed := r.chargeRouterModelCalls(ctx, spend)
	if spend.hold != nil {
		switch {
		case ctx.RequestCostPriced && normalizeReplayCurrency(ctx.RequestCostCurrency) == spend.currency:
			spend.settle(ctx.RequestCost, store.BudgetSettledPartialUsage)
		case ctx.UpstreamStatusCode >= http.StatusOK && ctx.UpstreamStatusCode < http.StatusMultipleChoices:
			spend.settle(spend.estimate, store.BudgetSettledEstimate)
		default:
			spend.settle(0, store.BudgetSettledReleased)
		}
		changed = true
	}
	if charged, unpriced := spend.totals(); !changed && (spend.report != nil || charged == 0 && unpriced == 0) {
		return
	}
	status, _ := spendbudget.Process().Tightest(spend.limits)
	if spend.report == nil {
		spend.setReport(status, 0)
	} else {
		spend.updateReport(status)
	}
	r.refreshReplayBudget(ctx)
}

// chargeRouterModelCalls charges the model calls the Router made for the
// request since the last charge, at their deployments' prices. A call with
// no price in the budgets' currency spends nothing and is counted.
func (r *OpenAIRouter) chargeRouterModelCalls(ctx *RequestContext, spend *requestSpend) bool {
	calls := ctx.ModelCallMeter.Calls()
	if len(calls) <= spend.metered {
		return false
	}
	amount := 0.0
	for _, measured := range calls[spend.metered:] {
		call := r.priceModelCall(measured)
		if call.Cost != nil && call.Currency == spend.currency {
			amount += *call.Cost
		} else {
			spend.unpriced++
		}
	}
	spend.metered = len(calls)
	spendbudget.Process().Charge(spend.limits, amount)
	spend.add(amount, 0)
	return true
}

// add counts what the request spent, and its calls with no price, to it and
// to the client request a hop serves.
func (s *requestSpend) add(amount float64, unpriced int) {
	s.charged += amount
	s.unpriced += unpriced
	if client := s.client; client != nil {
		client.hops.Lock()
		client.hops.charged += amount
		client.hops.unpriced += unpriced
		client.hops.Unlock()
	}
}

// totals are what the request and its Looper hops spent, and their calls
// with no price.
func (s *requestSpend) totals() (float64, int) {
	s.hops.Lock()
	defer s.hops.Unlock()
	return s.charged + s.hops.charged, s.unpriced + s.hops.unpriced
}

// noteHopRefusal keeps the first refusal that stopped one of the request's
// Looper hops, which the run answers with when it fails.
func (s *requestSpend) noteHopRefusal(refusal *budgetRefusal) {
	s.hops.Lock()
	defer s.hops.Unlock()
	if s.hops.refusal == nil {
		s.hops.refusal = refusal
	}
}

// hopRefusal is the refusal that stopped one of the request's Looper hops.
func (s *requestSpend) hopRefusal() *budgetRefusal {
	if s == nil {
		return nil
	}
	s.hops.Lock()
	defer s.hops.Unlock()
	return s.hops.refusal
}

// recheck reads a Looper hop's refusal again once the run's other calls are
// done: a call that only calls in flight kept out and that what they settled
// leaves no room for is out of budget, and one that fits now keeps asking
// for a retry.
func (s *requestSpend) recheck(refusal *budgetRefusal) *budgetRefusal {
	if !refusal.inFlight {
		return refusal
	}
	if again := spendbudget.Process().Recheck(s.limits, refusal.estimate); again != nil {
		return s.refused(again)
	}
	return refusal
}

func (s *requestSpend) settle(actual float64, settlement string) {
	s.hold.Settle(actual)
	s.hold = nil
	s.add(actual, 0)
	status, _ := spendbudget.Process().Tightest(s.limits)
	if s.report == nil {
		s.setReport(status, s.estimate)
	}
	s.report.Settlement = settlement
	s.updateReport(status)
}

func (s *requestSpend) setReport(status spendbudget.Status, estimate float64) {
	s.report = &store.BudgetAccounting{Currency: s.currency, Estimate: spendbudget.Round(estimate)}
	s.updateReport(status)
}

func (s *requestSpend) updateReport(status spendbudget.Status) {
	charged, unpriced := s.totals()
	s.report.Scope, s.report.Limit, s.report.Remaining = status.Scope, status.Limit, status.Remaining
	s.report.UnpricedCalls = unpriced
	if s.report.Settlement != "" || charged != 0 {
		charged = spendbudget.Round(charged)
		s.report.Charged = &charged
	}
}

func (s *requestSpend) refused(refusal *spendbudget.Refusal) *budgetRefusal {
	s.setReport(refusal.Status, refusal.Estimate)
	return &budgetRefusal{
		code: budgetExceededCode, status: refusal.Status, estimate: refusal.Estimate,
		inFlight: refusal.InFlight, message: refusal.Error(),
	}
}

// refreshReplayBudget writes the request's budget report into its record.
func (r *OpenAIRouter) refreshReplayBudget(ctx *RequestContext) {
	if ctx.spend == nil || ctx.spend.report == nil || ctx.RouterReplayID == "" {
		return
	}
	recorder := ctx.RouterReplayRecorder
	if recorder == nil {
		recorder = r.ReplayRecorder
	}
	if recorder == nil {
		return
	}
	record, ok := recorder.GetRecord(ctx.RouterReplayID)
	if !ok {
		return
	}
	usage := recordUsageCost(record)
	accounting := &store.CostAccounting{}
	if usage.CostAccounting != nil {
		copied := *usage.CostAccounting
		accounting = &copied
	}
	report := *ctx.spend.report
	accounting.Budget = &report
	usage.CostAccounting = accounting
	r.writeRouterReplayUsageCost(ctx, usage)
}

// budgetRefusalResponse answers a call a budget refused: 429
// budget_exceeded, or 403 budget_unpriced, which retrying cannot change.
// Both name the budget and what it has left. A refusal that no retry before
// the budget's window starts again can clear tells the client not to retry,
// with Retry-After at that start; one only the calls in flight cause asks
// for a retry in a few seconds.
func (r *OpenAIRouter) budgetRefusalResponse(ctx *RequestContext, refusal *budgetRefusal) *ext_proc.ProcessingResponse {
	status, category, path := http.StatusTooManyRequests, llmprotocol.ErrorRateLimited, headers.ResponsePathRateLimited
	if refusal.code == budgetUnpricedCode {
		status, category, path = http.StatusForbidden, llmprotocol.ErrorPermission, headers.ResponsePathError
	}
	ctx.ImmediateProtocolError = llmprotocol.NewError(category, refusal.code, refusal.message, nil)
	ctx.ResponsePath = path
	response := r.createErrorResponseOnPath(status, refusal.message, ctx.ResponsePath)
	appendBudgetRetryHeaders(response, refusal)
	if refusal.status.Scope != "" {
		appendImmediateResponseHeader(response, headers.VSRBudgetRemaining, formatBudgetAmount(refusal.status.Remaining))
		appendImmediateResponseHeader(response, headers.VSRBudgetScope, refusal.status.Scope)
	}
	if refusal.code == budgetExceededCode && !refusal.inFlight {
		switch ctx.SourceFormat {
		case llmprotocol.OpenAIResponsesV1:
			response = r.codexUsageLimit(ctx, response, refusal)
		case llmprotocol.AnthropicMessagesV1:
			// Claude Code 2.1.325 on a Claude or gateway login calls any
			// other 429 "temporarily limiting requests (not your usage
			// limit)"; with this header it shows the message alone.
			appendImmediateResponseHeader(response, headers.AnthropicOverageDisabledReason, budgetExceededCode)
		}
	}
	logging.ComponentWarnEvent("extproc", "spend_budget_refused", map[string]interface{}{
		"request_id": ctx.RequestID, "code": refusal.code, "scope": refusal.status.Scope, "in_flight": refusal.inFlight,
		"limit": refusal.status.Limit, "remaining": refusal.status.Remaining, "estimate": refusal.estimate,
		"spent": refusal.status.Spent, "reserved": refusal.status.Reserved,
	})
	return response
}

func appendBudgetRetryHeaders(response *ext_proc.ProcessingResponse, refusal *budgetRefusal) {
	if refusal.inFlight {
		appendImmediateResponseHeader(response, "retry-after", strconv.Itoa(int(inFlightRetryAfter.Seconds())))
		appendImmediateResponseHeader(response, headers.XShouldRetry, "true")
		return
	}
	if refusal.code == budgetExceededCode && !refusal.status.ResetAt.IsZero() {
		wait := max(int(math.Ceil(time.Until(refusal.status.ResetAt).Seconds())), 1)
		appendImmediateResponseHeader(response, "retry-after", strconv.Itoa(wait))
	}
	appendImmediateResponseHeader(response, headers.XShouldRetry, "false")
}

// codexUsageLimit makes an out-of-budget refusal on the Responses API the
// error type usage_limit_reached, the 429 Codex 0.166.0 ends a turn on and
// shows a message with: any other it retries after Retry-After, hours for a
// daily budget, or ends with a fixed text that omits the message. resets_at
// is the window's start, and x-codex-promo-message carries the message.
func (r *OpenAIRouter) codexUsageLimit(ctx *RequestContext, response *ext_proc.ProcessingResponse, refusal *budgetRefusal) *ext_proc.ProcessingResponse {
	response = r.encodeImmediateResponseForClient(response, ctx)
	immediate := response.GetImmediateResponse()
	var body struct {
		Error map[string]any `json:"error"`
	}
	if json.Unmarshal(immediate.GetBody(), &body) != nil || body.Error == nil {
		return response
	}
	body.Error["type"] = "usage_limit_reached"
	if !refusal.status.ResetAt.IsZero() {
		body.Error["resets_at"] = refusal.status.ResetAt.Unix()
	}
	if encoded, err := json.Marshal(body); err == nil {
		immediate.Body = encoded
	}
	appendImmediateResponseHeader(response, headers.XCodexPromoMessage, strings.ToUpper(refusal.message[:1])+refusal.message[1:])
	return response
}

// respondBudgetRefused answers a refused call and records the request as
// failed with the refusal's code, under the model its client asked for. A
// hop's refusal is kept for the run it serves.
func (r *OpenAIRouter) respondBudgetRefused(ctx *RequestContext, selectedModel string, refusal *budgetRefusal) *ext_proc.ProcessingResponse {
	response := r.budgetRefusalResponse(ctx, refusal)
	status := int(response.GetImmediateResponse().GetStatus().GetCode())
	response = r.recordRejection(ctx, cmp.Or(ctx.spend.model, ctx.RequestModel), selectedModel, status, response, refusal.code)
	r.refreshReplayBudget(ctx)
	if client := ctx.spend.client; client != nil {
		client.noteHopRefusal(refusal)
	}
	return response
}

// addBudget reports the tightest budget of the request as its response
// headers leave.
func (builder *responseHeaderMutationBuilder) addBudget(ctx *RequestContext) {
	if status, ok := spendStatus(ctx); ok {
		builder.addString(headers.VSRBudgetRemaining, formatBudgetAmount(status.Remaining))
		builder.addString(headers.VSRBudgetScope, status.Scope)
	}
}

// addResponseBudgetHeaders rewrites them on a buffered response once its
// call has settled.
func addResponseBudgetHeaders(ctx *RequestContext, response *ext_proc.ProcessingResponse) {
	if status, ok := spendStatus(ctx); ok {
		setResponseBodyHeaderOverwrite(response, headers.VSRBudgetRemaining, formatBudgetAmount(status.Remaining))
		setResponseBodyHeaderOverwrite(response, headers.VSRBudgetScope, status.Scope)
	}
}

func spendStatus(ctx *RequestContext) (spendbudget.Status, bool) {
	if ctx == nil || ctx.spend == nil {
		return spendbudget.Status{}, false
	}
	return spendbudget.Process().Tightest(ctx.spend.limits)
}

func formatBudgetAmount(amount float64) string {
	return strconv.FormatFloat(amount, 'f', 6, 64)
}
