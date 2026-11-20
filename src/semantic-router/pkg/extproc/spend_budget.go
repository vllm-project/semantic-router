package extproc

import (
	"context"
	"fmt"
	"math"
	"net/http"
	"strconv"
	"strings"
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

// requestSpend is a request's place in the spend budgets: the budgets it
// belongs to, the reservation of its call in flight, and what it spent.
type requestSpend struct {
	limits   []spendbudget.Limit
	currency string
	hold     *spendbudget.Hold
	estimate float64
	charged  float64
	// metered counts the Router model calls already charged.
	metered  int
	unpriced int
	// report is what Replay records, once a call reserved, settled or was
	// refused.
	report *store.BudgetAccounting
}

// budgetRefusal is a call a spend budget does not admit.
type budgetRefusal struct {
	code     string
	message  string
	status   spendbudget.Status
	estimate float64
}

// requestSpend returns the request's spend state, nil when no budget
// applies. A client request marks its context with its budgets, so the
// Looper hops it makes in process spend the same ones.
func (r *OpenAIRouter) requestSpend(ctx *RequestContext) *requestSpend {
	if ctx == nil || r == nil || r.Config == nil || !r.Config.Budgets.Enabled() {
		return nil
	}
	if ctx.spend != nil {
		return ctx.spend
	}
	limits, inherited := spendbudget.LimitsFrom(ctx.TraceContext)
	if !inherited || !ctx.LooperRequest {
		limits = r.spendLimits(ctx)
		parent := ctx.TraceContext
		if parent == nil {
			parent = context.Background()
		}
		ctx.TraceContext = spendbudget.WithLimits(parent, limits)
	}
	// Keep even an empty scope on the context: Looper hops must inherit it
	// instead of resolving a new scope from their own identity.
	if len(limits) == 0 {
		return nil
	}
	ctx.spend = &requestSpend{limits: limits, currency: r.Config.Budgets.EffectiveCurrency()}
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
		spend.unpriced++
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
	return estimateDispatchCost(llmprotocol.EstimateInput(request).Tokens, int64(output), pricing), true
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

// finishRequestSpend ends the request's part in the budgets: a call still
// reserved is charged the usage a stream cut short reported, when it was
// priced, else its estimate when a provider answered it, since its usage
// never arrived, and released otherwise; Router model calls made since the
// last charge are charged.
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
	if !changed {
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
	spend.charged += amount
	return true
}

func (s *requestSpend) settle(actual float64, settlement string) {
	s.hold.Settle(actual)
	s.hold = nil
	s.charged += actual
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
	s.report.Scope, s.report.Limit, s.report.Remaining = status.Scope, status.Limit, status.Remaining
	s.report.UnpricedCalls = s.unpriced
	if s.report.Settlement != "" || s.charged != 0 {
		charged := spendbudget.Round(s.charged)
		s.report.Charged = &charged
	}
}

func (s *requestSpend) refused(refusal *spendbudget.Refusal) *budgetRefusal {
	s.setReport(refusal.Status, refusal.Estimate)
	return &budgetRefusal{code: budgetExceededCode, status: refusal.Status, estimate: refusal.Estimate, message: refusal.Error()}
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
// budget_exceeded, with Retry-After until a user window resets, or 403
// budget_unpriced, which retrying cannot change. Both name the budget and
// what it has left.
func (r *OpenAIRouter) budgetRefusalResponse(ctx *RequestContext, refusal *budgetRefusal) *ext_proc.ProcessingResponse {
	status, category, path := http.StatusTooManyRequests, llmprotocol.ErrorRateLimited, headers.ResponsePathRateLimited
	if refusal.code == budgetUnpricedCode {
		status, category, path = http.StatusForbidden, llmprotocol.ErrorPermission, headers.ResponsePathError
	}
	ctx.ImmediateProtocolError = llmprotocol.NewError(category, refusal.code, refusal.message, nil)
	ctx.ResponsePath = path
	response := r.createErrorResponseOnPath(status, refusal.message, ctx.ResponsePath)
	if status == http.StatusTooManyRequests && !refusal.status.ResetAt.IsZero() {
		wait := max(int(math.Ceil(time.Until(refusal.status.ResetAt).Seconds())), 1)
		appendImmediateResponseHeader(response, "retry-after", strconv.Itoa(wait))
	}
	if refusal.status.Scope != "" {
		appendImmediateResponseHeader(response, headers.VSRBudgetRemaining, formatBudgetAmount(refusal.status.Remaining))
		appendImmediateResponseHeader(response, headers.VSRBudgetScope, refusal.status.Scope)
	}
	logging.ComponentWarnEvent("extproc", "spend_budget_refused", map[string]interface{}{
		"request_id": ctx.RequestID, "code": refusal.code, "scope": refusal.status.Scope,
		"limit": refusal.status.Limit, "remaining": refusal.status.Remaining, "estimate": refusal.estimate,
	})
	return response
}

// respondBudgetRefused answers a refused call and records the request as
// failed with the refusal's code.
func (r *OpenAIRouter) respondBudgetRefused(ctx *RequestContext, selectedModel string, refusal *budgetRefusal) *ext_proc.ProcessingResponse {
	response := r.budgetRefusalResponse(ctx, refusal)
	status := int(response.GetImmediateResponse().GetStatus().GetCode())
	response = r.recordRejection(ctx, ctx.RequestModel, selectedModel, status, response, refusal.code)
	r.refreshReplayBudget(ctx)
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
