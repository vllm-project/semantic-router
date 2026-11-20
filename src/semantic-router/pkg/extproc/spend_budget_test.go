package extproc

import (
	"context"
	"encoding/json"
	"net/http"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/spendbudget"
)

// The priced model costs 1 per million input tokens and 1000 per million
// output tokens, so a call that may write 1000 tokens reserves at least 1.
func spendTestRouter(budgets config.BudgetsConfig) *OpenAIRouter {
	cfg := &config.RouterConfig{Budgets: budgets}
	cfg.ModelConfig = map[string]config.ModelParams{
		"priced":   {Pricing: config.ModelPricing{Currency: "USD", PromptPer1M: 1, CompletionPer1M: 1000}, MaxOutputTokens: 1000},
		"unpriced": {MaxOutputTokens: 1000},
	}
	return &OpenAIRouter{Config: cfg}
}

func spendTestContext(t *testing.T, user, session, agent string) *RequestContext {
	headers := map[string]string{headers.AuthzUserID: user}
	if session != "" {
		headers["x-claude-code-session-id"] = t.Name() + "/" + session
	}
	if agent != "" {
		headers["x-claude-code-agent-id"] = agent
	}
	return &RequestContext{Headers: headers, TraceContext: context.Background(), RequestID: t.Name() + "/" + user + "/" + agent}
}

func spendTestRequest(maxTokens int64) *llmprotocol.Request {
	request := &llmprotocol.Request{
		Messages: []llmprotocol.Message{{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "hello"}}}},
	}
	request.Sampling.MaxOutputTokens = llmprotocol.Int64(maxTokens)
	return request
}

// Subagents of one session share its run budget, and their reservations are
// taken before any of them settles, so ten at once cannot overspend it.
func TestConcurrentSubagentsCannotOverspendTheirRun(t *testing.T) {
	router := spendTestRouter(config.BudgetsConfig{Run: &config.RunBudget{Limit: 3.5}})
	var admitted, refused atomic.Int64
	var group sync.WaitGroup
	for i := range 10 {
		group.Go(func() {
			ctx := spendTestContext(t, "alice", "s1", "agent-"+strconv.Itoa(i))
			if refusal := router.reserveDispatchSpend(ctx, "priced", spendTestRequest(1000)); refusal == nil {
				admitted.Add(1)
			} else if refusal.code == budgetExceededCode {
				refused.Add(1)
			}
		})
	}
	group.Wait()
	if admitted.Load() != 3 || refused.Load() != 7 {
		t.Fatalf("admitted %d and refused %d calls reserving about 1 each from a run of 3.5", admitted.Load(), refused.Load())
	}
	other := spendTestContext(t, "bob", "s1", "")
	if refusal := router.reserveDispatchSpend(other, "priced", spendTestRequest(1000)); refusal != nil {
		t.Fatalf("another user's session of the same id shares alice's run: %v", refusal.message)
	}
}

// The estimate bounds output by the request's limit, else the model's, and
// the actual cost replaces it when the call settles.
func TestDispatchReservesTheEstimateAndSettlesTheActualCost(t *testing.T) {
	router := spendTestRouter(config.BudgetsConfig{Run: &config.RunBudget{Limit: 2}})
	ctx := spendTestContext(t, "alice", "s1", "")
	if refusal := router.reserveDispatchSpend(ctx, "priced", spendTestRequest(5000)); refusal != nil {
		t.Fatal(refusal.message)
	}
	if estimate := ctx.spend.hold.Estimate(); estimate < 1 || estimate > 1.001 {
		t.Fatalf("estimate %v, want about 1: 1000 output tokens, the model's limit under the request's 5000", estimate)
	}
	ctx.RequestCost, ctx.RequestCostCurrency, ctx.RequestCostPriced = 0.25, "USD", true
	var usage routerreplay.UsageCost
	router.settleRequestSpend(ctx, &usage)
	report := usage.CostAccounting.Budget
	if report == nil || report.Scope != "run" || report.Remaining != 1.75 || report.Settlement != store.BudgetSettledUsage ||
		report.Charged == nil || *report.Charged != 0.25 {
		t.Fatalf("budget report %+v", report)
	}
	if refusal := router.reserveDispatchSpend(spendTestContext(t, "alice", "s1", ""), "priced", spendTestRequest(1000)); refusal != nil {
		t.Fatalf("a call estimated at about 1 does not fit the 1.75 left: %s", refusal.message)
	}
	if refusal := router.reserveDispatchSpend(spendTestContext(t, "alice", "s1", ""), "priced", spendTestRequest(1000)); refusal == nil {
		t.Fatal("a second call estimated at about 1 fits the 0.75 left")
	}
}

// A call answered without usage is charged its estimate; a call no provider
// answered is charged nothing.
func TestRequestEndChargesAnAnsweredCallItsEstimateAndReleasesTheRest(t *testing.T) {
	router := spendTestRouter(config.BudgetsConfig{Run: &config.RunBudget{Limit: 1.5}})
	failed := spendTestContext(t, "alice", "s1", "")
	if refusal := router.reserveDispatchSpend(failed, "priced", spendTestRequest(1000)); refusal != nil {
		t.Fatal(refusal.message)
	}
	failed.UpstreamStatusCode = http.StatusBadGateway
	router.finishRequestSpend(failed)
	if report := failed.spend.report; report.Settlement != store.BudgetSettledReleased || *report.Charged != 0 {
		t.Fatalf("a failed call was charged: %+v", report)
	}
	answered := spendTestContext(t, "alice", "s1", "")
	if refusal := router.reserveDispatchSpend(answered, "priced", spendTestRequest(1000)); refusal != nil {
		t.Fatalf("a released reservation still counts: %s", refusal.message)
	}
	answered.UpstreamStatusCode = http.StatusOK
	router.finishRequestSpend(answered)
	router.finishRequestSpend(answered)
	if report := answered.spend.report; report.Settlement != store.BudgetSettledEstimate || *report.Charged < 1 {
		t.Fatalf("an answered call without usage was not charged its estimate: %+v", report)
	}
	if refusal := router.refuseExhaustedSpend(spendTestContext(t, "alice", "s1", "")); refusal != nil {
		t.Fatalf("a run with about 0.5 left is refused before routing: %v", refusal)
	}
	if refusal := router.reserveDispatchSpend(spendTestContext(t, "alice", "s1", ""), "priced", spendTestRequest(1000)); refusal == nil {
		t.Fatal("a second call estimated at 1 fits the 0.5 left")
	}
}

// A refusal answers 429 budget_exceeded, names the budget and carries what
// it has left; an exhausted budget is refused before routing.
func TestBudgetRefusalAnswers429AndNamesTheBudget(t *testing.T) {
	router := spendTestRouter(config.BudgetsConfig{Users: []config.UserBudget{{Name: "daily", Limit: 0.5, Window: "day"}}})
	ctx := spendTestContext(t, t.Name(), "", "")
	refusal := router.reserveDispatchSpend(ctx, "priced", spendTestRequest(1000))
	if refusal == nil {
		t.Fatal("a call estimated at 1 fits a daily budget of 0.5")
	}
	response := router.budgetRefusalResponse(ctx, refusal)
	immediate := response.GetImmediateResponse()
	if immediate.GetStatus().GetCode() != http.StatusTooManyRequests {
		t.Fatalf("status %d, want 429", immediate.GetStatus().GetCode())
	}
	headerValues := map[string]string{}
	for _, option := range immediate.GetHeaders().GetSetHeaders() {
		headerValues[option.GetHeader().GetKey()] = string(option.GetHeader().GetRawValue())
	}
	if headerValues[headers.VSRBudgetScope] != "user/daily" || headerValues[headers.VSRBudgetRemaining] != "0.500000" ||
		headerValues["retry-after"] == "" {
		t.Fatalf("headers %v", headerValues)
	}
	if ctx.ImmediateProtocolError == nil || ctx.ImmediateProtocolError.Code != budgetExceededCode {
		t.Fatalf("protocol error %+v", ctx.ImmediateProtocolError)
	}
	var body map[string]map[string]any
	if err := json.Unmarshal(immediate.GetBody(), &body); err != nil || body["error"]["message"] == "" {
		t.Fatalf("body %s: %v", immediate.GetBody(), err)
	}

	exhausted := spendTestContext(t, t.Name()+"-spent", "", "")
	spendbudget.Process().Charge(router.requestSpend(exhausted).limits, 0.5)
	if refusal := router.refuseExhaustedSpend(spendTestContext(t, t.Name()+"-spent", "", "")); refusal == nil || refusal.code != budgetExceededCode {
		t.Fatalf("an exhausted budget was not refused before routing: %+v", refusal)
	}
}

// Unpriced calls covered by a budget never pass silently: refused by
// default, or admitted at zero and counted.
func TestUnpricedCallsAreRefusedOrCounted(t *testing.T) {
	refusing := spendTestRouter(config.BudgetsConfig{Run: &config.RunBudget{Limit: 1}})
	refusal := refusing.reserveDispatchSpend(spendTestContext(t, "alice", "s1", ""), "unpriced", spendTestRequest(10))
	if refusal == nil || refusal.code != budgetUnpricedCode {
		t.Fatalf("an unpriced call under the default policy: %+v", refusal)
	}
	if status := refusing.budgetRefusalResponse(spendTestContext(t, "alice", "s1", ""), refusal).GetImmediateResponse().GetStatus().GetCode(); status != http.StatusForbidden {
		t.Fatalf("an unpriced refusal answers %d, want 403: retrying cannot price the model", status)
	}
	allowing := spendTestRouter(config.BudgetsConfig{Unpriced: config.BudgetUnpricedAllow, Run: &config.RunBudget{Limit: 1}})
	ctx := spendTestContext(t, "alice", "s2", "")
	if refusal := allowing.reserveDispatchSpend(ctx, "unpriced", spendTestRequest(10)); refusal != nil {
		t.Fatal(refusal.message)
	}
	ctx.UpstreamStatusCode = http.StatusOK
	allowing.finishRequestSpend(ctx)
	if report := ctx.spend.report; report.UnpricedCalls != 1 || *report.Charged != 0 {
		t.Fatalf("an allowed unpriced call: %+v", report)
	}
}

// A Looper hop served in process spends the budgets of the client request
// it serves, whatever its own headers say.
func TestLooperHopSpendsItsClientRequestBudgets(t *testing.T) {
	router := spendTestRouter(config.BudgetsConfig{Run: &config.RunBudget{Limit: 1.5}})
	client := spendTestContext(t, "alice", "s1", "")
	if refusal := router.refuseExhaustedSpend(client); refusal != nil {
		t.Fatal(refusal.message)
	}
	hop := &RequestContext{Headers: map[string]string{}, TraceContext: client.TraceContext, LooperRequest: true}
	if refusal := router.reserveDispatchSpend(hop, "priced", spendTestRequest(1000)); refusal != nil {
		t.Fatal(refusal.message)
	}
	if refusal := router.reserveDispatchSpend(spendTestContext(t, "alice", "s1", ""), "priced", spendTestRequest(1000)); refusal == nil {
		t.Fatal("the hop's reservation is not in alice's run")
	}
}

func TestSpendBudgetsOnlyRefuseUnpricedCallsTheyCover(t *testing.T) {
	userRule := config.UserBudget{Name: t.Name() + "-user", Match: config.BudgetMatch{User: "alice"}, Limit: 2, Window: "day"}
	groupRule := config.UserBudget{Name: t.Name() + "-group", Match: config.BudgetMatch{Group: "engineering"}, Limit: 2, Window: "day"}
	for _, tc := range []struct {
		name        string
		budgets     config.BudgetsConfig
		user        string
		groups      string
		model       string
		wantRefused bool
	}{
		{name: "disabled", user: "bob", model: "unpriced"},
		{name: "unmatched user", budgets: config.BudgetsConfig{Users: []config.UserBudget{userRule}}, user: "bob", model: "unpriced"},
		{name: "unmatched group", budgets: config.BudgetsConfig{Users: []config.UserBudget{groupRule}}, user: "bob", groups: "support", model: "unpriced"},
		{name: "unmatched priced call", budgets: config.BudgetsConfig{Users: []config.UserBudget{userRule}}, user: "bob", model: "priced"},
		{name: "matched user", budgets: config.BudgetsConfig{Users: []config.UserBudget{userRule}}, user: "alice", model: "unpriced", wantRefused: true},
		{name: "matched group", budgets: config.BudgetsConfig{Users: []config.UserBudget{groupRule}}, user: "bob", groups: "support, engineering", model: "unpriced", wantRefused: true},
		{name: "run covers unmatched user", budgets: config.BudgetsConfig{Run: &config.RunBudget{Limit: 2}, Users: []config.UserBudget{userRule}}, user: "bob", model: "unpriced", wantRefused: true},
		{name: "unscoped rule covers all users", budgets: config.BudgetsConfig{Users: []config.UserBudget{{Name: t.Name() + "-all", Limit: 2, Window: "day"}}}, user: "bob", model: "unpriced", wantRefused: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router := spendTestRouter(tc.budgets)
			ctx := spendTestContext(t, tc.user, "scope", "")
			ctx.Headers[headers.AuthzUserGroups] = tc.groups
			if refusal := router.refuseExhaustedSpend(ctx); refusal != nil {
				t.Fatal(refusal.message)
			}
			refusal := router.reserveDispatchSpend(ctx, tc.model, spendTestRequest(10))
			if tc.wantRefused {
				if refusal == nil || refusal.code != budgetUnpricedCode {
					t.Fatalf("covered unpriced call: %+v, want budget_unpriced", refusal)
				}
				if status := router.budgetRefusalResponse(ctx, refusal).GetImmediateResponse().GetStatus().GetCode(); status != http.StatusForbidden {
					t.Fatalf("status %d, want 403", status)
				}
				return
			}
			if refusal != nil {
				t.Fatalf("uncovered call refused: %s", refusal.message)
			}
			var usage routerreplay.UsageCost
			router.settleRequestSpend(ctx, &usage)
			router.finishRequestSpend(ctx)
			if usage.CostAccounting != nil && usage.CostAccounting.Budget != nil {
				t.Fatalf("uncovered call acquired budget accounting: %+v", usage.CostAccounting.Budget)
			}
			if status, ok := spendStatus(ctx); ok {
				t.Fatalf("uncovered call reports a budget: %+v", status)
			}
		})
	}
}

// Both an empty and a matched client scope survive multiple in-process hops;
// a hop's own identity must not replace its client's scope.
func TestLooperHopKeepsTheClientBudgetScope(t *testing.T) {
	for _, tc := range []struct {
		name        string
		clientUser  string
		hopUser     string
		wantRefused bool
	}{
		{name: "uncovered client and covered hop", clientUser: "bob", hopUser: "alice"},
		{name: "covered client and uncovered hop", clientUser: "alice", hopUser: "bob", wantRefused: true},
		{name: "hop without inherited scope", hopUser: "alice", wantRefused: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router := spendTestRouter(config.BudgetsConfig{Users: []config.UserBudget{{
				Name: t.Name(), Match: config.BudgetMatch{User: "alice"}, Limit: 2, Window: "day",
			}}})
			parent := context.Background()
			if tc.clientUser != "" {
				client := spendTestContext(t, tc.clientUser, "", "")
				if refusal := router.refuseExhaustedSpend(client); refusal != nil {
					t.Fatal(refusal.message)
				}
				parent = client.TraceContext
			}
			for hopIndex := range 2 {
				hop := spendTestContext(t, tc.hopUser, "", strconv.Itoa(hopIndex))
				hop.TraceContext, hop.LooperRequest = parent, true
				if refusal := router.refuseExhaustedSpend(hop); refusal != nil {
					t.Fatal(refusal.message)
				}
				refusal := router.reserveDispatchSpend(hop, "unpriced", spendTestRequest(10))
				if tc.wantRefused {
					if refusal == nil || refusal.code != budgetUnpricedCode {
						t.Fatalf("hop %d escaped its budget: %+v", hopIndex, refusal)
					}
				} else if refusal != nil {
					t.Fatalf("hop %d acquired a budget absent from its client: %s", hopIndex, refusal.message)
				}
				parent = hop.TraceContext
			}
		})
	}
}
