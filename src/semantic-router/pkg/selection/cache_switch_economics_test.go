package selection

import (
	"context"
	"math"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelpricing"
)

func cacheTestRate(v float64) *float64 { return &v }

// testCacheRates prices premium like a frontier model, cheap at a third of it,
// near at 90% of it and pricier above it, each with a cache read and write rate.
func testCacheRates() map[string]modelpricing.Rates {
	return map[string]modelpricing.Rates{
		"premium": {Currency: "USD", PromptPer1M: 3, CachedInputPer1M: 0.3, CacheWritePer1M: cacheTestRate(3.75), CompletionPer1M: 15},
		"cheap":   {Currency: "USD", PromptPer1M: 1, CachedInputPer1M: 0.1, CacheWritePer1M: cacheTestRate(1.25), CompletionPer1M: 5},
		"near":    {Currency: "USD", PromptPer1M: 2.7, CachedInputPer1M: 0.27, CacheWritePer1M: cacheTestRate(3.375), CompletionPer1M: 13.5},
		"pricier": {Currency: "USD", PromptPer1M: 5, CachedInputPer1M: 0.5, CacheWritePer1M: cacheTestRate(6.25), CompletionPer1M: 25},
		"no-cache": {
			Currency: "USD", PromptPer1M: 1, CompletionPer1M: 5,
		},
		"euro": {Currency: "EUR", PromptPer1M: 1, CachedInputPer1M: 0.1, CacheWritePer1M: cacheTestRate(1.25), CompletionPer1M: 5},
	}
}

// testCacheReceipt is a warm turn on premium: 18000 tokens read from its cache,
// 1000 written, 1000 uncached, 500 out.
func testCacheReceipt() modelpricing.Receipt {
	return modelpricing.Receipt{
		Model: "premium",
		Usage: modelpricing.Usage{
			PromptTokens:      20000,
			CachedInputTokens: 18000,
			CacheWriteTokens:  1000,
			CompletionTokens:  500,
		},
		CacheReported: true,
	}
}

func testCacheEconomicsConfig() config.CacheEconomicsConfig {
	return config.CacheEconomicsConfig{Enabled: true, CacheTTLSeconds: 300, DefaultRemainingTurns: 3}
}

func TestEvaluateCacheSwitchPricesRewriteAgainstSaving(t *testing.T) {
	for _, tc := range []struct {
		name       string
		proposed   string
		age        time.Duration
		edit       func(*modelpricing.Receipt)
		turns      float64
		turnsKnown bool
		decision   string
		reason     string
	}{
		{"a much cheaper model repays the rewrite", "cheap", time.Minute, nil, 0, false, CacheEconomicsSwitch, CacheReasonSwitchPays},
		{"a slightly cheaper model does not", "near", time.Minute, nil, 0, false, CacheEconomicsHold, CacheReasonSwitchDoesNotPay},
		{"a longer remaining run repays it", "near", time.Minute, nil, 40, true, CacheEconomicsSwitch, CacheReasonSwitchPays},
		{"a pricier model is a quality move", "pricier", time.Minute, nil, 0, false, CacheEconomicsSwitch, CacheReasonProposalNotCheaper},
		{"a cache past its lifetime has no value", "near", 301 * time.Second, nil, 0, false, CacheEconomicsSwitch, CacheReasonCacheExpired},
		{"one-hour writes keep the cache warm", "near", 10 * time.Minute, func(r *modelpricing.Receipt) { r.Usage.CacheWrite1hTokens = 1000 }, 0, false, CacheEconomicsHold, CacheReasonSwitchDoesNotPay},
		{"nothing cached means nothing to lose", "near", time.Minute, func(r *modelpricing.Receipt) {
			r.Usage.CachedInputTokens, r.Usage.CacheWriteTokens = 0, 0
		}, 0, false, CacheEconomicsSwitch, CacheReasonNoWarmCache},
		{"a receipt from another model is unknown", "cheap", time.Minute, func(r *modelpricing.Receipt) { r.Model = "near" }, 0, false, CacheEconomicsFallback, CacheReasonReceiptOtherModel},
		{"unreported cache counts are unknown", "cheap", time.Minute, func(r *modelpricing.Receipt) { r.CacheReported = false }, 0, false, CacheEconomicsFallback, CacheReasonCacheNotReported},
		{"an unpriced model is unknown", "unpriced", time.Minute, nil, 0, false, CacheEconomicsFallback, CacheReasonNoCachePrice},
		{"a model without cache prices is unknown", "no-cache", time.Minute, nil, 0, false, CacheEconomicsFallback, CacheReasonNoCachePrice},
		{"another currency is unknown", "euro", time.Minute, nil, 0, false, CacheEconomicsFallback, CacheReasonCurrencyMismatch},
	} {
		t.Run(tc.name, func(t *testing.T) {
			receipt := testCacheReceipt()
			if tc.edit != nil {
				tc.edit(&receipt)
			}
			got := EvaluateCacheSwitch(CacheSwitchInput{
				Config: testCacheEconomicsConfig(), Current: "premium", Proposed: tc.proposed,
				Receipt: &receipt, ReceiptAge: tc.age, Rates: testCacheRates(),
				RemainingTurns: tc.turns, RemainingTurnsKnown: tc.turnsKnown,
			})
			if got.Decision != tc.decision || got.Reason != tc.reason {
				t.Fatalf("decision = %s/%s, want %s/%s: %+v", got.Decision, got.Reason, tc.decision, tc.reason, got)
			}
		})
	}
}

func TestEvaluateCacheSwitchRecordsItsNumbers(t *testing.T) {
	receipt := testCacheReceipt()
	got := EvaluateCacheSwitch(CacheSwitchInput{
		Config: testCacheEconomicsConfig(), Current: "premium", Proposed: "cheap",
		Receipt: &receipt, ReceiptAge: time.Minute, Rates: testCacheRates(),
	})
	near := func(a, b float64) bool { return math.Abs(a-b) < 1e-9 }
	// premium: 1000*3 + 18000*0.3 + 1000*3.75 + 500*15; cheap: 1000*1 + 18000*0.1 + 1000*1.25 + 500*5.
	// The rewrite is the 19000-token prefix written at 1.25 instead of read at 0.1.
	if got.CachedPrefixTokens != 19000 || got.Currency != "USD" ||
		!near(got.CurrentTurnCost, 0.01965) || !near(got.ProposedTurnCost, 0.00655) ||
		!near(got.RewriteCost, 0.02185) || !near(got.Saving, 3*(0.01965-0.00655)) ||
		got.RemainingTurns != 3 || got.RemainingTurnsSource != CacheTurnsSourceDefault ||
		got.CacheTTLSeconds != 300 || got.CacheAgeSeconds != 60 {
		t.Fatalf("economics numbers = %+v", got)
	}
	if got.ToMap()["rewrite_cost"] != got.RewriteCost || got.ToMap()["decision"] != CacheEconomicsSwitch {
		t.Fatalf("economics map = %#v", got.ToMap())
	}
	if none := EvaluateCacheSwitch(CacheSwitchInput{Config: testCacheEconomicsConfig(), Current: "premium", Proposed: "cheap"}); none.Reason != CacheReasonNoReceipt {
		t.Fatalf("missing receipt = %+v", none)
	}
}

func warmCacheSession(current string) *AgenticSessionContext {
	receipt := testCacheReceipt()
	return &AgenticSessionContext{
		ID: "sess-cache", TurnIndex: 6, PreviousModel: current,
		HistoryTokens: 18000, ContextTokens: 20000, CacheWarmth: 1, CacheWarmthOK: true,
		CacheReceipt: &receipt, CacheReceiptAge: time.Minute, ModelRates: testCacheRates(),
	}
}

func selectWithCacheEconomics(t *testing.T, cfg *SessionAwareConfig, scores map[string]float64, proposed string, session *AgenticSessionContext) *SelectionResult {
	t.Helper()
	selector := NewSessionAwareSelector(cfg)
	selector.SetBaseSelector(stubSessionBase{result: &SelectionResult{
		SelectedModel: proposed, Score: scores[proposed], Method: MethodMultiFactor, AllScores: scores,
	}})
	result, err := selector.Select(context.Background(), &SelectionContext{
		CandidateModels: []config.ModelRef{{Model: "premium"}, {Model: proposed}},
		AgenticSession:  session,
	})
	if err != nil {
		t.Fatalf("Select returned error: %v", err)
	}
	return result
}

func TestSessionAwareCacheEconomicsHoldsASwitchThatDoesNotPay(t *testing.T) {
	scores := map[string]float64{"premium": 0.3, "near": 0.9}
	cfg := &SessionAwareConfig{QualityGapMultiplier: 1, MaxCacheCostMultiplier: 1}

	before := selectWithCacheEconomics(t, cfg, scores, "near", warmCacheSession("premium"))
	if before.SelectedModel != "near" || before.SessionPolicy.CacheEconomics != nil {
		t.Fatalf("without the policy the heuristics switch: %q %#v", before.SelectedModel, before.SessionPolicy.CacheEconomics)
	}

	cfg.CacheEconomics = testCacheEconomicsConfig()
	result := selectWithCacheEconomics(t, cfg, scores, "near", warmCacheSession("premium"))
	if result.SelectedModel != "premium" {
		t.Fatalf("a switch that does not repay its cache rewrite was made: %q", result.SelectedModel)
	}
	trace := result.SessionPolicy
	if trace.DecisionReason != SessionReasonCacheSwitchDoesNotPay || trace.CacheEconomics == nil ||
		trace.CacheEconomics.ProposedModel != "near" || trace.CacheEconomics.Decision != CacheEconomicsHold {
		t.Fatalf("policy trace = %q %#v", trace.DecisionReason, trace.CacheEconomics)
	}
	mapped := trace.ToMap()
	economics, _ := mapped["cache_economics"].(map[string]interface{})
	candidate, _ := mapped["candidate_traces"].(map[string]interface{})["near"].(map[string]interface{})
	if economics["reason"] != CacheReasonSwitchDoesNotPay || candidate["cache_economics"] == nil {
		t.Fatalf("replay map = %#v", mapped)
	}
}

// The heuristic cache terms (0.2 weight, 1.75 cost multiplier, warm session)
// hold premium at 0.92 against cheap at 0.43; the receipts show cheap repays
// its rewrite, so they no longer count.
func TestSessionAwareCacheEconomicsReplaceTheHeuristicCacheTerms(t *testing.T) {
	scores := map[string]float64{"premium": 0.50, "cheap": 0.85}
	cfg := func() *SessionAwareConfig {
		return &SessionAwareConfig{
			StayBias: 0.10, SwitchMargin: 0.05, PrefixCacheWeight: 0.20,
			HandoffPenaltyWeight: 1, DefaultHandoffPenalty: 0.05,
			QualityGapMultiplier: 1, MaxCacheCostMultiplier: 2.5,
			CacheEconomics: testCacheEconomicsConfig(),
		}
	}

	result := selectWithCacheEconomics(t, cfg(), scores, "cheap", warmCacheSession("premium"))
	if result.SelectedModel != "cheap" || result.SessionPolicy.DecisionReason != SessionReasonCacheSwitchPays {
		t.Fatalf("a switch that repays its cache rewrite was held: %q %q", result.SelectedModel, result.SessionPolicy.DecisionReason)
	}
	if trace := result.SessionPolicy.CandidateTraces["cheap"]; trace.PrefixCachePenalty != 0 || trace.CacheEconomics == nil {
		t.Fatalf("heuristic cache penalty still applied: %#v", trace)
	}

	unknown := warmCacheSession("premium")
	unknown.CacheReceipt = nil
	fallback := selectWithCacheEconomics(t, cfg(), scores, "cheap", unknown)
	if fallback.SelectedModel != "premium" {
		t.Fatalf("without a receipt the heuristics must decide as before, got %q", fallback.SelectedModel)
	}
	if economics := fallback.SessionPolicy.CacheEconomics; economics == nil ||
		economics.Decision != CacheEconomicsFallback || economics.Reason != CacheReasonNoReceipt {
		t.Fatalf("fallback must say why: %#v", economics)
	}
}

func TestSessionAwareCacheEconomicsLeaveAnHonoredReleaseAlone(t *testing.T) {
	session := warmCacheSession("premium")
	session.ReleaseHonored = true
	cfg := &SessionAwareConfig{QualityGapMultiplier: 1, MaxCacheCostMultiplier: 1, CacheEconomics: testCacheEconomicsConfig()}
	result := selectWithCacheEconomics(t, cfg, map[string]float64{"premium": 0.3, "near": 0.9}, "near", session)
	if result.SelectedModel != "near" || result.SessionPolicy.CacheEconomics != nil {
		t.Fatalf("a client release asks for a fresh decision: %q %#v", result.SelectedModel, result.SessionPolicy.CacheEconomics)
	}
}
