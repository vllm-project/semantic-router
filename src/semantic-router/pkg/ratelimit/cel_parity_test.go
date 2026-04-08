package ratelimit

import (
	"fmt"
	"testing"
	"time"
)

// ---------------------------------------------------------------------------
// CEL parity tests
//
// These tests verify that the semantic router's calculateCELCost produces the
// exact same cost values as the Envoy AI Gateway CEL expressions defined in:
//   cuebiq-gw-phase2.yaml → llmRequestCosts → cel
//
// Each model's CEL formula is encoded as a Go function and compared against
// calculateCELCost with full pricing (including cache read/write rates).
// ---------------------------------------------------------------------------

// envoyCELCost computes the cost exactly as the Envoy AI Gateway CEL expression
// would, using integer arithmetic with per-token rates in $10⁻⁸ units.
type envoyCELFormula struct {
	inputRate         int64
	outputRate        int64
	cacheReadRate     int64
	cacheCreationRate int64
}

func (f envoyCELFormula) cost(usage TokenUsage) int64 {
	return int64(usage.InputTokens)*f.inputRate +
		int64(usage.OutputTokens)*f.outputRate +
		int64(usage.CachedInputTokens)*f.cacheReadRate +
		int64(usage.CacheCreationTokens)*f.cacheCreationRate
}

// Models and their Envoy CEL rates from cuebiq-gw-phase2.yaml
var envoyGatewayFormulas = map[string]envoyCELFormula{
	// cel: "input_tokens * uint(330) + output_tokens * uint(1650) + cached_input_tokens * uint(33) + cache_creation_input_tokens * uint(412)"
	"claude-sonnet-4-6": {inputRate: 330, outputRate: 1650, cacheReadRate: 33, cacheCreationRate: 412},
	// cel: "input_tokens * uint(300) + output_tokens * uint(1500) + cached_input_tokens * uint(30) + cache_creation_input_tokens * uint(375)"
	"claude-sonnet-4-5": {inputRate: 300, outputRate: 1500, cacheReadRate: 30, cacheCreationRate: 375},
	// cel: "input_tokens * uint(550) + output_tokens * uint(2750) + cached_input_tokens * uint(55) + cache_creation_input_tokens * uint(687)"
	"claude-opus-4-6": {inputRate: 550, outputRate: 2750, cacheReadRate: 55, cacheCreationRate: 687},
	// cel: "input_tokens * uint(100) + output_tokens * uint(500) + cached_input_tokens * uint(10) + cache_creation_input_tokens * uint(125)"
	"claude-haiku-4-5": {inputRate: 100, outputRate: 500, cacheReadRate: 10, cacheCreationRate: 125},
	// cel: "input_tokens * uint(10) + output_tokens * uint(10)"
	"meta-llama3-2-1b": {inputRate: 10, outputRate: 10, cacheReadRate: 0, cacheCreationRate: 0},
}

// Full pricing matching values-staging.yaml (dollars per 1M tokens)
var fullPricingTable = map[string]ModelPricingRates{
	"claude-sonnet-4-6": {PromptPer1M: 3.30, CompletionPer1M: 16.50, CacheReadPer1M: 0.33, CacheWritePer1M: 4.125},
	"claude-sonnet-4-5": {PromptPer1M: 3.00, CompletionPer1M: 15.00, CacheReadPer1M: 0.30, CacheWritePer1M: 3.75},
	"claude-opus-4-6":   {PromptPer1M: 5.50, CompletionPer1M: 27.50, CacheReadPer1M: 0.55, CacheWritePer1M: 6.875},
	"claude-haiku-4-5":  {PromptPer1M: 1.00, CompletionPer1M: 5.00, CacheReadPer1M: 0.10, CacheWritePer1M: 1.25},
	"meta-llama3-2-1b":  {PromptPer1M: 0.10, CompletionPer1M: 0.10, CacheReadPer1M: 0, CacheWritePer1M: 0},
}

func testFullPricingFunc(model string) (ModelPricingRates, bool) {
	if rates, ok := fullPricingTable[model]; ok {
		return rates, true
	}
	return ModelPricingRates{}, false
}

// ---------------------------------------------------------------------------
// Test: CEL parity — every model, input+output only
// ---------------------------------------------------------------------------

func TestCELParity_InputOutputOnly(t *testing.T) {
	p := &ValkeyLimiterProvider{
		pricingFullFunc: testFullPricingFunc,
	}

	usages := []TokenUsage{
		{InputTokens: 1000, OutputTokens: 500},
		{InputTokens: 0, OutputTokens: 100},
		{InputTokens: 5000, OutputTokens: 0},
		{InputTokens: 1, OutputTokens: 1},
		{InputTokens: 1_000_000, OutputTokens: 500_000},
	}

	for model, formula := range envoyGatewayFormulas {
		for _, usage := range usages {
			t.Run(fmt.Sprintf("%s/in=%d_out=%d", model, usage.InputTokens, usage.OutputTokens), func(t *testing.T) {
				srCost := p.calculateCELCost(model, usage)
				envoyCost := formula.cost(usage)
				if srCost != envoyCost {
					t.Errorf("CEL parity mismatch for %s: semantic-router=%d, envoy-gateway=%d (delta=%d)",
						model, srCost, envoyCost, srCost-envoyCost)
				}
			})
		}
	}
}

// ---------------------------------------------------------------------------
// Test: CEL parity — with cached input tokens (prompt caching read)
// ---------------------------------------------------------------------------

func TestCELParity_WithCachedInputTokens(t *testing.T) {
	p := &ValkeyLimiterProvider{
		pricingFullFunc: testFullPricingFunc,
	}

	usages := []TokenUsage{
		{InputTokens: 500, OutputTokens: 200, CachedInputTokens: 3000},
		{InputTokens: 100, OutputTokens: 50, CachedInputTokens: 10000},
		{InputTokens: 0, OutputTokens: 100, CachedInputTokens: 50000},
	}

	for model, formula := range envoyGatewayFormulas {
		for _, usage := range usages {
			t.Run(fmt.Sprintf("%s/cached=%d", model, usage.CachedInputTokens), func(t *testing.T) {
				srCost := p.calculateCELCost(model, usage)
				envoyCost := formula.cost(usage)
				if srCost != envoyCost {
					t.Errorf("CEL parity mismatch for %s: semantic-router=%d, envoy-gateway=%d",
						model, srCost, envoyCost)
				}
			})
		}
	}
}

// ---------------------------------------------------------------------------
// Test: CEL parity — with cache creation tokens (prompt caching write)
// ---------------------------------------------------------------------------

func TestCELParity_WithCacheCreationTokens(t *testing.T) {
	p := &ValkeyLimiterProvider{
		pricingFullFunc: testFullPricingFunc,
	}

	usages := []TokenUsage{
		{InputTokens: 1000, OutputTokens: 500, CacheCreationTokens: 2000},
		{InputTokens: 100, OutputTokens: 50, CacheCreationTokens: 8000},
	}

	for model, formula := range envoyGatewayFormulas {
		for _, usage := range usages {
			t.Run(fmt.Sprintf("%s/creation=%d", model, usage.CacheCreationTokens), func(t *testing.T) {
				srCost := p.calculateCELCost(model, usage)
				envoyCost := formula.cost(usage)
				if srCost != envoyCost {
					t.Errorf("CEL parity mismatch for %s: semantic-router=%d, envoy-gateway=%d",
						model, srCost, envoyCost)
				}
			})
		}
	}
}

// ---------------------------------------------------------------------------
// Test: CEL parity — full formula (all 4 token types)
// ---------------------------------------------------------------------------

func TestCELParity_FullFormula(t *testing.T) {
	p := &ValkeyLimiterProvider{
		pricingFullFunc: testFullPricingFunc,
	}

	// Realistic scenario: request with prompt caching
	usage := TokenUsage{
		InputTokens:         1000,
		OutputTokens:        500,
		CachedInputTokens:   5000,
		CacheCreationTokens: 2000,
	}

	for model, formula := range envoyGatewayFormulas {
		t.Run(model, func(t *testing.T) {
			srCost := p.calculateCELCost(model, usage)
			envoyCost := formula.cost(usage)
			if srCost != envoyCost {
				t.Errorf("CEL parity mismatch:\n  semantic-router = %d\n  envoy-gateway   = %d\n  delta           = %d",
					srCost, envoyCost, srCost-envoyCost)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Test: CEL parity — specific gateway CEL expressions (hardcoded validation)
// ---------------------------------------------------------------------------

func TestCELParity_Sonnet46_ExactGatewayFormula(t *testing.T) {
	// From cuebiq-gw-phase2.yaml:
	// cel: "input_tokens * uint(330) + output_tokens * uint(1650) + cached_input_tokens * uint(33) + cache_creation_input_tokens * uint(412)"
	p := &ValkeyLimiterProvider{
		pricingFullFunc: testFullPricingFunc,
	}

	usage := TokenUsage{
		InputTokens:         1000,
		OutputTokens:        500,
		CachedInputTokens:   3000,
		CacheCreationTokens: 1000,
	}

	got := p.calculateCELCost("claude-sonnet-4-6", usage)
	// Manual: 1000*330 + 500*1650 + 3000*33 + 1000*412
	//       = 330000 + 825000 + 99000 + 412000 = 1666000
	want := int64(1000*330 + 500*1650 + 3000*33 + 1000*412)
	if got != want {
		t.Errorf("Sonnet 4.6 CEL cost = %d, want %d", got, want)
	}
}

func TestCELParity_Opus46_ExactGatewayFormula(t *testing.T) {
	// cel: "input_tokens * uint(550) + output_tokens * uint(2750) + cached_input_tokens * uint(55) + cache_creation_input_tokens * uint(687)"
	p := &ValkeyLimiterProvider{
		pricingFullFunc: testFullPricingFunc,
	}

	usage := TokenUsage{
		InputTokens:         2000,
		OutputTokens:        1000,
		CachedInputTokens:   10000,
		CacheCreationTokens: 5000,
	}

	got := p.calculateCELCost("claude-opus-4-6", usage)
	want := int64(2000*550 + 1000*2750 + 10000*55 + 5000*687)
	if got != want {
		t.Errorf("Opus 4.6 CEL cost = %d, want %d", got, want)
	}
}

func TestCELParity_Haiku45_ExactGatewayFormula(t *testing.T) {
	// cel: "input_tokens * uint(100) + output_tokens * uint(500) + cached_input_tokens * uint(10) + cache_creation_input_tokens * uint(125)"
	p := &ValkeyLimiterProvider{
		pricingFullFunc: testFullPricingFunc,
	}

	usage := TokenUsage{
		InputTokens:         5000,
		OutputTokens:        2000,
		CachedInputTokens:   20000,
		CacheCreationTokens: 0,
	}

	got := p.calculateCELCost("claude-haiku-4-5", usage)
	want := int64(5000*100 + 2000*500 + 20000*10 + 0*125)
	if got != want {
		t.Errorf("Haiku 4.5 CEL cost = %d, want %d", got, want)
	}
}

func TestCELParity_Llama_NoCacheTokens(t *testing.T) {
	// cel: "input_tokens * uint(10) + output_tokens * uint(10)"
	// No cache token support for Meta models
	p := &ValkeyLimiterProvider{
		pricingFullFunc: testFullPricingFunc,
	}

	usage := TokenUsage{
		InputTokens:  10000,
		OutputTokens: 5000,
	}

	got := p.calculateCELCost("meta-llama3-2-1b", usage)
	want := int64(10000*10 + 5000*10)
	if got != want {
		t.Errorf("Llama CEL cost = %d, want %d", got, want)
	}
}

// ---------------------------------------------------------------------------
// Test: Cross-model budget accumulation with cache tokens (e2e with Valkey)
// ---------------------------------------------------------------------------

func TestCELParity_CrossModelBudget_WithCacheTokens(t *testing.T) {
	mock := newMockValkeyClient()

	rules := []ValkeyLimiterRule{
		{
			Name:          "power-user-budget",
			Match:         RuleMatch{Group: "ai-power-user"},
			TokensPerUnit: 15_000_000_000, // $150/month
			Unit:          30 * 24 * time.Hour,
		},
	}

	p := NewValkeyLimiterProvider(mock, rules, testValkeyPricingFunc,
		WithValkeyFullPricingFunc(testFullPricingFunc))

	ctx := Context{
		UserID: "alice@example.com",
		Groups: []string{"ai-power-user"},
	}

	// Request 1: Sonnet 4.6 with prompt caching
	ctx.Model = "claude-sonnet-4-6"
	err := p.Report(ctx, TokenUsage{
		InputTokens:       1000,
		OutputTokens:      500,
		CachedInputTokens: 3000,
	})
	if err != nil {
		t.Fatalf("report 1: %v", err)
	}
	// Cost: 1000*330 + 500*1650 + 3000*33 = 330000 + 825000 + 99000 = 1254000

	// Request 2: Opus 4.6 with cache creation
	ctx.Model = "claude-opus-4-6"
	err = p.Report(ctx, TokenUsage{
		InputTokens:         2000,
		OutputTokens:        1000,
		CacheCreationTokens: 5000,
	})
	if err != nil {
		t.Fatalf("report 2: %v", err)
	}
	// Cost: 2000*550 + 1000*2750 + 5000*687 = 1100000 + 2750000 + 3435000 = 7285000

	// Request 3: Haiku 4.5 — cheap model, no caching
	ctx.Model = "claude-haiku-4-5"
	err = p.Report(ctx, TokenUsage{
		InputTokens:  5000,
		OutputTokens: 2000,
	})
	if err != nil {
		t.Fatalf("report 3: %v", err)
	}
	// Cost: 5000*100 + 2000*500 = 500000 + 1000000 = 1500000

	// Total: 1254000 + 7285000 + 1500000 = 10039000
	expectedTotal := int64(1_254_000 + 7_285_000 + 1_500_000)

	key := "sr:budget:alice@example.com:month"
	got, ok := mock.get(key)
	if !ok {
		t.Fatalf("key not found")
	}
	if got != fmt.Sprintf("%d", expectedTotal) {
		t.Errorf("total spend = %s, want %d", got, expectedTotal)
	}

	// Verify user is still well under budget
	d, err := p.Check(ctx)
	if err != nil {
		t.Fatalf("check: %v", err)
	}
	if !d.Allowed {
		t.Error("expected ALLOW — user spent ~$0.09, budget is $150")
	}
	expectedRemaining := int64(15_000_000_000) - expectedTotal
	if d.Remaining != expectedRemaining {
		t.Errorf("remaining = %d, want %d", d.Remaining, expectedRemaining)
	}
}

// ---------------------------------------------------------------------------
// Test: Budget exhaustion scenario with cache-heavy workload
// ---------------------------------------------------------------------------

func TestCELParity_BudgetExhaustion_CacheHeavy(t *testing.T) {
	mock := newMockValkeyClient()

	rules := []ValkeyLimiterRule{
		{
			Name:          "standard-user-budget",
			Match:         RuleMatch{Group: "ai-standard-user"},
			TokensPerUnit: 1_500_000_000, // $15/month
			Unit:          30 * 24 * time.Hour,
		},
	}

	p := NewValkeyLimiterProvider(mock, rules, testValkeyPricingFunc,
		WithValkeyFullPricingFunc(testFullPricingFunc))

	ctx := Context{
		UserID: "bob@example.com",
		Groups: []string{"ai-standard-user"},
		Model:  "claude-sonnet-4-6",
	}

	// Pre-populate: user already spent $14.90 (1_490_000_000 CEL units)
	key := "sr:budget:bob@example.com:month"
	mock.set(key, "1490000000")

	// Still allowed (under $15)
	d, err := p.Check(ctx)
	if err != nil {
		t.Fatalf("check: %v", err)
	}
	if !d.Allowed {
		t.Error("expected ALLOW — user has $0.10 remaining")
	}

	// Report a request with cache creation that pushes over budget
	// Cost: 1000*330 + 500*1650 + 2000*412 = 330000 + 825000 + 824000 = 1979000 (~$0.02)
	err = p.Report(ctx, TokenUsage{
		InputTokens:         1000,
		OutputTokens:        500,
		CacheCreationTokens: 2000,
	})
	if err != nil {
		t.Fatalf("report: %v", err)
	}

	// Now check again — should still be under (1490000000 + 1979000 = 1491979000 < 1500000000)
	d, err = p.Check(ctx)
	if err != nil {
		t.Fatalf("check: %v", err)
	}
	if !d.Allowed {
		t.Errorf("expected ALLOW — total spend %d < budget 1500000000", 1_490_000_000+1_979_000)
	}

	// Push over: add $0.10 more (10_000_000 CEL units)
	mock.set(key, "1500000001")

	d, err = p.Check(ctx)
	if err != nil {
		t.Fatalf("check: %v", err)
	}
	if d.Allowed {
		t.Error("expected DENY — user exceeded $15 budget")
	}
	if d.Remaining != 0 {
		t.Errorf("remaining = %d, want 0", d.Remaining)
	}
}

// ---------------------------------------------------------------------------
// Test: Fallback to basic pricing when full pricing is not set
// ---------------------------------------------------------------------------

func TestCELParity_FallbackToBasicPricing(t *testing.T) {
	// Provider with basic pricing only (no full pricing func)
	p := &ValkeyLimiterProvider{
		pricingFunc: testValkeyPricingFunc,
	}

	usage := TokenUsage{
		InputTokens:       1000,
		OutputTokens:      500,
		CachedInputTokens: 3000, // should be ignored
	}

	got := p.calculateCELCost("claude-opus-4-6", usage)
	// Basic pricing: 1000*550 + 500*2750 = 1925000 (no cache contribution)
	want := int64(1000*550 + 500*2750)
	if got != want {
		t.Errorf("fallback cost = %d, want %d (cache tokens should be ignored)", got, want)
	}
}

// ---------------------------------------------------------------------------
// Test: Dollar-to-CEL conversion accuracy for budget configuration
// ---------------------------------------------------------------------------

func TestCELParity_DollarToCELConversion(t *testing.T) {
	// Validates the comment in values-staging.yaml:
	// "tokens_per_unit is expressed in CEL units (1 CEL = $10⁻⁸). $1 = 100,000,000 CEL units."
	tests := []struct {
		dollars  float64
		celUnits int64
	}{
		{1.00, 100_000_000},
		{15.00, 1_500_000_000},
		{150.00, 15_000_000_000},
		{0.01, 1_000_000},
	}

	for _, tc := range tests {
		t.Run(fmt.Sprintf("$%.2f", tc.dollars), func(t *testing.T) {
			got := int64(tc.dollars * 100_000_000)
			if got != tc.celUnits {
				t.Errorf("$%.2f → %d CEL, want %d", tc.dollars, got, tc.celUnits)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Test: Zero cache tokens don't affect cost (backward compatible)
// ---------------------------------------------------------------------------

func TestCELParity_ZeroCacheTokens_MatchesBasicFormula(t *testing.T) {
	pFull := &ValkeyLimiterProvider{pricingFullFunc: testFullPricingFunc}
	pBasic := &ValkeyLimiterProvider{pricingFunc: testValkeyPricingFunc}

	usage := TokenUsage{
		InputTokens:  1000,
		OutputTokens: 500,
		// CachedInputTokens and CacheCreationTokens both 0
	}

	for model := range fullPricingTable {
		t.Run(model, func(t *testing.T) {
			fullCost := pFull.calculateCELCost(model, usage)
			basicCost := pBasic.calculateCELCost(model, usage)
			if fullCost != basicCost {
				t.Errorf("full=%d != basic=%d — zero cache tokens should produce identical costs",
					fullCost, basicCost)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Test: End-to-end Valkey integration with month unit and cache tokens
// ---------------------------------------------------------------------------

func TestCELParity_E2E_MonthlyBudget_MultiModel(t *testing.T) {
	mock := newMockValkeyClient()

	rules := []ValkeyLimiterRule{
		{
			Name:          "power-user-budget",
			Match:         RuleMatch{Group: "ai-power-user"},
			TokensPerUnit: 15_000_000_000, // $150/month
			Unit:          30 * 24 * time.Hour,
		},
		{
			Name:          "standard-user-budget",
			Match:         RuleMatch{Group: "ai-standard-user"},
			TokensPerUnit: 1_500_000_000, // $15/month
			Unit:          30 * 24 * time.Hour,
		},
	}

	p := NewValkeyLimiterProvider(mock, rules, testValkeyPricingFunc,
		WithValkeyFullPricingFunc(testFullPricingFunc))

	// Power user: allowed, then accumulates across models
	powerCtx := Context{
		UserID: "power@example.com",
		Groups: []string{"ai-power-user"},
		Model:  "claude-sonnet-4-6",
	}

	d, _ := p.Check(powerCtx)
	if !d.Allowed {
		t.Fatal("power user should be allowed at start")
	}
	if d.Limit != 15_000_000_000 {
		t.Errorf("limit = %d, want 15000000000", d.Limit)
	}

	// Standard user: lower budget
	stdCtx := Context{
		UserID: "std@example.com",
		Groups: []string{"ai-standard-user"},
		Model:  "claude-haiku-4-5",
	}

	d, _ = p.Check(stdCtx)
	if !d.Allowed {
		t.Fatal("standard user should be allowed at start")
	}
	if d.Limit != 1_500_000_000 {
		t.Errorf("limit = %d, want 1500000000", d.Limit)
	}

	// Report usage for standard user with cache tokens
	_ = p.Report(stdCtx, TokenUsage{
		InputTokens:       2000,
		OutputTokens:      1000,
		CachedInputTokens: 10000,
	})
	// Haiku cost: 2000*100 + 1000*500 + 10000*10 = 200000 + 500000 + 100000 = 800000

	d, _ = p.Check(stdCtx)
	if !d.Allowed {
		t.Error("standard user should still be allowed after small request")
	}
	expectedRemaining := int64(1_500_000_000 - 800_000)
	if d.Remaining != expectedRemaining {
		t.Errorf("remaining = %d, want %d", d.Remaining, expectedRemaining)
	}
}
