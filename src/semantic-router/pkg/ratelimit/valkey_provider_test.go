package ratelimit

import (
	"context"
	"fmt"
	"strconv"
	"sync"
	"testing"
	"time"
)

// mockValkeyClient is an in-memory ValkeyClient for unit tests.
type mockValkeyClient struct {
	mu   sync.Mutex
	data map[string]string
	ttls map[string]time.Time
}

func newMockValkeyClient() *mockValkeyClient {
	return &mockValkeyClient{
		data: make(map[string]string),
		ttls: make(map[string]time.Time),
	}
}

func (m *mockValkeyClient) Get(_ context.Context, key string) (string, bool, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	val, ok := m.data[key]
	return val, ok, nil
}

func (m *mockValkeyClient) IncrBy(_ context.Context, key string, amount int64) (int64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	cur, _ := strconv.ParseInt(m.data[key], 10, 64)
	cur += amount
	m.data[key] = strconv.FormatInt(cur, 10)
	return cur, nil
}

func (m *mockValkeyClient) Expire(_ context.Context, key string, d time.Duration) (bool, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, ok := m.data[key]; !ok {
		return false, nil
	}
	m.ttls[key] = time.Now().Add(d)
	return true, nil
}

func (m *mockValkeyClient) TTL(_ context.Context, key string) (int64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	exp, ok := m.ttls[key]
	if !ok {
		return -1, nil
	}
	rem := time.Until(exp)
	if rem <= 0 {
		return -2, nil
	}
	return int64(rem.Seconds()), nil
}

func (m *mockValkeyClient) CustomCommand(_ context.Context, args []string) (interface{}, error) {
	// Simulate the atomic INCRBY+EXPIRE Lua script used by ReportUsage
	if len(args) >= 6 && args[0] == "EVAL" {
		key := args[3]
		amount, _ := strconv.ParseInt(args[4], 10, 64)
		ttlSec, _ := strconv.ParseInt(args[5], 10, 64)
		m.mu.Lock()
		defer m.mu.Unlock()
		cur, _ := strconv.ParseInt(m.data[key], 10, 64)
		cur += amount
		m.data[key] = strconv.FormatInt(cur, 10)
		if cur == amount {
			m.ttls[key] = time.Now().Add(time.Duration(ttlSec) * time.Second)
		}
		return cur, nil
	}
	return nil, fmt.Errorf("unsupported custom command: %v", args)
}

func (m *mockValkeyClient) Close() {}

// set is a helper to pre-populate the mock store.
func (m *mockValkeyClient) set(key, value string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.data[key] = value
}

// get is a helper to read from the mock store.
func (m *mockValkeyClient) get(key string) (string, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	v, ok := m.data[key]
	return v, ok
}

// exists checks whether the key is present.
func (m *mockValkeyClient) exists(key string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	_, ok := m.data[key]
	return ok
}

// --- helpers reused across tests ---

func testValkeyPricingFunc(model string) (float64, float64, string, bool) {
	pricing := map[string][2]float64{
		"claude-opus-4-6":   {5.50, 27.50},
		"claude-sonnet-4-6": {3.30, 16.50},
		"claude-sonnet-4-5": {3.00, 15.00},
		"claude-haiku-4-5":  {1.0, 5.0},
		"meta-llama3-2-1b":  {0.10, 0.10},
	}
	if p, ok := pricing[model]; ok {
		return p[0], p[1], "USD", true
	}
	return 0, 0, "", false
}

func valkeyPowerUserRules() []ValkeyLimiterRule {
	return []ValkeyLimiterRule{
		{
			Name:          "power-user-budget",
			Match:         RuleMatch{Group: "ai-power-user"},
			TokensPerUnit: 1_500_000_000, // $15
			Unit:          24 * time.Hour,
		},
		{
			Name:          "standard-user-budget",
			Match:         RuleMatch{Group: "ai-standard-user"},
			TokensPerUnit: 500_000_000, // $5
			Unit:          24 * time.Hour,
		},
	}
}

// --- tests ---

func TestValkeyLimiter_CheckAllowsUnderBudget(t *testing.T) {
	client := newMockValkeyClient()
	p := NewValkeyLimiterProvider(client, valkeyPowerUserRules(), testValkeyPricingFunc)

	ctx := Context{
		UserID: "alice@example.com",
		Groups: []string{"ai-power-user"},
		Model:  "claude-opus-4-6",
	}

	d, err := p.Check(ctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !d.Allowed {
		t.Error("expected ALLOW for user with no spend")
	}
	if d.Limit != 1_500_000_000 {
		t.Errorf("limit = %d, want 1500000000", d.Limit)
	}
}

func TestValkeyLimiter_CheckDeniesOverBudget(t *testing.T) {
	client := newMockValkeyClient()
	p := NewValkeyLimiterProvider(client, valkeyPowerUserRules(), testValkeyPricingFunc)

	// Pre-populate: standard user already spent $5.20 (520_000_000 CEL units)
	key := "sr:budget:bob@example.com:day"
	client.set(key, "520000000")

	ctx := Context{
		UserID: "bob@example.com",
		Groups: []string{"ai-standard-user"},
		Model:  "claude-opus-4-6",
	}

	d, err := p.Check(ctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if d.Allowed {
		t.Error("expected DENY for user over budget")
	}
	if d.Remaining != 0 {
		t.Errorf("remaining = %d, want 0", d.Remaining)
	}
}

func TestValkeyLimiter_CheckAllowsNoMatchingRule(t *testing.T) {
	client := newMockValkeyClient()
	p := NewValkeyLimiterProvider(client, valkeyPowerUserRules(), testValkeyPricingFunc)

	ctx := Context{
		UserID: "guest@example.com",
		Groups: []string{"unknown-group"},
		Model:  "claude-opus-4-6",
	}

	d, err := p.Check(ctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !d.Allowed {
		t.Error("expected ALLOW when no rule matches")
	}
}

func TestValkeyLimiter_ReportIncrementsCost(t *testing.T) {
	client := newMockValkeyClient()
	p := NewValkeyLimiterProvider(client, valkeyPowerUserRules(), testValkeyPricingFunc)

	ctx := Context{
		UserID: "alice@example.com",
		Groups: []string{"ai-power-user"},
		Model:  "claude-opus-4-6",
	}
	usage := TokenUsage{InputTokens: 1000, OutputTokens: 500}

	err := p.Report(ctx, usage)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Expected CEL cost:
	// input: 1000 * (5.50 * 100) = 1000 * 550 = 550,000
	// output: 500 * (27.50 * 100) = 500 * 2750 = 1,375,000
	// total: 1,925,000
	key := "sr:budget:alice@example.com:day"
	got, ok := client.get(key)
	if !ok {
		t.Fatalf("key %s not found", key)
	}
	if got != "1925000" {
		t.Errorf("spend = %s, want 1925000", got)
	}

	// Verify key exists (TTL was set)
	if !client.exists(key) {
		t.Fatal("key should exist")
	}
}

func TestValkeyLimiter_ReportAccumulatesAcrossModels(t *testing.T) {
	client := newMockValkeyClient()
	p := NewValkeyLimiterProvider(client, valkeyPowerUserRules(), testValkeyPricingFunc)

	ctx := Context{
		UserID: "alice@example.com",
		Groups: []string{"ai-power-user"},
	}

	// First request: Opus
	ctx.Model = "claude-opus-4-6"
	if err := p.Report(ctx, TokenUsage{InputTokens: 1000, OutputTokens: 500}); err != nil {
		t.Fatal(err)
	}

	// Second request: Haiku
	ctx.Model = "claude-haiku-4-5"
	if err := p.Report(ctx, TokenUsage{InputTokens: 2000, OutputTokens: 1000}); err != nil {
		t.Fatal(err)
	}

	// Opus cost: 1000*550 + 500*2750 = 1,925,000
	// Haiku cost: 2000*100 + 1000*500 = 700,000
	// Total: 2,625,000
	d, err := p.Check(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if !d.Allowed {
		t.Error("expected ALLOW, user is under $15 budget")
	}
	expectedRemaining := int64(1_500_000_000 - 2_625_000)
	if d.Remaining != expectedRemaining {
		t.Errorf("remaining = %d, want %d", d.Remaining, expectedRemaining)
	}
}

func TestValkeyLimiter_ReportNoMatchingRuleIsNoop(t *testing.T) {
	client := newMockValkeyClient()
	p := NewValkeyLimiterProvider(client, valkeyPowerUserRules(), testValkeyPricingFunc)

	ctx := Context{
		UserID: "guest@example.com",
		Groups: []string{"unknown"},
		Model:  "claude-opus-4-6",
	}

	err := p.Report(ctx, TokenUsage{InputTokens: 1000, OutputTokens: 500})
	if err != nil {
		t.Fatalf("expected no error for unmatched rule, got: %v", err)
	}
}

func TestValkeyLimiter_ReportFallbackToRawTokens(t *testing.T) {
	client := newMockValkeyClient()
	// No pricing function → falls back to raw tokens
	p := NewValkeyLimiterProvider(client, valkeyPowerUserRules(), nil)

	ctx := Context{
		UserID: "alice@example.com",
		Groups: []string{"ai-power-user"},
		Model:  "unknown-model",
	}

	err := p.Report(ctx, TokenUsage{InputTokens: 100, OutputTokens: 50})
	if err != nil {
		t.Fatal(err)
	}

	key := "sr:budget:alice@example.com:day"
	got, _ := client.get(key)
	if got != "150" {
		t.Errorf("spend = %s, want 150 (raw token fallback)", got)
	}
}

func TestValkeyLimiter_CustomKeyPrefix(t *testing.T) {
	client := newMockValkeyClient()
	p := NewValkeyLimiterProvider(client, valkeyPowerUserRules(), testValkeyPricingFunc,
		WithValkeyKeyPrefix("custom:rl"))

	ctx := Context{
		UserID: "alice@example.com",
		Groups: []string{"ai-power-user"},
		Model:  "claude-opus-4-6",
	}

	_ = p.Report(ctx, TokenUsage{InputTokens: 100, OutputTokens: 50})

	key := "custom:rl:alice@example.com:day"
	if !client.exists(key) {
		t.Errorf("expected key %s to exist with custom prefix", key)
	}
}

func TestValkeyLimiter_CalculateCELCost(t *testing.T) {
	p := &ValkeyLimiterProvider{pricingFunc: testValkeyPricingFunc}

	cost := p.calculateCELCost("claude-opus-4-6", TokenUsage{
		InputTokens:  1000,
		OutputTokens: 500,
	})
	// 1000*550 + 500*2750 = 550000 + 1375000 = 1925000
	if cost != 1_925_000 {
		t.Errorf("CEL cost = %d, want 1925000", cost)
	}
}

func TestValkeyLimiter_Name(t *testing.T) {
	p := &ValkeyLimiterProvider{}
	if p.Name() != "valkey-limiter" {
		t.Errorf("Name() = %q, want %q", p.Name(), "valkey-limiter")
	}
}

// --- CEL parity tests adapted for Valkey provider ---

func TestValkeyCELParity_InputOutputOnly(t *testing.T) {
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

func TestValkeyCELParity_FullFormula(t *testing.T) {
	p := &ValkeyLimiterProvider{
		pricingFullFunc: testFullPricingFunc,
	}

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

func TestValkeyCELParity_CrossModelBudget_WithCacheTokens(t *testing.T) {
	client := newMockValkeyClient()

	rules := []ValkeyLimiterRule{
		{
			Name:          "power-user-budget",
			Match:         RuleMatch{Group: "ai-power-user"},
			TokensPerUnit: 15_000_000_000, // $150/month
			Unit:          30 * 24 * time.Hour,
		},
	}

	p := NewValkeyLimiterProvider(client, rules, testValkeyPricingFunc,
		WithValkeyFullPricingFunc(testFullPricingFunc))

	ctx := Context{
		UserID: "alice@example.com",
		Groups: []string{"ai-power-user"},
	}

	// Request 1: Sonnet 4.6 with prompt caching
	ctx.Model = "claude-sonnet-4-6"
	if err := p.Report(ctx, TokenUsage{
		InputTokens:       1000,
		OutputTokens:      500,
		CachedInputTokens: 3000,
	}); err != nil {
		t.Fatalf("report 1: %v", err)
	}
	// Cost: 1000*330 + 500*1650 + 3000*33 = 1254000

	// Request 2: Opus 4.6 with cache creation
	ctx.Model = "claude-opus-4-6"
	if err := p.Report(ctx, TokenUsage{
		InputTokens:         2000,
		OutputTokens:        1000,
		CacheCreationTokens: 5000,
	}); err != nil {
		t.Fatalf("report 2: %v", err)
	}
	// Cost: 2000*550 + 1000*2750 + 5000*687 = 7285000

	// Request 3: Haiku 4.5
	ctx.Model = "claude-haiku-4-5"
	if err := p.Report(ctx, TokenUsage{
		InputTokens:  5000,
		OutputTokens: 2000,
	}); err != nil {
		t.Fatalf("report 3: %v", err)
	}
	// Cost: 5000*100 + 2000*500 = 1500000

	expectedTotal := int64(1_254_000 + 7_285_000 + 1_500_000)

	key := "sr:budget:alice@example.com:month"
	got, ok := client.get(key)
	if !ok {
		t.Fatalf("key not found")
	}
	if got != fmt.Sprintf("%d", expectedTotal) {
		t.Errorf("total spend = %s, want %d", got, expectedTotal)
	}

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
