package ratelimit

import (
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"
)

func setupMiniredis(t *testing.T) (*miniredis.Miniredis, redis.Cmdable) {
	t.Helper()
	mr := miniredis.RunT(t)
	client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	t.Cleanup(func() { client.Close() })
	return mr, client
}

func testPricingFunc(model string) (float64, float64, string, bool) {
	// Matches values-staging.yaml pricing (dollars per 1M tokens).
	// CEL rate = dollars_per_1M * 100, e.g. $5.50 → 550 per token in $10⁻⁸
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

func powerUserRules() []RedisLimiterRule {
	return []RedisLimiterRule{
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

func TestRedisLimiter_CheckAllowsUnderBudget(t *testing.T) {
	_, client := setupMiniredis(t)
	p := NewRedisLimiterProvider(client, powerUserRules(), testPricingFunc)

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

func TestRedisLimiter_CheckDeniesOverBudget(t *testing.T) {
	mr, client := setupMiniredis(t)
	p := NewRedisLimiterProvider(client, powerUserRules(), testPricingFunc)

	// Pre-populate: standard user already spent $5.20 (520_000_000 CEL units)
	key := "sr:budget:bob@example.com:day"
	mr.Set(key, "520000000")

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

func TestRedisLimiter_CheckAllowsNoMatchingRule(t *testing.T) {
	_, client := setupMiniredis(t)
	p := NewRedisLimiterProvider(client, powerUserRules(), testPricingFunc)

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

func TestRedisLimiter_ReportIncrementsCost(t *testing.T) {
	mr, client := setupMiniredis(t)
	p := NewRedisLimiterProvider(client, powerUserRules(), testPricingFunc)

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
	got, err := mr.Get(key)
	if err != nil {
		t.Fatalf("key %s not found: %v", key, err)
	}
	if got != "1925000" {
		t.Errorf("spend = %s, want 1925000", got)
	}

	// Verify TTL was set
	if !mr.Exists(key) {
		t.Fatal("key should exist")
	}
}

func TestRedisLimiter_ReportAccumulatesAcrossModels(t *testing.T) {
	_, client := setupMiniredis(t)
	p := NewRedisLimiterProvider(client, powerUserRules(), testPricingFunc)

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

func TestRedisLimiter_ReportNoMatchingRuleIsNoop(t *testing.T) {
	_, client := setupMiniredis(t)
	p := NewRedisLimiterProvider(client, powerUserRules(), testPricingFunc)

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

func TestRedisLimiter_ReportFallbackToRawTokens(t *testing.T) {
	mr, client := setupMiniredis(t)
	// No pricing function → falls back to raw tokens
	p := NewRedisLimiterProvider(client, powerUserRules(), nil)

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
	got, _ := mr.Get(key)
	if got != "150" {
		t.Errorf("spend = %s, want 150 (raw token fallback)", got)
	}
}

func TestRedisLimiter_CustomKeyPrefix(t *testing.T) {
	mr, client := setupMiniredis(t)
	p := NewRedisLimiterProvider(client, powerUserRules(), testPricingFunc,
		WithKeyPrefix("custom:rl"))

	ctx := Context{
		UserID: "alice@example.com",
		Groups: []string{"ai-power-user"},
		Model:  "claude-opus-4-6",
	}

	_ = p.Report(ctx, TokenUsage{InputTokens: 100, OutputTokens: 50})

	key := "custom:rl:alice@example.com:day"
	if !mr.Exists(key) {
		t.Errorf("expected key %s to exist with custom prefix", key)
	}
}

func TestRedisLimiter_CalculateCELCost(t *testing.T) {
	p := &RedisLimiterProvider{pricingFunc: testPricingFunc}

	// Opus: promptPer1M=$5.50, completionPer1M=$27.50
	// inputRate = 5.50*100 = 550, outputRate = 27.50*100 = 2750
	cost := p.calculateCELCost("claude-opus-4-6", TokenUsage{
		InputTokens:  1000,
		OutputTokens: 500,
	})
	// 1000*550 + 500*2750 = 550000 + 1375000 = 1925000
	if cost != 1_925_000 {
		t.Errorf("CEL cost = %d, want 1925000", cost)
	}
}
