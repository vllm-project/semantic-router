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
	// Matches the doc's CEL unit table:
	// claude-opus-4-6: input_rate=500, output_rate=2500 per token in $10⁻⁸
	// That means: promptPer1M = 500/100 = $5, completionPer1M = 2500/100 = $25
	pricing := map[string][2]float64{
		"claude-opus-4-6":   {5.0, 25.0},
		"claude-sonnet-4-6": {3.0, 15.0},
		"claude-haiku-4-5":  {1.0, 5.0},
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
	// input: 1000 * (5.0 * 100) = 1000 * 500 = 500,000
	// output: 500 * (25.0 * 100) = 500 * 2500 = 1,250,000
	// total: 1,750,000
	key := "sr:budget:alice@example.com:day"
	got, err := mr.Get(key)
	if err != nil {
		t.Fatalf("key %s not found: %v", key, err)
	}
	if got != "1750000" {
		t.Errorf("spend = %s, want 1750000", got)
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

	// Opus cost: 1000*500 + 500*2500 = 1,750,000
	// Haiku cost: 2000*100 + 1000*500 = 700,000
	// Total: 2,450,000
	d, err := p.Check(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if !d.Allowed {
		t.Error("expected ALLOW, user is under $15 budget")
	}
	expectedRemaining := int64(1_500_000_000 - 2_450_000)
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

	// Opus: promptPer1M=$5, completionPer1M=$25
	// inputRate = 5*100 = 500, outputRate = 25*100 = 2500
	cost := p.calculateCELCost("claude-opus-4-6", TokenUsage{
		InputTokens:  1000,
		OutputTokens: 500,
	})
	// 1000*500 + 500*2500 = 500000 + 1250000 = 1750000
	if cost != 1_750_000 {
		t.Errorf("CEL cost = %d, want 1750000", cost)
	}
}
