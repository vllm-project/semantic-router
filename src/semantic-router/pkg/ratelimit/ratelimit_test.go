package ratelimit

import (
	"fmt"
	"testing"
	"time"
)

// ── Decision basics ──

func TestDecisionDefaults(t *testing.T) {
	d := &Decision{}
	if d.Allowed {
		t.Error("zero-value Decision.Allowed should be false")
	}
	if d.Remaining != 0 {
		t.Errorf("zero-value Remaining = %d, want 0", d.Remaining)
	}
}

// ── TokenUsage ──

func TestTokenUsageFields(t *testing.T) {
	u := TokenUsage{InputTokens: 10, OutputTokens: 20, TotalTokens: 30}
	if u.TotalTokens != u.InputTokens+u.OutputTokens {
		t.Errorf("TotalTokens = %d, want %d", u.TotalTokens, u.InputTokens+u.OutputTokens)
	}
}

// ── ParseUnit ──

func TestParseUnit(t *testing.T) {
	tests := []struct {
		input string
		want  time.Duration
	}{
		{"second", time.Second},
		{"minute", time.Minute},
		{"hour", time.Hour},
		{"day", 24 * time.Hour},
		{"SECOND", time.Second},
		{"MINUTE", time.Minute},
		{"unknown", time.Minute}, // default
	}
	for _, tc := range tests {
		if got := ParseUnit(tc.input); got != tc.want {
			t.Errorf("ParseUnit(%q) = %v, want %v", tc.input, got, tc.want)
		}
	}
}

// ── mockProvider ──

type mockProvider struct {
	name     string
	decision *Decision
	err      error
	reported []TokenUsage
}

func (m *mockProvider) Name() string { return m.name }
func (m *mockProvider) Check(_ Context) (*Decision, error) {
	if m.err != nil {
		return nil, m.err
	}
	return m.decision, nil
}

func (m *mockProvider) Report(_ Context, usage TokenUsage) error {
	m.reported = append(m.reported, usage)
	return nil
}

// ── RateLimitResolver — nil safety ──

func TestRateLimitResolverNil(t *testing.T) {
	var resolver *RateLimitResolver

	d, err := resolver.Check(Context{})
	if err != nil {
		t.Fatalf("nil resolver should not error: %v", err)
	}
	if !d.Allowed {
		t.Error("nil resolver should allow")
	}

	// Report should not panic
	resolver.Report(Context{}, TokenUsage{})

	if got := resolver.ProviderNames(); got != nil {
		t.Errorf("nil resolver ProviderNames = %v, want nil", got)
	}
	if resolver.FailOpen() {
		t.Error("nil resolver FailOpen should be false")
	}
}

// ── RateLimitResolver — empty chain ──

func TestRateLimitResolverEmpty(t *testing.T) {
	resolver := NewRateLimitResolver()

	d, err := resolver.Check(Context{})
	if err != nil {
		t.Fatalf("empty resolver should not error: %v", err)
	}
	if !d.Allowed {
		t.Error("empty resolver should allow")
	}
}

// ── RateLimitResolver — all allow ──

func TestRateLimitResolverAllAllow(t *testing.T) {
	providerA := &mockProvider{
		name: "A",
		decision: &Decision{
			Allowed:   true,
			Remaining: 50,
			Limit:     100,
			ResetAt:   time.Now().Add(30 * time.Second),
		},
	}
	providerB := &mockProvider{
		name: "B",
		decision: &Decision{
			Allowed:   true,
			Remaining: 20,
			Limit:     50,
			ResetAt:   time.Now().Add(60 * time.Second),
		},
	}

	resolver := NewRateLimitResolver(providerA, providerB)
	d, err := resolver.Check(Context{UserID: "alice", Model: "gpt-4"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !d.Allowed {
		t.Error("expected allowed when all providers allow")
	}
	if d.Remaining != 20 {
		t.Errorf("Remaining = %d, want 20 (most restrictive)", d.Remaining)
	}
	if d.Limit != 50 {
		t.Errorf("Limit = %d, want 50 (most restrictive)", d.Limit)
	}
}

// ── RateLimitResolver — first deny ──

func TestRateLimitResolverFirstDeny(t *testing.T) {
	providerA := &mockProvider{
		name: "A",
		decision: &Decision{
			Allowed:    false,
			Remaining:  0,
			Limit:      100,
			RetryAfter: 10 * time.Second,
		},
	}
	providerB := &mockProvider{
		name: "B",
		decision: &Decision{
			Allowed:   true,
			Remaining: 50,
			Limit:     100,
		},
	}

	resolver := NewRateLimitResolver(providerA, providerB)
	d, err := resolver.Check(Context{UserID: "alice"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if d.Allowed {
		t.Error("expected denied when first provider denies")
	}
	if d.Provider != "A" {
		t.Errorf("Provider = %q, want %q", d.Provider, "A")
	}
}

// ── RateLimitResolver — second deny ──

func TestRateLimitResolverSecondDeny(t *testing.T) {
	providerA := &mockProvider{
		name: "A",
		decision: &Decision{
			Allowed:   true,
			Remaining: 50,
			Limit:     100,
		},
	}
	providerB := &mockProvider{
		name: "B",
		decision: &Decision{
			Allowed:    false,
			Remaining:  0,
			Limit:      10,
			RetryAfter: 5 * time.Second,
		},
	}

	resolver := NewRateLimitResolver(providerA, providerB)
	d, err := resolver.Check(Context{UserID: "alice"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if d.Allowed {
		t.Error("expected denied when second provider denies")
	}
	if d.Provider != "B" {
		t.Errorf("Provider = %q, want %q", d.Provider, "B")
	}
}

// ── RateLimitResolver — fail-closed (default) ──

func TestRateLimitResolverFailClosed(t *testing.T) {
	providerA := &mockProvider{
		name: "A",
		err:  fmt.Errorf("connection refused"),
	}

	resolver := NewRateLimitResolver(providerA)
	d, err := resolver.Check(Context{UserID: "alice"})
	if err == nil {
		t.Fatal("expected error in fail-closed mode")
	}
	if d.Allowed {
		t.Error("expected denied in fail-closed mode on error")
	}
}

// ── RateLimitResolver — fail-open ──

func TestRateLimitResolverFailOpen(t *testing.T) {
	providerA := &mockProvider{
		name: "A",
		err:  fmt.Errorf("connection refused"),
	}
	providerB := &mockProvider{
		name: "B",
		decision: &Decision{
			Allowed:   true,
			Remaining: 50,
			Limit:     100,
		},
	}

	resolver := NewRateLimitResolver(providerA, providerB)
	resolver.SetFailOpen(true)

	d, err := resolver.Check(Context{UserID: "alice"})
	if err != nil {
		t.Fatalf("fail-open should not error: %v", err)
	}
	if !d.Allowed {
		t.Error("expected allowed in fail-open mode")
	}
}

// ── RateLimitResolver — SetFailOpen / FailOpen ──

func TestRateLimitResolverFailOpenToggle(t *testing.T) {
	resolver := NewRateLimitResolver()

	if resolver.FailOpen() {
		t.Error("default FailOpen should be false")
	}
	resolver.SetFailOpen(true)
	if !resolver.FailOpen() {
		t.Error("FailOpen should be true after SetFailOpen(true)")
	}
	resolver.SetFailOpen(false)
	if resolver.FailOpen() {
		t.Error("FailOpen should be false after SetFailOpen(false)")
	}
}

// ── RateLimitResolver — ProviderNames ──

func TestRateLimitResolverProviderNames(t *testing.T) {
	resolver := NewRateLimitResolver(
		&mockProvider{name: "envoy-ratelimit"},
		&mockProvider{name: "local-limiter"},
	)
	names := resolver.ProviderNames()
	if len(names) != 2 || names[0] != "envoy-ratelimit" || names[1] != "local-limiter" {
		t.Errorf("ProviderNames() = %v, want [envoy-ratelimit local-limiter]", names)
	}
}

// ── RateLimitResolver — Report ──

func TestRateLimitResolverReport(t *testing.T) {
	pA := &mockProvider{name: "A", decision: &Decision{Allowed: true}}
	pB := &mockProvider{name: "B", decision: &Decision{Allowed: true}}

	resolver := NewRateLimitResolver(pA, pB)
	usage := TokenUsage{InputTokens: 10, OutputTokens: 20, TotalTokens: 30}
	resolver.Report(Context{}, usage)

	if len(pA.reported) != 1 || pA.reported[0].TotalTokens != 30 {
		t.Errorf("provider A report = %v, want [%v]", pA.reported, usage)
	}
	if len(pB.reported) != 1 || pB.reported[0].TotalTokens != 30 {
		t.Errorf("provider B report = %v, want [%v]", pB.reported, usage)
	}
}

// ── LocalLimiter — RPM ──

func TestLocalLimiterRPM(t *testing.T) {
	limiter := NewLocalLimiter([]Rule{
		{
			Name:            "test-rpm",
			Match:           RuleMatch{User: "*"},
			RequestsPerUnit: 3,
			Unit:            time.Minute,
		},
	})

	ctx := Context{UserID: "alice", Model: "gpt-4"}

	// First 3 requests should be allowed
	for i := 0; i < 3; i++ {
		d, err := limiter.Check(ctx)
		if err != nil {
			t.Fatalf("request %d: unexpected error: %v", i+1, err)
		}
		if !d.Allowed {
			t.Fatalf("request %d: expected allowed", i+1)
		}
	}

	// 4th should be denied
	d, err := limiter.Check(ctx)
	if err != nil {
		t.Fatalf("request 4: unexpected error: %v", err)
	}
	if d.Allowed {
		t.Error("request 4: expected denied (RPM exhausted)")
	}
	if d.Remaining != 0 {
		t.Errorf("Remaining = %d, want 0", d.Remaining)
	}
}

// ── LocalLimiter — TPM ──

func TestLocalLimiterTPM(t *testing.T) {
	limiter := NewLocalLimiter([]Rule{
		{
			Name:          "test-tpm",
			Match:         RuleMatch{User: "*"},
			TokensPerUnit: 100,
			Unit:          time.Minute,
		},
	})

	ctx := Context{UserID: "alice", Model: "gpt-4", TokenCount: 40}

	// 40 + 40 = 80 < 100 → allowed
	d, err := limiter.Check(ctx)
	if err != nil {
		t.Fatalf("request 1: unexpected error: %v", err)
	}
	if !d.Allowed {
		t.Fatal("request 1: expected allowed")
	}

	d, err = limiter.Check(ctx)
	if err != nil {
		t.Fatalf("request 2: unexpected error: %v", err)
	}
	if !d.Allowed {
		t.Fatal("request 2: expected allowed (80 tokens used)")
	}

	// 80 + 40 = 120 > 100 → denied
	d, err = limiter.Check(ctx)
	if err != nil {
		t.Fatalf("request 3: unexpected error: %v", err)
	}
	if d.Allowed {
		t.Error("request 3: expected denied (TPM exhausted)")
	}
}

// ── LocalLimiter — group matching ──

func TestLocalLimiterGroupMatch(t *testing.T) {
	limiter := NewLocalLimiter([]Rule{
		{
			Name:            "free-rpm",
			Match:           RuleMatch{Group: "free-tier"},
			RequestsPerUnit: 2,
			Unit:            time.Minute,
		},
		{
			Name:            "premium-rpm",
			Match:           RuleMatch{Group: "premium-tier"},
			RequestsPerUnit: 100,
			Unit:            time.Minute,
		},
	})

	freeCtx := Context{UserID: "carol", Groups: []string{"free-tier"}, Model: "gpt-4"}
	premiumCtx := Context{UserID: "bob", Groups: []string{"premium-tier"}, Model: "gpt-4"}

	// Free user: 2 allowed, 3rd denied
	for i := 0; i < 2; i++ {
		d, _ := limiter.Check(freeCtx)
		if !d.Allowed {
			t.Fatalf("free request %d: expected allowed", i+1)
		}
	}
	d, _ := limiter.Check(freeCtx)
	if d.Allowed {
		t.Error("free request 3: expected denied")
	}

	// Premium user: should still be allowed
	d, _ = limiter.Check(premiumCtx)
	if !d.Allowed {
		t.Error("premium request 1: expected allowed")
	}
}

// ── LocalLimiter — model matching ──

func TestLocalLimiterModelMatch(t *testing.T) {
	limiter := NewLocalLimiter([]Rule{
		{
			Name:            "expensive-model",
			Match:           RuleMatch{User: "*", Model: "gpt-4"},
			RequestsPerUnit: 5,
			Unit:            time.Minute,
		},
	})

	gpt4Ctx := Context{UserID: "alice", Model: "gpt-4"}
	gpt3Ctx := Context{UserID: "alice", Model: "gpt-3.5-turbo"}

	// gpt-4: should match rule
	d, _ := limiter.Check(gpt4Ctx)
	if !d.Allowed || d.Limit != 5 {
		t.Errorf("gpt-4: expected allowed with limit 5, got allowed=%v limit=%d", d.Allowed, d.Limit)
	}

	// gpt-3.5-turbo: should NOT match (no rule applies)
	d, _ = limiter.Check(gpt3Ctx)
	if !d.Allowed {
		t.Error("gpt-3.5-turbo: expected allowed (no matching rule)")
	}
}

// ── LocalLimiter — Report ──

func TestLocalLimiterReport(t *testing.T) {
	limiter := NewLocalLimiter([]Rule{
		{
			Name:          "test-tpm",
			Match:         RuleMatch{User: "*"},
			TokensPerUnit: 100,
			Unit:          time.Minute,
		},
	})

	ctx := Context{UserID: "alice", Model: "gpt-4", TokenCount: 30}

	// Check consumes 30 input tokens
	d, _ := limiter.Check(ctx)
	if !d.Allowed {
		t.Fatal("expected allowed")
	}

	// Report 50 output tokens → total = 30 + 50 = 80
	_ = limiter.Report(ctx, TokenUsage{OutputTokens: 50})

	// Next request with 30 input tokens: 80 + 30 = 110 > 100 → denied
	d, _ = limiter.Check(ctx)
	if d.Allowed {
		t.Error("expected denied after report pushed usage over limit")
	}
}

// ── LocalLimiter — window reset ──

func TestLocalLimiterWindowReset(t *testing.T) {
	limiter := NewLocalLimiter([]Rule{
		{
			Name:            "test-rpm",
			Match:           RuleMatch{User: "*"},
			RequestsPerUnit: 1,
			Unit:            50 * time.Millisecond,
		},
	})

	ctx := Context{UserID: "alice"}

	d, _ := limiter.Check(ctx)
	if !d.Allowed {
		t.Fatal("first request should be allowed")
	}

	d, _ = limiter.Check(ctx)
	if d.Allowed {
		t.Fatal("second request should be denied")
	}

	// Wait for window reset
	time.Sleep(60 * time.Millisecond)

	d, _ = limiter.Check(ctx)
	if !d.Allowed {
		t.Error("request after window reset should be allowed")
	}
}

// ── LocalLimiter — no rules matches all ──

func TestLocalLimiterNoRules(t *testing.T) {
	limiter := NewLocalLimiter(nil)
	d, err := limiter.Check(Context{UserID: "alice"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !d.Allowed {
		t.Error("no rules should allow all")
	}
}
