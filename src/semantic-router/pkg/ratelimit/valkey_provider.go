package ratelimit

import (
	"context"
	"fmt"
	"slices"
	"strconv"
	"time"

	glide "github.com/valkey-io/valkey-glide/go/v2"
	"github.com/valkey-io/valkey-glide/go/v2/config"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// ModelPricingFunc returns per-1M-token pricing for a model.
// promptPer1M and completionPer1M are in dollars; ok is false when the model
// has no pricing configured.
type ModelPricingFunc func(modelName string) (promptPer1M, completionPer1M float64, currency string, ok bool)

// ModelPricingFullFunc returns the complete pricing breakdown for a model,
// including cache read/write rates for prompt caching cost tracking.
type ModelPricingFullFunc func(modelName string) (result ModelPricingRates, ok bool)

// ModelPricingRates holds all per-1M-token rates needed for CEL cost calculation.
type ModelPricingRates struct {
	PromptPer1M     float64
	CompletionPer1M float64
	CacheReadPer1M  float64
	CacheWritePer1M float64
}

// ValkeyClient abstracts the Valkey operations used by the limiter,
// allowing *glide.Client in production and a mock in tests.
type ValkeyClient interface {
	Get(ctx context.Context, key string) (string, bool, error)
	IncrBy(ctx context.Context, key string, amount int64) (int64, error)
	Expire(ctx context.Context, key string, d time.Duration) (bool, error)
	TTL(ctx context.Context, key string) (int64, error)
	Close()
}

// glideAdapter wraps *glide.Client to implement ValkeyClient.
type glideAdapter struct {
	client *glide.Client
}

func (a *glideAdapter) Get(ctx context.Context, key string) (string, bool, error) {
	result, err := a.client.Get(ctx, key)
	if err != nil {
		return "", false, err
	}
	if result.IsNil() {
		return "", false, nil
	}
	return result.Value(), true, nil
}

func (a *glideAdapter) IncrBy(ctx context.Context, key string, amount int64) (int64, error) {
	return a.client.IncrBy(ctx, key, amount)
}

func (a *glideAdapter) Expire(ctx context.Context, key string, d time.Duration) (bool, error) {
	return a.client.Expire(ctx, key, d)
}

func (a *glideAdapter) TTL(ctx context.Context, key string) (int64, error) {
	return a.client.TTL(ctx, key)
}

func (a *glideAdapter) Close() {
	a.client.Close()
}

// NewGlideClient creates a ValkeyClient backed by a real *glide.Client.
func NewGlideClient(addr string, db int) (ValkeyClient, error) {
	host, port, err := parseHostPort(addr)
	if err != nil {
		return nil, fmt.Errorf("valkey-limiter: invalid address %q: %w", addr, err)
	}

	clientConfig := config.NewClientConfiguration().
		WithAddress(&config.NodeAddress{Host: host, Port: port})

	if db != 0 {
		clientConfig = clientConfig.WithDatabaseId(db)
	}

	client, err := glide.NewClient(clientConfig)
	if err != nil {
		return nil, fmt.Errorf("valkey-limiter: failed to connect to %s: %w", addr, err)
	}
	return &glideAdapter{client: client}, nil
}

// parseHostPort splits "host:port" into its parts.
func parseHostPort(addr string) (string, int, error) {
	host := addr
	port := 6379
	for i := len(addr) - 1; i >= 0; i-- {
		if addr[i] == ':' {
			host = addr[:i]
			p, err := strconv.Atoi(addr[i+1:])
			if err != nil {
				return "", 0, fmt.Errorf("invalid port in %q: %w", addr, err)
			}
			port = p
			break
		}
	}
	return host, port, nil
}

// ValkeyLimiterRule defines a group-based budget rule for the valkey-limiter.
type ValkeyLimiterRule struct {
	Name          string
	Match         RuleMatch
	TokensPerUnit int64
	Unit          time.Duration
}

// ValkeyLimiterProvider implements Provider using Valkey for cross-model,
// group-aware budget enforcement. Each user's spend across all models
// accumulates into a single Valkey counter keyed by user ID + time window.
//
// Cost is expressed in CEL units (1 unit = $10⁻⁸) for consistency with
// the Envoy AI Gateway's CEL cost formulas.
type ValkeyLimiterProvider struct {
	client          ValkeyClient
	rules           []ValkeyLimiterRule
	pricingFunc     ModelPricingFunc
	pricingFullFunc ModelPricingFullFunc
	keyPrefix       string
}

// ValkeyLimiterOption configures a ValkeyLimiterProvider.
type ValkeyLimiterOption func(*ValkeyLimiterProvider)

// WithValkeyKeyPrefix sets a custom Valkey key prefix (default "sr:budget").
func WithValkeyKeyPrefix(prefix string) ValkeyLimiterOption {
	return func(p *ValkeyLimiterProvider) {
		p.keyPrefix = prefix
	}
}

// WithValkeyFullPricingFunc sets a pricing function that includes cache read/write rates.
func WithValkeyFullPricingFunc(fn ModelPricingFullFunc) ValkeyLimiterOption {
	return func(p *ValkeyLimiterProvider) {
		p.pricingFullFunc = fn
	}
}

// NewValkeyLimiterProvider creates a valkey-limiter provider.
func NewValkeyLimiterProvider(
	client ValkeyClient,
	rules []ValkeyLimiterRule,
	pricingFunc ModelPricingFunc,
	opts ...ValkeyLimiterOption,
) *ValkeyLimiterProvider {
	p := &ValkeyLimiterProvider{
		client:      client,
		rules:       rules,
		pricingFunc: pricingFunc,
		keyPrefix:   "sr:budget",
	}
	for _, opt := range opts {
		opt(p)
	}
	return p
}

func (p *ValkeyLimiterProvider) Name() string {
	return "valkey-limiter"
}

// Check reads the user's accumulated spend from Valkey and compares it
// against the budget for their group.
func (p *ValkeyLimiterProvider) Check(ctx Context) (*Decision, error) {
	rule := p.findMatchingRule(ctx)
	if rule == nil {
		return &Decision{
			Allowed:  true,
			Provider: p.Name(),
		}, nil
	}

	key := p.valkeyKey(ctx.UserID, rule.Unit)
	bg := context.Background()

	val, exists, err := p.client.Get(bg, key)
	if err != nil {
		return nil, fmt.Errorf("valkey GET %s: %w", key, err)
	}

	var currentSpend int64
	if exists {
		currentSpend, err = strconv.ParseInt(val, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("valkey: invalid spend value for %s: %w", key, err)
		}
	}

	remaining := max(rule.TokensPerUnit-currentSpend, 0)

	resetAt := time.Now().Add(rule.Unit)
	ttl, err := p.client.TTL(bg, key)
	if err == nil && ttl > 0 {
		resetAt = time.Now().Add(time.Duration(ttl) * time.Second)
	}

	if currentSpend >= rule.TokensPerUnit {
		return &Decision{
			Allowed:    false,
			Remaining:  remaining,
			Limit:      rule.TokensPerUnit,
			ResetAt:    resetAt,
			RetryAfter: time.Until(resetAt),
			Provider:   p.Name(),
		}, nil
	}

	return &Decision{
		Allowed:   true,
		Remaining: remaining,
		Limit:     rule.TokensPerUnit,
		ResetAt:   resetAt,
		Provider:  p.Name(),
	}, nil
}

// Report converts token usage into CEL cost units using model pricing
// and increments the user's Valkey counter.
func (p *ValkeyLimiterProvider) Report(ctx Context, usage TokenUsage) error {
	rule := p.findMatchingRule(ctx)
	if rule == nil {
		return nil
	}

	cost := p.calculateCELCost(ctx.Model, usage)
	if cost <= 0 {
		return nil
	}

	key := p.valkeyKey(ctx.UserID, rule.Unit)
	bg := context.Background()

	newTotal, err := p.client.IncrBy(bg, key, cost)
	if err != nil {
		return fmt.Errorf("valkey INCRBY %s: %w", key, err)
	}

	// Set expiry if this is a new key (value equals the cost we just added).
	if newTotal == cost {
		if _, err := p.client.Expire(bg, key, rule.Unit); err != nil {
			logging.Warnf("valkey-limiter: failed to set TTL on %s: %v", key, err)
		}
	}

	logging.Debugf("valkey-limiter: report user=%s model=%s cost=%d new_total=%d limit=%d",
		ctx.UserID, ctx.Model, cost, newTotal, rule.TokensPerUnit)

	return nil
}

// Close closes the underlying Valkey client.
func (p *ValkeyLimiterProvider) Close() error {
	p.client.Close()
	return nil
}

func (p *ValkeyLimiterProvider) findMatchingRule(ctx Context) *ValkeyLimiterRule {
	for i := range p.rules {
		rule := &p.rules[i]
		if rule.Match.User != "" && rule.Match.User != "*" && rule.Match.User != ctx.UserID {
			continue
		}
		if rule.Match.Model != "" && rule.Match.Model != "*" && rule.Match.Model != ctx.Model {
			continue
		}
		if rule.Match.Group != "" && rule.Match.Group != "*" && !slices.Contains(ctx.Groups, rule.Match.Group) {
			continue
		}
		return rule
	}
	return nil
}

func (p *ValkeyLimiterProvider) valkeyKey(userID string, unit time.Duration) string {
	return fmt.Sprintf("%s:%s:%s", p.keyPrefix, userID, unitLabel(unit))
}

// calculateCELCost converts token usage into CEL cost units.
// CEL units are $10⁻⁸, so $1 = 100,000,000 CEL units.
func (p *ValkeyLimiterProvider) calculateCELCost(model string, usage TokenUsage) int64 {
	// Try full pricing first (includes cache rates)
	if p.pricingFullFunc != nil {
		rates, ok := p.pricingFullFunc(model)
		if ok {
			inputRate := int64(rates.PromptPer1M * 100)
			outputRate := int64(rates.CompletionPer1M * 100)
			cost := int64(usage.InputTokens)*inputRate + int64(usage.OutputTokens)*outputRate
			if usage.CachedInputTokens > 0 && rates.CacheReadPer1M > 0 {
				cost += int64(usage.CachedInputTokens) * int64(rates.CacheReadPer1M*100)
			}
			if usage.CacheCreationTokens > 0 && rates.CacheWritePer1M > 0 {
				cost += int64(usage.CacheCreationTokens) * int64(rates.CacheWritePer1M*100)
			}
			return cost
		}
	}

	// Fall back to basic pricing (input + output only)
	if p.pricingFunc == nil {
		return int64(usage.InputTokens + usage.OutputTokens)
	}

	promptPer1M, completionPer1M, _, ok := p.pricingFunc(model)
	if !ok {
		return int64(usage.InputTokens + usage.OutputTokens)
	}

	inputRate := int64(promptPer1M * 100)
	outputRate := int64(completionPer1M * 100)
	return int64(usage.InputTokens)*inputRate + int64(usage.OutputTokens)*outputRate
}

func unitLabel(d time.Duration) string {
	switch d {
	case time.Second:
		return "second"
	case time.Minute:
		return "minute"
	case time.Hour:
		return "hour"
	case 24 * time.Hour:
		return "day"
	case 30 * 24 * time.Hour:
		return "month"
	default:
		return fmt.Sprintf("%ds", int(d.Seconds()))
	}
}
