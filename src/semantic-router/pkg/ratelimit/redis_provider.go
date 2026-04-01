package ratelimit

import (
	"context"
	"fmt"
	"slices"
	"time"

	"github.com/redis/go-redis/v9"

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

// RedisLimiterRule defines a group-based budget rule for the redis-limiter.
type RedisLimiterRule struct {
	Name          string
	Match         RuleMatch
	TokensPerUnit int64
	Unit          time.Duration
}

// RedisLimiterProvider implements Provider using Redis for cross-model,
// group-aware budget enforcement. Each user's spend across all models
// accumulates into a single Redis counter keyed by user ID + time window.
//
// Cost is expressed in CEL units (1 unit = $10⁻⁸) for consistency with
// the Envoy AI Gateway's CEL cost formulas.
type RedisLimiterProvider struct {
	client          redis.Cmdable
	rules           []RedisLimiterRule
	pricingFunc     ModelPricingFunc
	pricingFullFunc ModelPricingFullFunc
	keyPrefix       string
}

// RedisLimiterOption configures a RedisLimiterProvider.
type RedisLimiterOption func(*RedisLimiterProvider)

// WithKeyPrefix sets a custom Redis key prefix (default "sr:budget").
func WithKeyPrefix(prefix string) RedisLimiterOption {
	return func(p *RedisLimiterProvider) {
		p.keyPrefix = prefix
	}
}

// WithFullPricingFunc sets a pricing function that includes cache read/write rates.
// When set, calculateCELCost uses this instead of the basic pricingFunc.
func WithFullPricingFunc(fn ModelPricingFullFunc) RedisLimiterOption {
	return func(p *RedisLimiterProvider) {
		p.pricingFullFunc = fn
	}
}

// NewRedisLimiterProvider creates a redis-limiter provider.
// The client can be any go-redis Cmdable (real client or mock).
// The pricingFunc is called during Report() to convert tokens to CEL cost.
func NewRedisLimiterProvider(
	client redis.Cmdable,
	rules []RedisLimiterRule,
	pricingFunc ModelPricingFunc,
	opts ...RedisLimiterOption,
) *RedisLimiterProvider {
	p := &RedisLimiterProvider{
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

func (p *RedisLimiterProvider) Name() string {
	return "redis-limiter"
}

// Check reads the user's accumulated spend from Redis and compares it
// against the budget for their group. The check is cross-model: the key
// contains no model identifier, so all model costs accumulate together.
func (p *RedisLimiterProvider) Check(ctx Context) (*Decision, error) {
	rule := p.findMatchingRule(ctx)
	if rule == nil {
		return &Decision{
			Allowed:  true,
			Provider: p.Name(),
		}, nil
	}

	key := p.redisKey(ctx.UserID, rule.Unit)
	bg := context.Background()

	currentSpend, err := p.client.Get(bg, key).Int64()
	if err != nil && err != redis.Nil {
		return nil, fmt.Errorf("redis GET %s: %w", key, err)
	}
	// redis.Nil means key doesn't exist → currentSpend stays 0

	remaining := max(rule.TokensPerUnit-currentSpend, 0)

	resetAt := time.Now().Add(rule.Unit)
	// Try to get the actual TTL for a more accurate reset time.
	ttl, err := p.client.TTL(bg, key).Result()
	if err == nil && ttl > 0 {
		resetAt = time.Now().Add(ttl)
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
// and increments the user's Redis counter.
//
// CEL cost = inputTokens × (promptPer1M / 1e6 × 1e8) + outputTokens × (completionPer1M / 1e6 × 1e8)
// Simplified: inputTokens × promptPer1M × 100 + outputTokens × completionPer1M × 100
//
// If no pricing is configured for the model, raw token count is used as fallback.
func (p *RedisLimiterProvider) Report(ctx Context, usage TokenUsage) error {
	rule := p.findMatchingRule(ctx)
	if rule == nil {
		return nil
	}

	cost := p.calculateCELCost(ctx.Model, usage)
	if cost <= 0 {
		return nil
	}

	key := p.redisKey(ctx.UserID, rule.Unit)
	bg := context.Background()

	newTotal, err := p.client.IncrBy(bg, key, cost).Result()
	if err != nil {
		return fmt.Errorf("redis INCRBY %s: %w", key, err)
	}

	// Set expiry if this is a new key (value equals the cost we just added).
	if newTotal == cost {
		if err := p.client.Expire(bg, key, rule.Unit).Err(); err != nil {
			logging.Warnf("redis-limiter: failed to set TTL on %s: %v", key, err)
		}
	}

	logging.Debugf("redis-limiter: report user=%s model=%s cost=%d new_total=%d limit=%d",
		ctx.UserID, ctx.Model, cost, newTotal, rule.TokensPerUnit)

	return nil
}

// Close closes the underlying Redis client if it implements io.Closer.
func (p *RedisLimiterProvider) Close() error {
	if c, ok := p.client.(*redis.Client); ok {
		return c.Close()
	}
	return nil
}

func (p *RedisLimiterProvider) findMatchingRule(ctx Context) *RedisLimiterRule {
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

func (p *RedisLimiterProvider) redisKey(userID string, unit time.Duration) string {
	return fmt.Sprintf("%s:%s:%s", p.keyPrefix, userID, unitLabel(unit))
}

// calculateCELCost converts token usage into CEL cost units.
// CEL units are $10⁻⁸, so $1 = 100,000,000 CEL units.
//
// The gateway's CEL formula uses per-token rates that are already in
// $10⁻⁸ units. Model pricing in config is per-1M tokens in dollars, so:
//
//	rate_per_token_in_cel = (dollars_per_1M / 1e6) × 1e8 = dollars_per_1M × 100
//
// When full pricing is available (cache_read_per_1m, cache_write_per_1m),
// the formula matches the Envoy AI Gateway CEL expression:
//
//	input_tokens * inputRate + output_tokens * outputRate
//	  + cached_input_tokens * cacheReadRate
//	  + cache_creation_input_tokens * cacheWriteRate
func (p *RedisLimiterProvider) calculateCELCost(model string, usage TokenUsage) int64 {
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
