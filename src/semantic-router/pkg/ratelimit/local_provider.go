package ratelimit

import (
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// LocalLimiter is an in-process rate limiter inspired by the Envoy AI Gateway's
// usage-based rate limiting. It supports two rule types:
//
//   - requests_per_unit (RPM): classic request counting per time window
//   - tokens_per_unit  (TPM): token budget tracking (AI Gateway-style)
//
// Rules are matched by user/group/model patterns. Token usage is reported
// after response via Report() to decrement token budgets.
//
// This provider uses sliding window counters keyed by a composite of
// user+model+rule, providing per-user per-model granularity without
// requiring an external service.
type LocalLimiter struct {
	rules   []Rule
	buckets sync.Map // key → *bucket
}

// Rule defines a single rate limit rule.
type Rule struct {
	Name            string
	Match           RuleMatch
	RequestsPerUnit int
	TokensPerUnit   int
	Unit            time.Duration
}

// RuleMatch specifies which requests a rule applies to.
// Empty string matches everything; "*" is an explicit wildcard.
type RuleMatch struct {
	User  string
	Group string
	Model string
}

type bucket struct {
	mu        sync.Mutex
	count     int64
	windowEnd time.Time
	unit      time.Duration
	limit     int64
}

// NewLocalLimiter creates a local limiter with the given rules.
func NewLocalLimiter(rules []Rule) *LocalLimiter {
	return &LocalLimiter{rules: rules}
}

func (l *LocalLimiter) Name() string {
	return "local-limiter"
}

// Check evaluates all matching rules. If any rule's budget is exhausted,
// the request is denied.
func (l *LocalLimiter) Check(ctx Context) (*Decision, error) {
	now := time.Now()

	var mostRestrictiveRemaining int64 = -1
	var mostRestrictiveLimit int64 = -1
	var earliestReset time.Time

	for i := range l.rules {
		rule := &l.rules[i]
		if !l.ruleMatches(rule, ctx) {
			continue
		}

		if rule.RequestsPerUnit > 0 {
			key := l.bucketKey(rule.Name, ctx.UserID, ctx.Model, "rpm")
			b := l.getOrCreateBucket(key, int64(rule.RequestsPerUnit), rule.Unit)
			allowed, remaining, resetAt := b.tryConsume(1, now)
			if !allowed {
				return &Decision{
					Allowed:    false,
					Remaining:  remaining,
					Limit:      int64(rule.RequestsPerUnit),
					ResetAt:    resetAt,
					RetryAfter: time.Until(resetAt),
					Provider:   l.Name(),
				}, nil
			}
			if mostRestrictiveRemaining < 0 || remaining < mostRestrictiveRemaining {
				mostRestrictiveRemaining = remaining
			}
			if mostRestrictiveLimit < 0 || int64(rule.RequestsPerUnit) < mostRestrictiveLimit {
				mostRestrictiveLimit = int64(rule.RequestsPerUnit)
			}
			if earliestReset.IsZero() || resetAt.Before(earliestReset) {
				earliestReset = resetAt
			}
		}

		if rule.TokensPerUnit > 0 {
			key := l.bucketKey(rule.Name, ctx.UserID, ctx.Model, "tpm")
			b := l.getOrCreateBucket(key, int64(rule.TokensPerUnit), rule.Unit)
			cost := int64(ctx.TokenCount)
			if cost <= 0 {
				cost = 0
			}
			allowed, remaining, resetAt := b.tryConsume(cost, now)
			if !allowed {
				return &Decision{
					Allowed:    false,
					Remaining:  remaining,
					Limit:      int64(rule.TokensPerUnit),
					ResetAt:    resetAt,
					RetryAfter: time.Until(resetAt),
					Provider:   l.Name(),
				}, nil
			}
			if mostRestrictiveRemaining < 0 || remaining < mostRestrictiveRemaining {
				mostRestrictiveRemaining = remaining
			}
			if mostRestrictiveLimit < 0 || int64(rule.TokensPerUnit) < mostRestrictiveLimit {
				mostRestrictiveLimit = int64(rule.TokensPerUnit)
			}
			if earliestReset.IsZero() || resetAt.Before(earliestReset) {
				earliestReset = resetAt
			}
		}
	}

	if mostRestrictiveRemaining < 0 {
		mostRestrictiveRemaining = 0
	}
	if mostRestrictiveLimit < 0 {
		mostRestrictiveLimit = 0
	}

	return &Decision{
		Allowed:   true,
		Remaining: mostRestrictiveRemaining,
		Limit:     mostRestrictiveLimit,
		ResetAt:   earliestReset,
		Provider:  l.Name(),
	}, nil
}

// Report records actual token usage from LLM responses for TPM rules.
// This adjusts the token budget based on actual usage rather than the
// estimated input token count used during Check().
func (l *LocalLimiter) Report(ctx Context, usage TokenUsage) error {
	for i := range l.rules {
		rule := &l.rules[i]
		if rule.TokensPerUnit <= 0 || !l.ruleMatches(rule, ctx) {
			continue
		}

		key := l.bucketKey(rule.Name, ctx.UserID, ctx.Model, "tpm")
		b := l.getOrCreateBucket(key, int64(rule.TokensPerUnit), rule.Unit)

		// The Check() already consumed estimated input tokens.
		// Now add the output tokens (the part we didn't know during Check).
		additionalTokens := int64(usage.OutputTokens)
		if additionalTokens > 0 {
			b.consumeOnly(additionalTokens, time.Now())
			logging.Debugf("Rate limit report: rule=%s user=%s model=%s output_tokens=%d",
				rule.Name, ctx.UserID, ctx.Model, additionalTokens)
		}
	}
	return nil
}

func (l *LocalLimiter) ruleMatches(rule *Rule, ctx Context) bool {
	if rule.Match.User != "" && rule.Match.User != "*" && rule.Match.User != ctx.UserID {
		return false
	}
	if rule.Match.Model != "" && rule.Match.Model != "*" && rule.Match.Model != ctx.Model {
		return false
	}
	if rule.Match.Group != "" && rule.Match.Group != "*" {
		matched := false
		for _, g := range ctx.Groups {
			if g == rule.Match.Group {
				matched = true
				break
			}
		}
		if !matched {
			return false
		}
	}
	return true
}

func (l *LocalLimiter) bucketKey(ruleName, userID, model, kind string) string {
	return strings.Join([]string{ruleName, userID, model, kind}, "|")
}

func (l *LocalLimiter) getOrCreateBucket(key string, limit int64, unit time.Duration) *bucket {
	if v, ok := l.buckets.Load(key); ok {
		return v.(*bucket)
	}
	b := &bucket{limit: limit, unit: unit}
	actual, _ := l.buckets.LoadOrStore(key, b)
	return actual.(*bucket)
}

// tryConsume attempts to consume `cost` from the bucket.
// Returns (allowed, remaining, windowEnd).
func (b *bucket) tryConsume(cost int64, now time.Time) (bool, int64, time.Time) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if now.After(b.windowEnd) {
		b.count = 0
		b.windowEnd = now.Add(b.unit)
	}

	remaining := b.limit - b.count
	if cost > remaining {
		return false, remaining, b.windowEnd
	}

	b.count += cost
	return true, b.limit - b.count, b.windowEnd
}

// consumeOnly decrements the bucket without checking. Used for post-response
// token reporting where we must track usage even if it goes negative.
func (b *bucket) consumeOnly(cost int64, now time.Time) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if now.After(b.windowEnd) {
		b.count = 0
		b.windowEnd = now.Add(b.unit)
	}

	b.count += cost
}

// ParseUnit converts a unit string to a time.Duration.
func ParseUnit(unit string) time.Duration {
	switch strings.ToLower(unit) {
	case "second":
		return time.Second
	case "minute":
		return time.Minute
	case "hour":
		return time.Hour
	case "day":
		return 24 * time.Hour
	default:
		return time.Minute
	}
}
