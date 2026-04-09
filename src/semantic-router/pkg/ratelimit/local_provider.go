package ratelimit

import (
	"slices"
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

// budgetState tracks the most restrictive quota seen across evaluated rules.
type budgetState struct {
	remaining int64
	limit     int64
	resetAt   time.Time
}

func (bs *budgetState) update(remaining int64, limit int64, resetAt time.Time) {
	if bs.remaining < 0 || remaining < bs.remaining {
		bs.remaining = remaining
	}
	if bs.limit < 0 || limit < bs.limit {
		bs.limit = limit
	}
	if bs.resetAt.IsZero() || resetAt.Before(bs.resetAt) {
		bs.resetAt = resetAt
	}
}

// checkBucket consumes from a bucket and returns a deny Decision if the budget is exhausted.
// On success it updates the budget state and returns nil.
func (l *LocalLimiter) checkBucket(key string, capacity int64, unit time.Duration, cost int64, now time.Time, bs *budgetState) *Decision {
	b := l.getOrCreateBucket(key, capacity, unit)
	allowed, remaining, resetAt := b.tryConsume(cost, now)
	if !allowed {
		return &Decision{
			Allowed:    false,
			Remaining:  remaining,
			Limit:      capacity,
			ResetAt:    resetAt,
			RetryAfter: time.Until(resetAt),
			Provider:   l.Name(),
		}
	}
	bs.update(remaining, capacity, resetAt)
	return nil
}

// Check evaluates all matching rules. If any rule's budget is exhausted,
// the request is denied.
func (l *LocalLimiter) Check(ctx Context) (*Decision, error) {
	now := time.Now()
	bs := budgetState{remaining: -1, limit: -1}

	for i := range l.rules {
		rule := &l.rules[i]
		if !l.ruleMatches(rule, ctx) {
			continue
		}

		if rule.RequestsPerUnit > 0 {
			key := l.bucketKey(rule.Name, ctx.UserID, ctx.Model, "rpm")
			if deny := l.checkBucket(key, int64(rule.RequestsPerUnit), rule.Unit, 1, now, &bs); deny != nil {
				return deny, nil
			}
		}

		if rule.TokensPerUnit > 0 {
			key := l.bucketKey(rule.Name, ctx.UserID, ctx.Model, "tpm")
			cost := max(int64(ctx.TokenCount), 0)
			if deny := l.checkBucket(key, int64(rule.TokensPerUnit), rule.Unit, cost, now, &bs); deny != nil {
				return deny, nil
			}
		}
	}

	if bs.remaining < 0 {
		bs.remaining = 0
	}
	if bs.limit < 0 {
		bs.limit = 0
	}

	return &Decision{
		Allowed:   true,
		Remaining: bs.remaining,
		Limit:     bs.limit,
		ResetAt:   bs.resetAt,
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
	if rule.Match.Group != "" && rule.Match.Group != "*" && !slices.Contains(ctx.Groups, rule.Match.Group) {
		return false
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
	case "month":
		return 30 * 24 * time.Hour // approximation: fixed 30-day window, not calendar month
	default:
		return time.Minute
	}
}
