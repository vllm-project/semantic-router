package ratelimit

import (
	"strings"
	"sync"
	"sync/atomic"
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
	// pruneEvery is the shortest rule unit, how often expired buckets are
	// dropped; lastPrune is when they last were, in Unix nanoseconds.
	pruneEvery time.Duration
	lastPrune  atomic.Int64
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
	// pruned marks a bucket dropped from the limiter; the key's next bucket
	// counts its window.
	pruned bool
}

// NewLocalLimiter creates a local limiter with the given rules.
func NewLocalLimiter(rules []Rule) *LocalLimiter {
	l := &LocalLimiter{rules: rules}
	for _, rule := range rules {
		if l.pruneEvery == 0 || rule.Unit < l.pruneEvery {
			l.pruneEvery = rule.Unit
		}
	}
	return l
}

func (l *LocalLimiter) Name() string {
	return "local-limiter"
}

// Check evaluates all matching rules. If any rule's budget is exhausted,
// the request is denied and quota consumed from buckets checked earlier for
// this request is refunded, so denied requests do not burn budget.
func (l *LocalLimiter) Check(ctx Context) (*Decision, error) {
	now := time.Now()
	l.pruneExpired(now)

	var mostRestrictiveRemaining int64 = -1
	var mostRestrictiveLimit int64 = -1
	var earliestReset time.Time

	// Quota consumed so far for this request, refunded if a later bucket denies.
	type consumption struct {
		b         *bucket
		cost      int64
		windowEnd time.Time
	}
	var consumed []consumption
	refundConsumed := func() {
		for _, c := range consumed {
			c.b.refund(c.cost, c.windowEnd)
		}
	}

	for i := range l.rules {
		rule := &l.rules[i]
		if !l.ruleMatches(rule, ctx) {
			continue
		}

		if rule.RequestsPerUnit > 0 {
			key := l.bucketKey(rule.Name, ctx.UserID, ctx.Model, "rpm")
			b := l.lockedBucket(key, int64(rule.RequestsPerUnit), rule.Unit)
			allowed, remaining, resetAt := b.tryConsume(1, now)
			b.mu.Unlock()
			if !allowed {
				refundConsumed()
				return &Decision{
					Allowed:    false,
					Remaining:  remaining,
					Limit:      int64(rule.RequestsPerUnit),
					ResetAt:    resetAt,
					RetryAfter: time.Until(resetAt),
					Provider:   l.Name(),
				}, nil
			}
			consumed = append(consumed, consumption{b: b, cost: 1, windowEnd: resetAt})
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
			b := l.lockedBucket(key, int64(rule.TokensPerUnit), rule.Unit)
			cost := int64(ctx.TokenCount)
			if cost <= 0 {
				cost = 0
			}
			allowed, remaining, resetAt := b.tryConsume(cost, now)
			b.mu.Unlock()
			if !allowed {
				refundConsumed()
				return &Decision{
					Allowed:    false,
					Remaining:  remaining,
					Limit:      int64(rule.TokensPerUnit),
					ResetAt:    resetAt,
					RetryAfter: time.Until(resetAt),
					Provider:   l.Name(),
				}, nil
			}
			if cost > 0 {
				consumed = append(consumed, consumption{b: b, cost: cost, windowEnd: resetAt})
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

		// The Check() already consumed estimated input tokens.
		// Now add the output tokens (the part we didn't know during Check).
		additionalTokens := int64(usage.OutputTokens)
		if additionalTokens > 0 {
			b := l.lockedBucket(key, int64(rule.TokensPerUnit), rule.Unit)
			b.consumeOnly(additionalTokens, time.Now())
			b.mu.Unlock()
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

// lockedBucket returns key's bucket, locked. A bucket pruned between the
// lookup and the lock is no longer key's, so the lookup repeats.
func (l *LocalLimiter) lockedBucket(key string, limit int64, unit time.Duration) *bucket {
	for {
		b := l.getOrCreateBucket(key, limit, unit)
		b.mu.Lock()
		if !b.pruned {
			return b
		}
		b.mu.Unlock()
	}
}

// pruneExpired drops the buckets whose window has ended, at most once per
// shortest rule unit. An expired bucket carries nothing into its next
// window, so no decision changes; without this the limiter keeps one bucket
// for every user and model it ever saw.
func (l *LocalLimiter) pruneExpired(now time.Time) {
	last := l.lastPrune.Load()
	if now.UnixNano()-last < int64(l.pruneEvery) || !l.lastPrune.CompareAndSwap(last, now.UnixNano()) {
		return
	}
	l.buckets.Range(func(key, value any) bool {
		b := value.(*bucket)
		b.mu.Lock()
		if now.After(b.windowEnd) {
			b.pruned = true
			l.buckets.CompareAndDelete(key, b)
		}
		b.mu.Unlock()
		return true
	})
}

// tryConsume attempts to consume `cost` from the bucket, whose lock the
// caller holds. Returns (allowed, remaining, windowEnd).
func (b *bucket) tryConsume(cost int64, now time.Time) (bool, int64, time.Time) {
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

// consumeOnly decrements the bucket, whose lock the caller holds, without
// checking. Used for post-response token reporting where we must track usage
// even if it goes negative.
func (b *bucket) consumeOnly(cost int64, now time.Time) {
	if now.After(b.windowEnd) {
		b.count = 0
		b.windowEnd = now.Add(b.unit)
	}

	b.count += cost
}

// refund returns quota that was consumed earlier for a request that was
// ultimately denied. It is a no-op if the window has rolled over since the
// consumption, so a refund never credits a fresh window.
func (b *bucket) refund(cost int64, windowEnd time.Time) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if !b.windowEnd.Equal(windowEnd) {
		return
	}

	b.count -= cost
	if b.count < 0 {
		b.count = 0
	}
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
