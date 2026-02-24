package ratelimit

import (
	"fmt"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// RateLimitResolver chains multiple Providers and evaluates rate limits
// using first-deny semantics: every provider is checked and if any provider
// denies, the request is rejected with a 429.
//
// Typical chain order:
//  1. EnvoyRLSProvider  (global limits via external Envoy Rate Limit Service)
//  2. LocalLimiter      (in-process per-user/model token and request budgets)
//
// Security modes:
//   - fail-closed (failOpen=false, default): provider errors cause rejection.
//   - fail-open   (failOpen=true): provider errors are logged but the request
//     is allowed through. Use only when availability > rate limit accuracy.
type RateLimitResolver struct {
	providers []Provider
	failOpen  bool
}

// NewRateLimitResolver creates a resolver with the given provider chain.
// By default the resolver is fail-closed (failOpen=false).
func NewRateLimitResolver(providers ...Provider) *RateLimitResolver {
	return &RateLimitResolver{providers: providers, failOpen: false}
}

// SetFailOpen configures whether the resolver allows requests through when
// a provider returns an error (not a denial — denials always block).
func (r *RateLimitResolver) SetFailOpen(failOpen bool) {
	if r != nil {
		r.failOpen = failOpen
	}
}

// FailOpen returns whether the resolver is in fail-open mode.
func (r *RateLimitResolver) FailOpen() bool {
	if r == nil {
		return false
	}
	return r.failOpen
}

// Check evaluates all providers. Returns a merged Decision.
//
//   - If any provider denies: returns denied Decision immediately.
//   - If all providers allow: returns allowed Decision with the most
//     restrictive remaining quota.
//   - If a provider errors: behavior depends on failOpen.
//
// The returned Decision aggregates the lowest Remaining/Limit across all
// providers for response header generation.
func (r *RateLimitResolver) Check(ctx Context) (*Decision, error) {
	if r == nil {
		return &Decision{Allowed: true}, nil
	}

	if len(r.providers) == 0 {
		return &Decision{Allowed: true}, nil
	}

	merged := &Decision{
		Allowed:   true,
		Remaining: -1, // sentinel: unset
		Limit:     -1,
	}

	tried := make([]string, 0, len(r.providers))
	for _, p := range r.providers {
		d, err := p.Check(ctx)
		if err != nil {
			tried = append(tried, p.Name())
			if r.failOpen {
				logging.Warnf("Rate limit provider %q error (fail_open=true, allowing): %v", p.Name(), err)
				continue
			}
			logging.Errorf("Rate limit provider %q error (fail_open=false, rejecting): %v", p.Name(), err)
			return &Decision{
				Allowed:    false,
				Provider:   p.Name(),
				RetryAfter: 5 * time.Second,
			}, fmt.Errorf("rate limit check failed at provider %q: %w", p.Name(), err)
		}

		tried = append(tried, p.Name())

		if !d.Allowed {
			logging.Infof("Rate limit DENIED by provider %q for user=%s model=%s (limit=%d, remaining=%d)",
				p.Name(), ctx.UserID, ctx.Model, d.Limit, d.Remaining)
			d.Provider = p.Name()
			return d, nil
		}

		// Merge: take the most restrictive values
		if merged.Remaining < 0 || d.Remaining < merged.Remaining {
			merged.Remaining = d.Remaining
		}
		if merged.Limit < 0 || d.Limit < merged.Limit {
			merged.Limit = d.Limit
		}
		if !d.ResetAt.IsZero() && (merged.ResetAt.IsZero() || d.ResetAt.Before(merged.ResetAt)) {
			merged.ResetAt = d.ResetAt
		}
	}

	if merged.Remaining < 0 {
		merged.Remaining = 0
	}
	if merged.Limit < 0 {
		merged.Limit = 0
	}

	logging.Debugf("Rate limit ALLOWED after checking [%s] for user=%s model=%s (remaining=%d)",
		strings.Join(tried, " → "), ctx.UserID, ctx.Model, merged.Remaining)
	return merged, nil
}

// Report forwards token usage to all providers for budget tracking.
// Errors are logged but not propagated (best-effort reporting).
func (r *RateLimitResolver) Report(ctx Context, usage TokenUsage) {
	if r == nil {
		return
	}
	for _, p := range r.providers {
		if err := p.Report(ctx, usage); err != nil {
			logging.Warnf("Rate limit provider %q report error: %v", p.Name(), err)
		}
	}
}

// ProviderNames returns the names of all registered providers (for logging).
func (r *RateLimitResolver) ProviderNames() []string {
	if r == nil {
		return nil
	}
	names := make([]string, len(r.providers))
	for i, p := range r.providers {
		names[i] = p.Name()
	}
	return names
}
