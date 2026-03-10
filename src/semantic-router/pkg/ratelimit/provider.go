// Package ratelimit provides a pluggable rate limiting framework for the
// semantic router, mirroring the authz credential resolution architecture.
//
// The router can enforce rate limits from multiple sources:
//
//   - Envoy RLS: An external Envoy Rate Limit Service (gRPC) performs global
//     rate limiting using descriptors (user, model, groups). This is the
//     standard Envoy approach (envoyproxy/ratelimit).
//
//   - Local limiter: An in-process token-aware rate limiter inspired by the
//     Envoy AI Gateway's usage-based rate limiting. Supports both
//     requests-per-unit (RPM) and tokens-per-unit (TPM) budgets keyed by
//     user/group/model.
//
// The RateLimitResolver chains multiple providers using first-deny semantics:
// if any provider denies, the request is rejected with 429.
//
// Adding a new provider:
//  1. Implement the Provider interface
//  2. Register it in buildRateLimitResolver (extproc/router.go)
//  3. Add a config type string in knownRateLimitProviderTypes
package ratelimit

import "time"

// Provider is a source of rate limiting decisions.
// Implementations check whether a request should be allowed and report
// token usage after responses.
type Provider interface {
	// Name returns a human-readable name for logging (e.g., "envoy-ratelimit", "local-limiter").
	Name() string

	// Check determines whether the request described by ctx should be allowed.
	// Returns a Decision indicating allow/deny and quota metadata.
	// Errors indicate provider failures (network, etc.), not rate limit denials.
	Check(ctx Context) (*Decision, error)

	// Report records actual token usage after a response is received.
	// This enables token-based (TPM) rate limiting where the budget is
	// consumed based on actual LLM token usage, not request count alone.
	Report(ctx Context, usage TokenUsage) error
}

// Context carries the per-request information needed for rate limit evaluation.
type Context struct {
	UserID     string
	Groups     []string
	Model      string
	Headers    map[string]string
	TokenCount int // estimated input tokens from classification
}

// Decision is the result of a rate limit check.
type Decision struct {
	Allowed    bool
	Remaining  int64
	Limit      int64
	ResetAt    time.Time
	RetryAfter time.Duration
	Provider   string // which provider made this decision
}

// TokenUsage represents actual token consumption from an LLM response,
// following the OpenAI usage schema.
type TokenUsage struct {
	InputTokens  int
	OutputTokens int
	TotalTokens  int
}
