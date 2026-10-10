package fallback

import (
	"encoding/json"
	"fmt"
	"slices"
	"time"
)

// FallbackOverride is a layer of fallback policy that a routing decision or a
// request-graph step sets over its recipe's policy: the policy's fields
// except the circuit breaker, which belongs to the backends. A field it
// leaves unset keeps the layer below; Enabled is tri-state.
type FallbackOverride struct {
	Enabled              *bool         `json:"enabled,omitempty" yaml:"enabled,omitempty"`
	MaxAttempts          int           `json:"max_attempts,omitempty" yaml:"max_attempts,omitempty"`
	TotalTimeout         time.Duration `json:"total_timeout,omitempty" yaml:"total_timeout,omitempty"`
	PerAttemptTimeout    time.Duration `json:"per_attempt_timeout,omitempty" yaml:"per_attempt_timeout,omitempty"`
	RetryableStatusCodes []int         `json:"retryable_status_codes,omitempty" yaml:"retryable_status_codes,omitempty"`
}

// Disabled is the override that turns fallback off for a call.
func Disabled() *FallbackOverride {
	off := false
	return &FallbackOverride{Enabled: &off}
}

type rawFallbackOverride struct {
	Enabled              *bool `json:"enabled,omitempty" yaml:"enabled,omitempty"`
	MaxAttempts          int   `json:"max_attempts,omitempty" yaml:"max_attempts,omitempty"`
	TotalTimeout         any   `json:"total_timeout,omitempty" yaml:"total_timeout,omitempty"`
	PerAttemptTimeout    any   `json:"per_attempt_timeout,omitempty" yaml:"per_attempt_timeout,omitempty"`
	RetryableStatusCodes []int `json:"retryable_status_codes,omitempty" yaml:"retryable_status_codes,omitempty"`
}

func (o *FallbackOverride) fromRaw(raw rawFallbackOverride) error {
	total, err := unmarshalDuration(raw.TotalTimeout)
	if err != nil {
		return fmt.Errorf("invalid total_timeout: %w", err)
	}
	perAttempt, err := unmarshalDuration(raw.PerAttemptTimeout)
	if err != nil {
		return fmt.Errorf("invalid per_attempt_timeout: %w", err)
	}
	*o = FallbackOverride{
		Enabled:              raw.Enabled,
		MaxAttempts:          raw.MaxAttempts,
		TotalTimeout:         total,
		PerAttemptTimeout:    perAttempt,
		RetryableStatusCodes: raw.RetryableStatusCodes,
	}
	return nil
}

// UnmarshalJSON reads durations as strings ("30s") or nanoseconds, as the
// recipe's policy does.
func (o *FallbackOverride) UnmarshalJSON(data []byte) error {
	var raw rawFallbackOverride
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	return o.fromRaw(raw)
}

// UnmarshalYAML reads durations as strings ("30s") or nanoseconds, as the
// recipe's policy does.
func (o *FallbackOverride) UnmarshalYAML(unmarshal func(interface{}) error) error {
	var raw rawFallbackOverride
	if err := unmarshal(&raw); err != nil {
		return err
	}
	return o.fromRaw(raw)
}

// Validate checks the override's own values. The policy it makes with the
// layers below is checked by FallbackPolicy.Validate.
func (o *FallbackOverride) Validate() error {
	if o == nil {
		return nil
	}
	switch {
	case o.MaxAttempts < 0:
		return fmt.Errorf("fallback max_attempts cannot be negative, got %d", o.MaxAttempts)
	case o.TotalTimeout < 0:
		return fmt.Errorf("fallback total_timeout cannot be negative, got %v", o.TotalTimeout)
	case o.PerAttemptTimeout < 0:
		return fmt.Errorf("fallback per_attempt_timeout cannot be negative, got %v", o.PerAttemptTimeout)
	}
	for _, code := range o.RetryableStatusCodes {
		if code < 100 || code > 599 {
			return fmt.Errorf("invalid retryable status code %d: must be between 100 and 599", code)
		}
	}
	return nil
}

// Resolve returns the policy one call runs under: p, the recipe's policy
// (which already carries the global one), with each override laid over it
// in turn, the least specific first: the decision's, then a request-graph
// step's. An override replaces the fields it sets, so the merge goes field by
// field from the most specific layer down.
func (p FallbackPolicy) Resolve(overrides ...*FallbackOverride) FallbackPolicy {
	p.RetryableStatusCodes = slices.Clone(p.RetryableStatusCodes)
	for _, o := range overrides {
		if o == nil {
			continue
		}
		if o.Enabled != nil {
			p.SetExplicitEnabled(*o.Enabled)
		}
		if o.MaxAttempts != 0 {
			p.MaxAttempts = o.MaxAttempts
		}
		if o.TotalTimeout != 0 {
			p.TotalTimeout = o.TotalTimeout
		}
		if o.PerAttemptTimeout != 0 {
			p.PerAttemptTimeout = o.PerAttemptTimeout
		}
		if len(o.RetryableStatusCodes) > 0 {
			p.RetryableStatusCodes = slices.Clone(o.RetryableStatusCodes)
		}
	}
	return p
}

// For returns the orchestrator of a call whose decision or request-graph
// step overrides the policy: the resolved policy, with the same circuit
// breakers, since those belong to the backends. Without an override it is o.
func (o *Orchestrator) For(overrides ...*FallbackOverride) *Orchestrator {
	if o == nil || !slices.ContainsFunc(overrides, func(override *FallbackOverride) bool { return override != nil }) {
		return o
	}
	policy := o.policy.Resolve(overrides...)
	return &Orchestrator{
		policy:         policy,
		classifier:     NewClassifier(policy),
		circuitBreaker: o.circuitBreaker,
		nowFunc:        o.nowFunc,
	}
}
