package config

import (
	"fmt"
	"time"
)

// What a request gets when its signals could not be read because the decision
// runtime refused their calls as overloaded: its admission queue was full
// (global.model_catalog.on_signal_overload.action).
const (
	// SignalOverloadRefuse is the default: a rule that fails closed refuses
	// the request, as on any failure, and one that fails open does not match.
	SignalOverloadRefuse = "refuse"
	// SignalOverloadRetry asks the overloaded calls again, for up to wait_ms,
	// and then answers the request a retryable error instead of a refusal or
	// a forward that rules it could not read would decide.
	SignalOverloadRetry = "retry"
)

const (
	maxSignalOverloadWaitMs = 60_000
	// maxSignalOverloadRetryAfterSeconds is the longest Retry-After the
	// clients of the Router honor: OpenAI's SDK and the AI SDK ignore one of
	// 60 s or more, and Codex sleeps the whole of it.
	maxSignalOverloadRetryAfterSeconds = 60
)

// SignalOverloadPolicy is global.model_catalog.on_signal_overload. With
// action retry, every request-stage call the decision runtime refuses as
// overloaded is asked again, whatever its rules' failure policy, for up to
// WaitMs within the signals' deadline; a rule's own timeout_ms bounds each
// attempt, not the wait. If the runtime still refuses, a request that only
// the rules it could not read refuse, or that one of them would refuse had it
// matched, is answered a retryable error with RetryAfterSeconds: a rule that
// fails open, unread for overload, keeps the request from being forwarded. A
// match still refuses, and the request is never forwarded unchecked.
type SignalOverloadPolicy struct {
	// Action is refuse (the default) or retry.
	Action string `yaml:"action,omitempty" json:"action,omitempty" jsonschema:"enum=refuse,enum=retry"`
	// WaitMs is how long a request waits for the decision runtime to have
	// capacity, asking again, before it gets the retryable error.
	WaitMs int `yaml:"wait_ms,omitempty" json:"wait_ms,omitempty" jsonschema:"minimum=0,maximum=60000"`
	// RetryAfterSeconds is the retryable error's Retry-After. 0 derives it
	// from WaitMs, rounded up to whole seconds and at least 1.
	RetryAfterSeconds int `yaml:"retry_after_seconds,omitempty" json:"retry_after_seconds,omitempty" jsonschema:"minimum=0,maximum=60"`
}

func cloneSignalOverloadPolicy(policy *SignalOverloadPolicy) *SignalOverloadPolicy {
	if policy == nil {
		return nil
	}
	cloned := *policy
	return &cloned
}

// SignalOverloadRetries reports whether the request stage waits for an
// overloaded decision runtime, and answers a retryable error instead of a
// refusal or a forward that rules it could not read decide.
func (c *RouterConfig) SignalOverloadRetries() bool {
	return c != nil && c.ModelSignalOverload != nil && c.ModelSignalOverload.Action == SignalOverloadRetry
}

// SignalOverloadWait is how long such a request waits for capacity first.
func (c *RouterConfig) SignalOverloadWait() time.Duration {
	if !c.SignalOverloadRetries() {
		return 0
	}
	return time.Duration(c.ModelSignalOverload.WaitMs) * time.Millisecond
}

// SignalOverloadRetryAfter is the retryable error's Retry-After in seconds.
func (c *RouterConfig) SignalOverloadRetryAfter() int {
	if c == nil || c.ModelSignalOverload == nil {
		return 1
	}
	if seconds := c.ModelSignalOverload.RetryAfterSeconds; seconds > 0 {
		return seconds
	}
	return max(1, (c.ModelSignalOverload.WaitMs+999)/1000)
}

func validateModelSignalOverloadContracts(cfg *RouterConfig) error {
	policy := cfg.ModelSignalOverload
	if policy == nil {
		return nil
	}
	switch policy.Action {
	case "", SignalOverloadRefuse, SignalOverloadRetry:
	default:
		return fmt.Errorf("global.model_catalog.on_signal_overload.action must be refuse or retry")
	}
	if policy.WaitMs < 0 || policy.WaitMs > maxSignalOverloadWaitMs {
		return fmt.Errorf("global.model_catalog.on_signal_overload.wait_ms must be between 0 and %d", maxSignalOverloadWaitMs)
	}
	if policy.RetryAfterSeconds < 0 || policy.RetryAfterSeconds > maxSignalOverloadRetryAfterSeconds {
		return fmt.Errorf("global.model_catalog.on_signal_overload.retry_after_seconds must be between 0 and %d", maxSignalOverloadRetryAfterSeconds)
	}
	if policy.Action != SignalOverloadRetry && (policy.WaitMs != 0 || policy.RetryAfterSeconds != 0) {
		return fmt.Errorf("global.model_catalog.on_signal_overload.wait_ms and retry_after_seconds apply only to action retry")
	}
	return nil
}
