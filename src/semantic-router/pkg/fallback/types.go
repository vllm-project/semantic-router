// Package fallback provides bounded, protocol-safe upstream-error model fallback
// across hard-eligible candidates for LLM inference requests.
package fallback

import (
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"gopkg.in/yaml.v2"
)

var (
	_ yaml.Unmarshaler = (*FallbackPolicy)(nil)
	_ yaml.Unmarshaler = (*CircuitBreakerConfig)(nil)
)

// Common error definitions for fallback boundaries.
var (
	ErrFallbackDisabled         = errors.New("fallback policy is disabled")
	ErrMaxAttemptsExceeded      = errors.New("maximum fallback attempts exceeded")
	ErrTotalDeadlineExceeded    = errors.New("total fallback deadline exceeded")
	ErrNoEligibleCandidates     = errors.New("no unvisited eligible fallback candidates remain")
	ErrResponseAlreadyCommitted = errors.New("cannot fallback: response has already been committed to client")
	ErrNonIdempotentSideEffects = errors.New("cannot fallback: request resulted in non-idempotent tool side-effects")
	ErrBodyNotReplayable        = errors.New("cannot fallback: request body cannot be replayed")
	ErrNonRetryableError        = errors.New("upstream outcome is not retryable")
	ErrCircuitOpen              = errors.New("all fallback candidate backends have open circuit breakers")
)

// TriggerClass categorizes the class of upstream error.
type TriggerClass string

const (
	TriggerClassNone          TriggerClass = "none"
	TriggerClassConnection    TriggerClass = "connection_error"
	TriggerClassTimeout       TriggerClass = "timeout"
	TriggerClass5xx           TriggerClass = "upstream_5xx"
	TriggerClassRateLimit     TriggerClass = "rate_limit"
	TriggerClassProtocolError TriggerClass = "protocol_error"
)

// FallbackPolicy defines the bounded execution parameters for cross-model fallback.
type FallbackPolicy struct {
	Version              int                  `json:"version,omitempty" yaml:"version,omitempty"`
	Enabled              bool                 `json:"enabled" yaml:"enabled"`
	MaxAttempts          int                  `json:"max_attempts,omitempty" yaml:"max_attempts,omitempty"`
	TotalTimeout         time.Duration        `json:"total_timeout,omitempty" yaml:"total_timeout,omitempty"`
	PerAttemptTimeout    time.Duration        `json:"per_attempt_timeout,omitempty" yaml:"per_attempt_timeout,omitempty"`
	RetryableStatusCodes []int                `json:"retryable_status_codes,omitempty" yaml:"retryable_status_codes,omitempty"`
	CircuitBreaker       CircuitBreakerConfig `json:"circuit_breaker,omitempty" yaml:"circuit_breaker,omitempty"`

	rawEnabled *bool
}

// ExplicitEnabled returns whether the enabled field was explicitly configured (tri-state).
// Returns nil if enabled was omitted during unmarshaling or unset.
func (p FallbackPolicy) ExplicitEnabled() *bool {
	if p.rawEnabled == nil {
		return nil
	}
	val := *p.rawEnabled
	return &val
}

// SetExplicitEnabled explicitly marks the enabled field as set to the given value,
// distinguishing an intentional false from an omitted/unset zero value.
func (p *FallbackPolicy) SetExplicitEnabled(enabled bool) {
	p.Enabled = enabled
	p.rawEnabled = &enabled
}

// WithExplicitEnabled returns a copy of FallbackPolicy with enabled explicitly marked.
func (p FallbackPolicy) WithExplicitEnabled(enabled bool) FallbackPolicy {
	p.SetExplicitEnabled(enabled)
	return p
}

func (p *FallbackPolicy) fromRaw(
	version int,
	enabled *bool,
	maxAttempts int,
	rawTotalTimeout any,
	rawPerAttemptTimeout any,
	retryableStatusCodes []int,
	cb CircuitBreakerConfig,
) error {
	p.Version = version
	p.rawEnabled = enabled
	if enabled != nil {
		p.Enabled = *enabled
	}
	p.MaxAttempts = maxAttempts
	p.RetryableStatusCodes = retryableStatusCodes
	p.CircuitBreaker = cb

	totalTimeout, err := unmarshalDuration(rawTotalTimeout)
	if err != nil {
		return fmt.Errorf("invalid total_timeout: %w", err)
	}
	p.TotalTimeout = totalTimeout

	perAttemptTimeout, err := unmarshalDuration(rawPerAttemptTimeout)
	if err != nil {
		return fmt.Errorf("invalid per_attempt_timeout: %w", err)
	}
	p.PerAttemptTimeout = perAttemptTimeout
	return nil
}

// UnmarshalJSON supports both string duration format ("30s") and numeric nanoseconds,
// preserving tri-state presence for enabled.
func (p *FallbackPolicy) UnmarshalJSON(data []byte) error {
	type rawFallbackPolicy struct {
		Version              int                  `json:"version,omitempty"`
		Enabled              *bool                `json:"enabled,omitempty"`
		MaxAttempts          int                  `json:"max_attempts,omitempty"`
		TotalTimeout         any                  `json:"total_timeout,omitempty"`
		PerAttemptTimeout    any                  `json:"per_attempt_timeout,omitempty"`
		RetryableStatusCodes []int                `json:"retryable_status_codes,omitempty"`
		CircuitBreaker       CircuitBreakerConfig `json:"circuit_breaker,omitempty"`
	}
	var raw rawFallbackPolicy
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	return p.fromRaw(raw.Version, raw.Enabled, raw.MaxAttempts, raw.TotalTimeout, raw.PerAttemptTimeout, raw.RetryableStatusCodes, raw.CircuitBreaker)
}

// UnmarshalYAML supports both string duration format ("30s") and numeric nanoseconds,
// preserving tri-state presence for enabled to support recipe inheritance.
func (p *FallbackPolicy) UnmarshalYAML(unmarshal func(interface{}) error) error {
	type rawFallbackPolicy struct {
		Version              int                  `yaml:"version,omitempty"`
		Enabled              *bool                `yaml:"enabled,omitempty"`
		MaxAttempts          int                  `yaml:"max_attempts,omitempty"`
		TotalTimeout         any                  `yaml:"total_timeout,omitempty"`
		PerAttemptTimeout    any                  `yaml:"per_attempt_timeout,omitempty"`
		RetryableStatusCodes []int                `yaml:"retryable_status_codes,omitempty"`
		CircuitBreaker       CircuitBreakerConfig `yaml:"circuit_breaker,omitempty"`
	}
	var raw rawFallbackPolicy
	if err := unmarshal(&raw); err != nil {
		return err
	}
	return p.fromRaw(raw.Version, raw.Enabled, raw.MaxAttempts, raw.TotalTimeout, raw.PerAttemptTimeout, raw.RetryableStatusCodes, raw.CircuitBreaker)
}

// DefaultPolicy returns the canonical v1 fallback policy configuration.
// By default, fallback is disabled (Enabled: false) for safe rollout across deployments.
func DefaultPolicy() FallbackPolicy {
	p := FallbackPolicy{
		Version:           1,
		Enabled:           false,
		MaxAttempts:       3,
		TotalTimeout:      30 * time.Second,
		PerAttemptTimeout: 10 * time.Second,
		RetryableStatusCodes: []int{
			502, // Bad Gateway
			503, // Service Unavailable
			504, // Gateway Timeout
		},
		CircuitBreaker: DefaultCircuitBreakerConfig(),
	}
	p.SetExplicitEnabled(false)
	return p
}

// DefaultEnabledPolicy returns the default policy with Enabled set to true, useful for tests.
func DefaultEnabledPolicy() FallbackPolicy {
	p := DefaultPolicy()
	p.SetExplicitEnabled(true)
	return p
}

// Inherit returns a copy of FallbackPolicy with unconfigured fields inherited from base.
func (p FallbackPolicy) Inherit(base FallbackPolicy) FallbackPolicy {
	if p.Version == 0 {
		p.Version = base.Version
	}
	if p.rawEnabled != nil {
		p.Enabled = *p.rawEnabled
	} else if !p.Enabled {
		p.Enabled = base.Enabled
		if base.rawEnabled != nil {
			val := *base.rawEnabled
			p.rawEnabled = &val
		} else if base.Enabled {
			t := true
			p.rawEnabled = &t
		}
	}
	if p.MaxAttempts == 0 {
		p.MaxAttempts = base.MaxAttempts
	}
	if p.TotalTimeout == 0 {
		p.TotalTimeout = base.TotalTimeout
	}
	if p.PerAttemptTimeout == 0 {
		p.PerAttemptTimeout = base.PerAttemptTimeout
	}
	if len(p.RetryableStatusCodes) == 0 && len(base.RetryableStatusCodes) > 0 {
		p.RetryableStatusCodes = append([]int(nil), base.RetryableStatusCodes...)
	}
	if p.CircuitBreaker.ConsecutiveFailures == 0 {
		p.CircuitBreaker.ConsecutiveFailures = base.CircuitBreaker.ConsecutiveFailures
	}
	if p.CircuitBreaker.CooldownPeriod == 0 {
		p.CircuitBreaker.CooldownPeriod = base.CircuitBreaker.CooldownPeriod
	}
	if p.CircuitBreaker.HalfOpenProbes == 0 {
		p.CircuitBreaker.HalfOpenProbes = base.CircuitBreaker.HalfOpenProbes
	}
	return p
}

// WithDefaults returns a copy of FallbackPolicy with unconfigured fields filled from defaults.
func (p FallbackPolicy) WithDefaults() FallbackPolicy {
	return p.Inherit(DefaultPolicy())
}

// Clone returns a deep copy of the FallbackPolicy.
func (p *FallbackPolicy) Clone() *FallbackPolicy {
	if p == nil {
		return nil
	}
	cp := *p
	if p.RetryableStatusCodes != nil {
		cp.RetryableStatusCodes = append([]int(nil), p.RetryableStatusCodes...)
	}
	if p.rawEnabled != nil {
		val := *p.rawEnabled
		cp.rawEnabled = &val
	}
	return &cp
}

// Validate checks that the fallback policy contains valid, well-formed configuration values.
func (p FallbackPolicy) Validate() error {
	if p.Version < 0 || (p.Version > 0 && p.Version != 1) {
		return fmt.Errorf("unsupported fallback policy version %d: version 1 is required", p.Version)
	}
	if p.Enabled {
		if p.MaxAttempts < 1 {
			return fmt.Errorf("fallback max_attempts must be >= 1 when enabled, got %d", p.MaxAttempts)
		}
	} else {
		if p.MaxAttempts < 0 {
			return fmt.Errorf("fallback max_attempts cannot be negative, got %d", p.MaxAttempts)
		}
	}
	if p.TotalTimeout < 0 {
		return fmt.Errorf("fallback total_timeout cannot be negative, got %v", p.TotalTimeout)
	}
	if p.PerAttemptTimeout < 0 {
		return fmt.Errorf("fallback per_attempt_timeout cannot be negative, got %v", p.PerAttemptTimeout)
	}
	if p.TotalTimeout > 0 && p.PerAttemptTimeout > 0 && p.PerAttemptTimeout > p.TotalTimeout {
		return fmt.Errorf("fallback per_attempt_timeout (%v) cannot exceed total_timeout (%v)", p.PerAttemptTimeout, p.TotalTimeout)
	}
	for _, code := range p.RetryableStatusCodes {
		if code < 100 || code > 599 {
			return fmt.Errorf("invalid retryable status code %d: must be between 100 and 599", code)
		}
	}
	if err := p.CircuitBreaker.Validate(); err != nil {
		return fmt.Errorf("circuit_breaker: %w", err)
	}
	return nil
}

// CircuitBreakerConfig controls per-backend backoff to avoid hammering flapping backends.
type CircuitBreakerConfig struct {
	ConsecutiveFailures int           `json:"consecutive_failures,omitempty" yaml:"consecutive_failures,omitempty"`
	CooldownPeriod      time.Duration `json:"cooldown_period,omitempty" yaml:"cooldown_period,omitempty"`
	HalfOpenProbes      int           `json:"half_open_probes,omitempty" yaml:"half_open_probes,omitempty"`
}

// UnmarshalJSON supports both string duration format ("30s") and numeric nanoseconds.
func (cb *CircuitBreakerConfig) UnmarshalJSON(data []byte) error {
	type rawCircuitBreakerConfig struct {
		ConsecutiveFailures int `json:"consecutive_failures,omitempty"`
		CooldownPeriod      any `json:"cooldown_period,omitempty"`
		HalfOpenProbes      int `json:"half_open_probes,omitempty"`
	}
	var raw rawCircuitBreakerConfig
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	cb.ConsecutiveFailures = raw.ConsecutiveFailures
	cb.HalfOpenProbes = raw.HalfOpenProbes
	cooldown, err := unmarshalDuration(raw.CooldownPeriod)
	if err != nil {
		return fmt.Errorf("invalid cooldown_period: %w", err)
	}
	cb.CooldownPeriod = cooldown
	return nil
}

// UnmarshalYAML supports both string duration format ("30s") and numeric nanoseconds.
func (cb *CircuitBreakerConfig) UnmarshalYAML(unmarshal func(interface{}) error) error {
	type rawCircuitBreakerConfig struct {
		ConsecutiveFailures int `yaml:"consecutive_failures,omitempty"`
		CooldownPeriod      any `yaml:"cooldown_period,omitempty"`
		HalfOpenProbes      int `yaml:"half_open_probes,omitempty"`
	}
	var raw rawCircuitBreakerConfig
	if err := unmarshal(&raw); err != nil {
		return err
	}
	cb.ConsecutiveFailures = raw.ConsecutiveFailures
	cb.HalfOpenProbes = raw.HalfOpenProbes
	cooldown, err := unmarshalDuration(raw.CooldownPeriod)
	if err != nil {
		return fmt.Errorf("invalid cooldown_period: %w", err)
	}
	cb.CooldownPeriod = cooldown
	return nil
}

// Validate checks that the circuit breaker configuration contains valid parameters.
func (cb CircuitBreakerConfig) Validate() error {
	if cb.ConsecutiveFailures < 0 {
		return fmt.Errorf("consecutive_failures cannot be negative, got %d", cb.ConsecutiveFailures)
	}
	if cb.CooldownPeriod < 0 {
		return fmt.Errorf("cooldown_period cannot be negative, got %v", cb.CooldownPeriod)
	}
	if cb.HalfOpenProbes < 0 {
		return fmt.Errorf("half_open_probes cannot be negative, got %d", cb.HalfOpenProbes)
	}
	return nil
}

// DefaultCircuitBreakerConfig returns the default circuit breaker settings.
func DefaultCircuitBreakerConfig() CircuitBreakerConfig {
	return CircuitBreakerConfig{
		ConsecutiveFailures: 3,
		CooldownPeriod:      30 * time.Second,
		HalfOpenProbes:      1,
	}
}

func unmarshalDuration(v any) (time.Duration, error) {
	switch val := v.(type) {
	case string:
		if val == "" {
			return 0, nil
		}
		return time.ParseDuration(val)
	case float64:
		return time.Duration(val), nil
	case int64:
		return time.Duration(val), nil
	case int:
		return time.Duration(val), nil
	case nil:
		return 0, nil
	default:
		return 0, fmt.Errorf("cannot parse %T as duration", v)
	}
}

// CommitState tracks the safety boundaries of an in-flight request.
type CommitState struct {
	// ResponseCommitted is true when headers or response bytes have already been sent to client.
	ResponseCommitted bool

	// HasNonIdempotentSideEffects is true when upstream executed non-idempotent tool calls.
	HasNonIdempotentSideEffects bool

	// BodyReplayable is true when the neutral request body is buffered and available for replay.
	BodyReplayable bool
}

// AttemptOutcome records the detailed execution result of one model attempt.
type AttemptOutcome struct {
	AttemptID         string        `json:"attempt_id"`
	Ordinal           int           `json:"ordinal"`
	Model             string        `json:"model"`
	Backend           string        `json:"backend"`
	ProviderRequestID string        `json:"provider_request_id,omitempty"`
	StatusCode        int           `json:"status_code"`
	Error             error         `json:"-"`
	ErrorMessage      string        `json:"error,omitempty"`
	TriggerClass      TriggerClass  `json:"trigger_class,omitempty"`
	Retryable         bool          `json:"retryable"`
	Duration          time.Duration `json:"duration_ns"`
	PromptTokens      int           `json:"prompt_tokens"`
	CompletionTokens  int           `json:"completion_tokens"`
	TotalTokens       int           `json:"total_tokens"`
	Cost              float64       `json:"cost,omitempty"`
	Currency          string        `json:"currency,omitempty"`
	Discarded         bool          `json:"discarded"`
}

// TokenUsageSummary groups billable vs discarded token counts across attempts.
type TokenUsageSummary struct {
	BillablePromptTokens      int `json:"billable_prompt_tokens"`
	BillableCompletionTokens  int `json:"billable_completion_tokens"`
	BillableTotalTokens       int `json:"billable_total_tokens"`
	DiscardedPromptTokens     int `json:"discarded_prompt_tokens"`
	DiscardedCompletionTokens int `json:"discarded_completion_tokens"`
	DiscardedTotalTokens      int `json:"discarded_total_tokens"`
}

// ExecutionRecord tracks the entire lifecycle of a request across all its attempts.
type ExecutionRecord struct {
	RequestID      string            `json:"request_id"`
	SessionID      string            `json:"session_id,omitempty"`
	ConversationID string            `json:"conversation_id,omitempty"`
	DecisionName   string            `json:"decision_name,omitempty"`
	InitialModel   string            `json:"initial_model"`
	SelectedModel  string            `json:"selected_model,omitempty"`
	Attempts       []AttemptOutcome  `json:"attempts"`
	StartTime      time.Time         `json:"start_time"`
	TotalDuration  time.Duration     `json:"total_duration_ns"`
	FinalStatus    string            `json:"final_status"`
	UsageSummary   TokenUsageSummary `json:"usage_summary"`
	VisitedModels  []string          `json:"visited_models"`
}
