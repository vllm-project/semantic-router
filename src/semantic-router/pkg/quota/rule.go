package quota

import (
	"errors"
	"fmt"
	"strings"
	"time"
)

var ErrInvalidRateLimitRule = errors.New("invalid rate-limit rule")

// Metric is the quantity consumed by a rate-limit rule.
type Metric string

const (
	MetricRequests           Metric = "requests"
	MetricInputTokens        Metric = "input_tokens"
	MetricOutputTokens       Metric = "output_tokens"
	MetricTotalTokens        Metric = "total_tokens"
	MetricServedInputTokens  Metric = "served_input_tokens"
	MetricServedOutputTokens Metric = "served_output_tokens"
	MetricServedTotalTokens  Metric = "served_total_tokens"
	MetricCost               Metric = "cost"
	MetricConcurrentRequests Metric = "concurrent_requests"
)

// Algorithm identifies the exact counter algorithm and therefore the rule's
// required parameter variant.
type Algorithm string

const (
	AlgorithmSlidingLog     Algorithm = "sliding_log"
	AlgorithmCalendarWindow Algorithm = "calendar_window"
	AlgorithmTokenBucket    Algorithm = "token_bucket"
	AlgorithmGCRA           Algorithm = "gcra"
	AlgorithmConcurrency    Algorithm = "concurrency"
)

// Accounting identifies when a metric is charged.
type Accounting string

const (
	AccountingRequest        Accounting = "request"
	AccountingResponseActual Accounting = "response_actual"
)

// Enforcement determines whether a rule can reject traffic.
type Enforcement string

const (
	EnforcementEnforce Enforcement = "enforce"
	EnforcementShadow  Enforcement = "shadow"
)

// CalendarPeriod is a calendar boundary interpreted in CalendarTimezone by
// the control-plane compiler.
type CalendarPeriod string

const (
	CalendarPeriodDay   CalendarPeriod = "day"
	CalendarPeriodMonth CalendarPeriod = "month"
)

// RateLimitRule is the store-independent discriminated rule union.
//
// WholeLimit is used by whole-unit sliding-log, calendar-window, and
// concurrency rules. CostLimit occupies the same public "limit" field for a
// cost rule. Token-bucket and GCRA rules use only their dedicated parameters.
// Pointers preserve the required distinction between absent and explicit zero.
type RateLimitRule struct {
	ID          string
	Metric      Metric
	Algorithm   Algorithm
	Accounting  Accounting
	Enforcement Enforcement
	Ordinal     int

	WholeLimit *QuotaInteger
	CostLimit  *CurrencyDecimal

	Window               time.Duration
	CalendarPeriod       CalendarPeriod
	CalendarTimezone     string
	BucketCapacity       *QuotaInteger
	RefillAmount         *QuotaInteger
	RefillPeriod         time.Duration
	GCRAEmissionInterval time.Duration
	GCRABurstTolerance   *QuotaInteger
}

// Validate enforces both the algorithm discriminant and the metric/accounting
// compatibility matrix. Fields belonging to another algorithm are rejected,
// not ignored.
func (r RateLimitRule) Validate() error {
	if err := validateOpaqueID("rule ID", r.ID); err != nil {
		return invalidRule("%s", err)
	}
	if r.Ordinal < 0 {
		return invalidRule("ordinal must be non-negative")
	}
	if !r.Metric.valid() {
		return invalidRule("unsupported metric %q", r.Metric)
	}
	if !r.Algorithm.valid() {
		return invalidRule("unsupported algorithm %q", r.Algorithm)
	}
	if !r.Accounting.valid() {
		return invalidRule("unsupported accounting %q", r.Accounting)
	}
	if !r.Enforcement.valid() {
		return invalidRule("unsupported enforcement %q", r.Enforcement)
	}
	if err := r.validateMetricAccounting(); err != nil {
		return err
	}
	return r.validateAlgorithmVariant()
}

func (r RateLimitRule) validateMetricAccounting() error {
	switch {
	case r.Metric == MetricRequests:
		if r.Accounting != AccountingRequest {
			return invalidRule("requests require request accounting")
		}
		if r.Algorithm == AlgorithmConcurrency {
			return invalidRule("requests do not support concurrency")
		}
	case r.Metric == MetricConcurrentRequests:
		if r.Accounting != AccountingRequest || r.Algorithm != AlgorithmConcurrency {
			return invalidRule("concurrent_requests require request accounting with concurrency")
		}
	case r.Metric.responseActual():
		if r.Accounting != AccountingResponseActual {
			return invalidRule("%s requires response_actual accounting", r.Metric)
		}
		if r.Algorithm != AlgorithmSlidingLog && r.Algorithm != AlgorithmCalendarWindow {
			return invalidRule("%s supports only sliding_log or calendar_window", r.Metric)
		}
	default:
		return invalidRule("unsupported metric %q", r.Metric)
	}
	return nil
}

func (r RateLimitRule) validateAlgorithmVariant() error {
	switch r.Algorithm {
	case AlgorithmSlidingLog:
		if err := r.validateLimit(); err != nil {
			return err
		}
		if r.Window <= 0 || r.Window%time.Second != 0 {
			return invalidRule("sliding_log window must be a positive whole number of seconds")
		}
		return r.rejectUnexpected(
			fieldSet{calendar: true, tokenBucket: true, gcra: true},
		)
	case AlgorithmCalendarWindow:
		if err := r.validateLimit(); err != nil {
			return err
		}
		if r.CalendarPeriod != CalendarPeriodDay && r.CalendarPeriod != CalendarPeriodMonth {
			return invalidRule("calendar_window period must be day or month")
		}
		if err := validateTimezone(r.CalendarTimezone); err != nil {
			return invalidRule("calendar_window timezone: %v", err)
		}
		return r.rejectUnexpected(
			fieldSet{window: true, tokenBucket: true, gcra: true},
		)
	case AlgorithmTokenBucket:
		if !positive(r.BucketCapacity) {
			return invalidRule("token_bucket capacity must be positive")
		}
		if !positive(r.RefillAmount) {
			return invalidRule("token_bucket refill amount must be positive")
		}
		if r.RefillPeriod <= 0 || r.RefillPeriod%time.Millisecond != 0 {
			return invalidRule("token_bucket refill period must be a positive whole number of milliseconds")
		}
		return r.rejectUnexpected(
			fieldSet{limit: true, window: true, calendar: true, gcra: true},
		)
	case AlgorithmGCRA:
		if r.GCRAEmissionInterval <= 0 || r.GCRAEmissionInterval%time.Microsecond != 0 {
			return invalidRule("gcra emission interval must be a positive whole number of microseconds")
		}
		if r.GCRABurstTolerance == nil {
			return invalidRule("gcra burst tolerance is required")
		}
		return r.rejectUnexpected(
			fieldSet{limit: true, window: true, calendar: true, tokenBucket: true},
		)
	case AlgorithmConcurrency:
		if r.Metric != MetricConcurrentRequests {
			return invalidRule("concurrency requires concurrent_requests")
		}
		if !positive(r.WholeLimit) || r.CostLimit != nil {
			return invalidRule("concurrency requires a positive whole-unit limit")
		}
		return r.rejectUnexpected(
			fieldSet{window: true, calendar: true, tokenBucket: true, gcra: true},
		)
	default:
		return invalidRule("unsupported algorithm %q", r.Algorithm)
	}
}

func (r RateLimitRule) validateLimit() error {
	if r.Metric == MetricCost {
		if r.WholeLimit != nil || r.CostLimit == nil || r.CostLimit.IsZero() {
			return invalidRule("cost requires a positive currency-decimal limit")
		}
		return nil
	}
	if !positive(r.WholeLimit) || r.CostLimit != nil {
		return invalidRule("%s requires a positive whole-unit limit", r.Metric)
	}
	return nil
}

type fieldSet struct {
	limit       bool
	window      bool
	calendar    bool
	tokenBucket bool
	gcra        bool
}

func (r RateLimitRule) rejectUnexpected(forbidden fieldSet) error {
	if forbidden.limit && (r.WholeLimit != nil || r.CostLimit != nil) {
		return invalidRule("%s does not accept a limit field", r.Algorithm)
	}
	if forbidden.window && r.Window != 0 {
		return invalidRule("%s does not accept a window", r.Algorithm)
	}
	if forbidden.calendar && (r.CalendarPeriod != "" || r.CalendarTimezone != "") {
		return invalidRule("%s does not accept calendar fields", r.Algorithm)
	}
	if forbidden.tokenBucket && (r.BucketCapacity != nil || r.RefillAmount != nil || r.RefillPeriod != 0) {
		return invalidRule("%s does not accept token-bucket fields", r.Algorithm)
	}
	if forbidden.gcra && (r.GCRAEmissionInterval != 0 || r.GCRABurstTolerance != nil) {
		return invalidRule("%s does not accept GCRA fields", r.Algorithm)
	}
	return nil
}

func (m Metric) valid() bool {
	switch m {
	case MetricRequests,
		MetricInputTokens,
		MetricOutputTokens,
		MetricTotalTokens,
		MetricServedInputTokens,
		MetricServedOutputTokens,
		MetricServedTotalTokens,
		MetricCost,
		MetricConcurrentRequests:
		return true
	default:
		return false
	}
}

func (m Metric) responseActual() bool {
	return m.valid() && m != MetricRequests && m != MetricConcurrentRequests
}

func (a Algorithm) valid() bool {
	switch a {
	case AlgorithmSlidingLog,
		AlgorithmCalendarWindow,
		AlgorithmTokenBucket,
		AlgorithmGCRA,
		AlgorithmConcurrency:
		return true
	default:
		return false
	}
}

func (a Accounting) valid() bool {
	return a == AccountingRequest || a == AccountingResponseActual
}

func (e Enforcement) valid() bool {
	return e == EnforcementEnforce || e == EnforcementShadow
}

func positive(value *QuotaInteger) bool {
	return value != nil && !value.IsZero()
}

func validateTimezone(value string) error {
	if value == "" {
		return errors.New("is required")
	}
	if strings.TrimSpace(value) != value || value == "Local" {
		return errors.New("must be a canonical IANA timezone name")
	}
	if _, err := time.LoadLocation(value); err != nil {
		return fmt.Errorf("unknown IANA timezone %q", value)
	}
	return nil
}

func invalidRule(format string, args ...any) error {
	return fmt.Errorf("%w: %s", ErrInvalidRateLimitRule, fmt.Sprintf(format, args...))
}
