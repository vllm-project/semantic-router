package accesscontrol

import (
	"regexp"
	"strconv"
	"strings"
	"time"
)

type RateMetric string

const (
	RateMetricRequests           RateMetric = "requests"
	RateMetricInputTokens        RateMetric = "input_tokens"
	RateMetricOutputTokens       RateMetric = "output_tokens"
	RateMetricTotalTokens        RateMetric = "total_tokens"
	RateMetricConcurrentRequests RateMetric = "concurrent_requests"
	RateMetricServedInputTokens  RateMetric = "served_input_tokens"
	RateMetricServedOutputTokens RateMetric = "served_output_tokens"
	RateMetricServedTotalTokens  RateMetric = "served_total_tokens"
	RateMetricCost               RateMetric = "cost"
)

func (m RateMetric) Valid() bool {
	switch m {
	case RateMetricRequests, RateMetricInputTokens, RateMetricOutputTokens,
		RateMetricTotalTokens, RateMetricConcurrentRequests,
		RateMetricServedInputTokens, RateMetricServedOutputTokens,
		RateMetricServedTotalTokens, RateMetricCost:
		return true
	default:
		return false
	}
}

func (m RateMetric) actualResponseMetric() bool {
	switch m {
	case RateMetricInputTokens, RateMetricOutputTokens, RateMetricTotalTokens,
		RateMetricServedInputTokens, RateMetricServedOutputTokens,
		RateMetricServedTotalTokens, RateMetricCost:
		return true
	default:
		return false
	}
}

type RateAlgorithm string

const (
	RateAlgorithmSlidingLog     RateAlgorithm = "sliding_log"
	RateAlgorithmCalendarWindow RateAlgorithm = "calendar_window"
	RateAlgorithmTokenBucket    RateAlgorithm = "token_bucket"
	RateAlgorithmGCRA           RateAlgorithm = "gcra"
	RateAlgorithmConcurrency    RateAlgorithm = "concurrency"
)

func (a RateAlgorithm) Valid() bool {
	switch a {
	case RateAlgorithmSlidingLog, RateAlgorithmCalendarWindow,
		RateAlgorithmTokenBucket, RateAlgorithmGCRA, RateAlgorithmConcurrency:
		return true
	default:
		return false
	}
}

type RateAccounting string

const (
	RateAccountingRequest        RateAccounting = "request"
	RateAccountingResponseActual RateAccounting = "response_actual"
)

func (a RateAccounting) Valid() bool {
	return a == RateAccountingRequest || a == RateAccountingResponseActual
}

type RateEnforcement string

const (
	RateEnforcementEnforce RateEnforcement = "enforce"
	RateEnforcementShadow  RateEnforcement = "shadow"
)

func (e RateEnforcement) Valid() bool {
	return e == RateEnforcementEnforce || e == RateEnforcementShadow
}

type CalendarPeriod string

const (
	CalendarPeriodDay   CalendarPeriod = "day"
	CalendarPeriodMonth CalendarPeriod = "month"
)

func (p CalendarPeriod) Valid() bool {
	return p == CalendarPeriodDay || p == CalendarPeriodMonth
}

// QuotaValue is an exact decimal string. Whole-unit metrics use an integer;
// cost may use up to 15 fractional digits. Runtime compilation converts this
// value to the fixed-width counter representation without float arithmetic.
type QuotaValue string

type RateLimitRule struct {
	ID                   RateLimitRuleID
	PolicyID             RateLimitPolicyID
	Metric               RateMetric
	Algorithm            RateAlgorithm
	Limit                QuotaValue
	Window               time.Duration
	CalendarPeriod       CalendarPeriod
	Timezone             string
	BucketCapacity       QuotaValue
	RefillAmount         QuotaValue
	RefillPeriod         time.Duration
	GCRAEmissionInterval time.Duration
	GCRABurstTolerance   *int64
	Accounting           RateAccounting
	Enforcement          RateEnforcement
	Ordinal              uint32
}

func (r RateLimitRule) Validate() error {
	errs := []error{
		validateRequired("id", string(r.ID)),
		validateRequired("policy_id", string(r.PolicyID)),
	}
	if !r.Metric.Valid() {
		errs = append(errs, invalid("metric", "is not a supported rate metric"))
	}
	if !r.Algorithm.Valid() {
		errs = append(errs, invalid("algorithm", "is not a supported rate algorithm"))
	}
	if !r.Accounting.Valid() {
		errs = append(errs, invalid("accounting", "is not request or response_actual"))
	}
	if !r.Enforcement.Valid() {
		errs = append(errs, invalid("enforcement", "is not enforce or shadow"))
	}
	errs = append(errs, r.validateAlgorithm(), r.validateMetricMatrix())
	return joinValidation(errs...)
}

func (r RateLimitRule) validateAlgorithm() error {
	switch r.Algorithm {
	case RateAlgorithmSlidingLog:
		return joinValidation(
			r.validatePositiveLimit(),
			requireDurationMultiple("window", r.Window, time.Second),
			requireEmptyCalendar(r),
			requireEmptyBucket(r),
			requireEmptyGCRA(r),
		)
	case RateAlgorithmCalendarWindow:
		var periodErr, timezoneErr error
		if !r.CalendarPeriod.Valid() {
			periodErr = invalid("calendar_period", "must be day or month")
		}
		if strings.TrimSpace(r.Timezone) == "" {
			timezoneErr = invalid("timezone", "must be an IANA timezone")
		} else if _, err := time.LoadLocation(r.Timezone); err != nil {
			timezoneErr = invalid("timezone", "must be an IANA timezone")
		}
		return joinValidation(
			r.validatePositiveLimit(),
			periodErr,
			timezoneErr,
			requireZeroDuration("window", r.Window),
			requireEmptyBucket(r),
			requireEmptyGCRA(r),
		)
	case RateAlgorithmTokenBucket:
		return joinValidation(
			requireEmpty("limit", r.Limit),
			requireZeroDuration("window", r.Window),
			requireEmptyCalendar(r),
			validatePositiveInteger("bucket_capacity", r.BucketCapacity),
			validatePositiveInteger("refill_amount", r.RefillAmount),
			requireDurationMultiple("refill_period", r.RefillPeriod, time.Millisecond),
			requireEmptyGCRA(r),
		)
	case RateAlgorithmGCRA:
		return joinValidation(
			requireEmpty("limit", r.Limit),
			requireZeroDuration("window", r.Window),
			requireEmptyCalendar(r),
			requireEmptyBucket(r),
			requireDurationMultiple("gcra_emission_interval", r.GCRAEmissionInterval, time.Microsecond),
			validateNonNegativeInt64("gcra_burst_tolerance", r.GCRABurstTolerance),
		)
	case RateAlgorithmConcurrency:
		return joinValidation(
			r.validatePositiveLimit(),
			requireZeroDuration("window", r.Window),
			requireEmptyCalendar(r),
			requireEmptyBucket(r),
			requireEmptyGCRA(r),
		)
	default:
		return nil
	}
}

func (r RateLimitRule) validateMetricMatrix() error {
	switch {
	case r.Metric == RateMetricRequests:
		if r.Accounting != RateAccountingRequest || r.Algorithm == RateAlgorithmConcurrency {
			return invalid("metric", "requests requires request accounting and a non-concurrency algorithm")
		}
	case r.Metric.actualResponseMetric():
		if r.Accounting != RateAccountingResponseActual ||
			(r.Algorithm != RateAlgorithmSlidingLog && r.Algorithm != RateAlgorithmCalendarWindow) {
			return invalid("metric", "token and cost metrics require response_actual accounting with sliding_log or calendar_window")
		}
	case r.Metric == RateMetricConcurrentRequests:
		if r.Accounting != RateAccountingRequest || r.Algorithm != RateAlgorithmConcurrency {
			return invalid("metric", "concurrent_requests requires request accounting and concurrency")
		}
	}
	return nil
}

func (r RateLimitRule) validatePositiveLimit() error {
	if r.Metric == RateMetricCost {
		return validatePositiveCost("limit", r.Limit)
	}
	return validatePositiveInteger("limit", r.Limit)
}

var (
	positiveIntegerPattern = regexp.MustCompile(`^[1-9][0-9]{0,41}$`)
	positiveCostPattern    = regexp.MustCompile(`^(0|[1-9][0-9]*)(\.[0-9]{1,15})?$`)
)

func validatePositiveInteger(field string, value QuotaValue) error {
	if !positiveIntegerPattern.MatchString(string(value)) {
		return invalid(field, "must be a positive integer with at most 42 digits")
	}
	return nil
}

func validateNonNegativeInt64(field string, value *int64) error {
	if value == nil || *value < 0 {
		return invalid(field, "must be a non-negative integer")
	}
	return nil
}

func validatePositiveCost(field string, value QuotaValue) error {
	text := string(value)
	if !positiveCostPattern.MatchString(text) {
		return invalid(field, "must be a positive canonical decimal with at most 15 fractional digits")
	}
	digits := strings.ReplaceAll(text, ".", "")
	if len(digits) > 42 {
		return invalid(field, "must contain at most 42 total digits")
	}
	allZero := true
	for _, char := range digits {
		if char != '0' {
			allZero = false
			break
		}
	}
	if allZero {
		return invalid(field, "must be positive")
	}
	return nil
}

func requireEmpty(field string, value QuotaValue) error {
	if value != "" {
		return invalid(field, "must be empty for this algorithm")
	}
	return nil
}

func requirePositiveDuration(field string, value time.Duration) error {
	if value <= 0 {
		return invalid(field, "must be positive")
	}
	return nil
}

func requireDurationMultiple(field string, value, unit time.Duration) error {
	if err := requirePositiveDuration(field, value); err != nil {
		return err
	}
	if value%unit != 0 {
		return invalid(field, "must use "+unit.String()+" granularity")
	}
	return nil
}

func requireZeroDuration(field string, value time.Duration) error {
	if value != 0 {
		return invalid(field, "must be empty for this algorithm")
	}
	return nil
}

func requireEmptyCalendar(rule RateLimitRule) error {
	var periodErr, timezoneErr error
	if rule.CalendarPeriod != "" {
		periodErr = invalid("calendar_period", "must be empty for this algorithm")
	}
	if rule.Timezone != "" {
		timezoneErr = invalid("timezone", "must be empty for this algorithm")
	}
	return joinValidation(periodErr, timezoneErr)
}

func requireEmptyBucket(rule RateLimitRule) error {
	return joinValidation(
		requireEmpty("bucket_capacity", rule.BucketCapacity),
		requireEmpty("refill_amount", rule.RefillAmount),
		requireZeroDuration("refill_period", rule.RefillPeriod),
	)
}

func requireEmptyGCRA(rule RateLimitRule) error {
	return joinValidation(
		requireZeroDuration("gcra_emission_interval", rule.GCRAEmissionInterval),
		requireNil("gcra_burst_tolerance", rule.GCRABurstTolerance),
	)
}

func requireNil(field string, value *int64) error {
	if value != nil {
		return invalid(field, "must be empty for this algorithm")
	}
	return nil
}

type RateLimitPolicy struct {
	NamespaceID NamespaceID
	ID          RateLimitPolicyID
	DisplayName string
	Status      PolicyStatus
	Revision    Revision
	Rules       []RateLimitRule
	CreatedAt   time.Time
	UpdatedAt   time.Time
}

func (p RateLimitPolicy) Validate() error {
	var statusErr error
	if !p.Status.Valid() {
		statusErr = invalid("status", "is not a valid policy status")
	}
	errs := []error{
		validateRequired("namespace_id", string(p.NamespaceID)),
		validateRequired("id", string(p.ID)),
		validateRequired("display_name", p.DisplayName),
		statusErr,
		validateRevision(p.Revision),
		validateTimestamps(p.CreatedAt, p.UpdatedAt),
	}
	seenIDs := make(map[RateLimitRuleID]struct{}, len(p.Rules))
	seenOrdinals := make(map[uint32]struct{}, len(p.Rules))
	for index, rule := range p.Rules {
		if rule.PolicyID != p.ID {
			errs = append(errs, invalid("rules", "rule policy_id must match its parent policy"))
		}
		if err := rule.Validate(); err != nil {
			errs = append(errs, invalid("rules", "rule at index "+strconv.Itoa(index)+": "+err.Error()))
		}
		if _, exists := seenIDs[rule.ID]; exists {
			errs = append(errs, invalid("rules", "contains a duplicate rule id"))
		}
		if _, exists := seenOrdinals[rule.Ordinal]; exists {
			errs = append(errs, invalid("rules", "contains a duplicate ordinal"))
		}
		seenIDs[rule.ID] = struct{}{}
		seenOrdinals[rule.Ordinal] = struct{}{}
	}
	return joinValidation(errs...)
}
