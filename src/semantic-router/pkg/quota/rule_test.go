package quota

import (
	"errors"
	"testing"
	"time"
)

func TestRateLimitRuleValidVariants(t *testing.T) {
	t.Parallel()

	wholeLimit := quotaIntegerPointer(t, "12")
	costLimit := currencyDecimalPointer(t, "5.25")
	zero := quotaIntegerPointer(t, "0")
	tests := []struct {
		name string
		rule RateLimitRule
	}{
		{
			name: "request sliding log",
			rule: baseRule(MetricRequests, AlgorithmSlidingLog, AccountingRequest, wholeLimit),
		},
		{
			name: "actual tokens sliding log",
			rule: baseRule(MetricTotalTokens, AlgorithmSlidingLog, AccountingResponseActual, wholeLimit),
		},
		{
			name: "cost sliding log",
			rule: func() RateLimitRule {
				rule := baseRule(MetricCost, AlgorithmSlidingLog, AccountingResponseActual, nil)
				rule.CostLimit = costLimit
				return rule
			}(),
		},
		{
			name: "calendar day",
			rule: func() RateLimitRule {
				rule := baseRule(MetricServedOutputTokens, AlgorithmCalendarWindow, AccountingResponseActual, wholeLimit)
				rule.Window = 0
				rule.CalendarPeriod = CalendarPeriodDay
				rule.CalendarTimezone = "UTC"
				return rule
			}(),
		},
		{
			name: "calendar month",
			rule: func() RateLimitRule {
				rule := baseRule(MetricRequests, AlgorithmCalendarWindow, AccountingRequest, wholeLimit)
				rule.Window = 0
				rule.CalendarPeriod = CalendarPeriodMonth
				rule.CalendarTimezone = "America/Los_Angeles"
				return rule
			}(),
		},
		{
			name: "token bucket",
			rule: RateLimitRule{
				ID: "rule-token-bucket", Metric: MetricRequests, Algorithm: AlgorithmTokenBucket,
				Accounting: AccountingRequest, Enforcement: EnforcementEnforce,
				BucketCapacity: quotaIntegerPointer(t, "100"), RefillAmount: quotaIntegerPointer(t, "10"),
				RefillPeriod: 250 * time.Millisecond,
			},
		},
		{
			name: "gcra zero burst tolerance",
			rule: RateLimitRule{
				ID: "rule-gcra", Metric: MetricRequests, Algorithm: AlgorithmGCRA,
				Accounting: AccountingRequest, Enforcement: EnforcementShadow,
				GCRAEmissionInterval: 500 * time.Microsecond, GCRABurstTolerance: zero,
			},
		},
		{
			name: "concurrency",
			rule: RateLimitRule{
				ID: "rule-concurrency", Metric: MetricConcurrentRequests, Algorithm: AlgorithmConcurrency,
				Accounting: AccountingRequest, Enforcement: EnforcementEnforce, WholeLimit: wholeLimit,
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if err := test.rule.Validate(); err != nil {
				t.Fatalf("Validate() error = %v", err)
			}
		})
	}
}

func TestRateLimitRuleMetricMatrix(t *testing.T) {
	t.Parallel()

	wholeLimit := quotaIntegerPointer(t, "12")
	actualMetrics := []Metric{
		MetricInputTokens,
		MetricOutputTokens,
		MetricTotalTokens,
		MetricServedInputTokens,
		MetricServedOutputTokens,
		MetricServedTotalTokens,
	}
	for _, metric := range actualMetrics {
		t.Run(string(metric), func(t *testing.T) {
			t.Parallel()
			rule := baseRule(metric, AlgorithmSlidingLog, AccountingResponseActual, wholeLimit)
			if err := rule.Validate(); err != nil {
				t.Fatalf("Validate() error = %v", err)
			}
			rule.Accounting = AccountingRequest
			if err := rule.Validate(); !errors.Is(err, ErrInvalidRateLimitRule) {
				t.Fatalf("request accounting error = %v, want %v", err, ErrInvalidRateLimitRule)
			}
			rule.Accounting = AccountingResponseActual
			rule.Algorithm = AlgorithmTokenBucket
			rule.Window = 0
			rule.WholeLimit = nil
			rule.BucketCapacity = quotaIntegerPointer(t, "10")
			rule.RefillAmount = quotaIntegerPointer(t, "1")
			rule.RefillPeriod = time.Second
			if err := rule.Validate(); !errors.Is(err, ErrInvalidRateLimitRule) {
				t.Fatalf("token bucket error = %v, want %v", err, ErrInvalidRateLimitRule)
			}
		})
	}
}

func TestRateLimitRuleRejectsInvalidVariants(t *testing.T) {
	t.Parallel()

	limit := quotaIntegerPointer(t, "12")
	zero := quotaIntegerPointer(t, "0")
	valid := baseRule(MetricRequests, AlgorithmSlidingLog, AccountingRequest, limit)
	tests := []struct {
		name   string
		mutate func(*RateLimitRule)
	}{
		{name: "missing ID", mutate: func(rule *RateLimitRule) { rule.ID = "" }},
		{name: "negative ordinal", mutate: func(rule *RateLimitRule) { rule.Ordinal = -1 }},
		{name: "unknown metric", mutate: func(rule *RateLimitRule) { rule.Metric = "bogus" }},
		{name: "unknown algorithm", mutate: func(rule *RateLimitRule) { rule.Algorithm = "bogus" }},
		{name: "unknown accounting", mutate: func(rule *RateLimitRule) { rule.Accounting = "bogus" }},
		{name: "unknown enforcement", mutate: func(rule *RateLimitRule) { rule.Enforcement = "bogus" }},
		{name: "zero limit", mutate: func(rule *RateLimitRule) { rule.WholeLimit = zero }},
		{name: "missing limit", mutate: func(rule *RateLimitRule) { rule.WholeLimit = nil }},
		{name: "subsecond sliding window", mutate: func(rule *RateLimitRule) { rule.Window = 1500 * time.Millisecond }},
		{name: "sliding calendar field", mutate: func(rule *RateLimitRule) { rule.CalendarPeriod = CalendarPeriodDay }},
		{name: "sliding token bucket field", mutate: func(rule *RateLimitRule) { rule.BucketCapacity = limit }},
		{name: "request response accounting", mutate: func(rule *RateLimitRule) { rule.Accounting = AccountingResponseActual }},
		{name: "request concurrency", mutate: func(rule *RateLimitRule) { rule.Algorithm = AlgorithmConcurrency; rule.Window = 0 }},
		{name: "whole metric cost limit", mutate: func(rule *RateLimitRule) { rule.CostLimit = currencyDecimalPointer(t, "5") }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			rule := valid
			test.mutate(&rule)
			if err := rule.Validate(); !errors.Is(err, ErrInvalidRateLimitRule) {
				t.Fatalf("Validate() error = %v, want %v", err, ErrInvalidRateLimitRule)
			}
		})
	}
}

func TestRateLimitRuleRejectsInvalidAlgorithmParameters(t *testing.T) {
	t.Parallel()

	limit := quotaIntegerPointer(t, "12")
	tests := []struct {
		name string
		rule RateLimitRule
	}{
		{
			name: "calendar missing timezone",
			rule: RateLimitRule{
				ID: "calendar", Metric: MetricRequests, Algorithm: AlgorithmCalendarWindow,
				Accounting: AccountingRequest, Enforcement: EnforcementEnforce,
				WholeLimit: limit, CalendarPeriod: CalendarPeriodDay,
			},
		},
		{
			name: "calendar invalid timezone",
			rule: RateLimitRule{
				ID: "calendar", Metric: MetricRequests, Algorithm: AlgorithmCalendarWindow,
				Accounting: AccountingRequest, Enforcement: EnforcementEnforce,
				WholeLimit: limit, CalendarPeriod: CalendarPeriodDay, CalendarTimezone: "Not/A_Zone",
			},
		},
		{
			name: "calendar local timezone",
			rule: RateLimitRule{
				ID: "calendar", Metric: MetricRequests, Algorithm: AlgorithmCalendarWindow,
				Accounting: AccountingRequest, Enforcement: EnforcementEnforce,
				WholeLimit: limit, CalendarPeriod: CalendarPeriodDay, CalendarTimezone: "Local",
			},
		},
		{
			name: "token bucket missing capacity",
			rule: RateLimitRule{
				ID: "bucket", Metric: MetricRequests, Algorithm: AlgorithmTokenBucket,
				Accounting: AccountingRequest, Enforcement: EnforcementEnforce,
				RefillAmount: quotaIntegerPointer(t, "1"), RefillPeriod: time.Second,
			},
		},
		{
			name: "token bucket zero refill",
			rule: RateLimitRule{
				ID: "bucket", Metric: MetricRequests, Algorithm: AlgorithmTokenBucket,
				Accounting: AccountingRequest, Enforcement: EnforcementEnforce,
				BucketCapacity: quotaIntegerPointer(t, "10"), RefillAmount: quotaIntegerPointer(t, "0"), RefillPeriod: time.Second,
			},
		},
		{
			name: "token bucket submillisecond period",
			rule: RateLimitRule{
				ID: "bucket", Metric: MetricRequests, Algorithm: AlgorithmTokenBucket,
				Accounting: AccountingRequest, Enforcement: EnforcementEnforce,
				BucketCapacity: quotaIntegerPointer(t, "10"), RefillAmount: quotaIntegerPointer(t, "1"), RefillPeriod: 1500 * time.Microsecond,
			},
		},
		{
			name: "gcra missing burst tolerance",
			rule: RateLimitRule{
				ID: "gcra", Metric: MetricRequests, Algorithm: AlgorithmGCRA,
				Accounting: AccountingRequest, Enforcement: EnforcementEnforce,
				GCRAEmissionInterval: time.Second,
			},
		},
		{
			name: "gcra submicrosecond interval",
			rule: RateLimitRule{
				ID: "gcra", Metric: MetricRequests, Algorithm: AlgorithmGCRA,
				Accounting: AccountingRequest, Enforcement: EnforcementEnforce,
				GCRAEmissionInterval: 1500 * time.Nanosecond, GCRABurstTolerance: quotaIntegerPointer(t, "0"),
			},
		},
		{
			name: "concurrency window",
			rule: RateLimitRule{
				ID: "concurrency", Metric: MetricConcurrentRequests, Algorithm: AlgorithmConcurrency,
				Accounting: AccountingRequest, Enforcement: EnforcementEnforce,
				WholeLimit: limit, Window: time.Minute,
			},
		},
		{
			name: "cost integer limit",
			rule: RateLimitRule{
				ID: "cost", Metric: MetricCost, Algorithm: AlgorithmSlidingLog,
				Accounting: AccountingResponseActual, Enforcement: EnforcementEnforce,
				WholeLimit: limit, Window: time.Minute,
			},
		},
		{
			name: "cost zero limit",
			rule: RateLimitRule{
				ID: "cost", Metric: MetricCost, Algorithm: AlgorithmSlidingLog,
				Accounting: AccountingResponseActual, Enforcement: EnforcementEnforce,
				CostLimit: currencyDecimalPointer(t, "0"), Window: time.Minute,
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if err := test.rule.Validate(); !errors.Is(err, ErrInvalidRateLimitRule) {
				t.Fatalf("Validate() error = %v, want %v", err, ErrInvalidRateLimitRule)
			}
		})
	}
}

func baseRule(metric Metric, algorithm Algorithm, accounting Accounting, limit *QuotaInteger) RateLimitRule {
	return RateLimitRule{
		ID: "rule-1", Metric: metric, Algorithm: algorithm, Accounting: accounting,
		Enforcement: EnforcementEnforce, WholeLimit: limit, Window: time.Minute,
	}
}

func quotaIntegerPointer(t *testing.T, value string) *QuotaInteger {
	t.Helper()
	parsed := mustQuotaInteger(t, value)
	return &parsed
}

func currencyDecimalPointer(t *testing.T, value string) *CurrencyDecimal {
	t.Helper()
	parsed := mustCurrencyDecimal(t, value)
	return &parsed
}
