package accesscontrol

import (
	"errors"
	"testing"
	"time"
)

type ruleValidationTest struct {
	name    string
	rule    RateLimitRule
	wantErr bool
}

func TestValidRateLimitRules(t *testing.T) {
	tests := []ruleValidationTest{
		{
			name: "request sliding log",
			rule: baseRule(RateMetricRequests, RateAlgorithmSlidingLog, RateAccountingRequest,
				func(rule *RateLimitRule) { rule.Limit = "12"; rule.Window = time.Minute }),
		},
		{
			name: "actual tokens calendar window",
			rule: baseRule(RateMetricTotalTokens, RateAlgorithmCalendarWindow, RateAccountingResponseActual,
				func(rule *RateLimitRule) {
					rule.Limit = "1000000"
					rule.CalendarPeriod = CalendarPeriodMonth
					rule.Timezone = "America/Los_Angeles"
				}),
		},
		{
			name: "request token bucket",
			rule: baseRule(RateMetricRequests, RateAlgorithmTokenBucket, RateAccountingRequest,
				func(rule *RateLimitRule) {
					rule.BucketCapacity = "100"
					rule.RefillAmount = "10"
					rule.RefillPeriod = time.Second
				}),
		},
		{
			name: "request gcra",
			rule: baseRule(RateMetricRequests, RateAlgorithmGCRA, RateAccountingRequest,
				func(rule *RateLimitRule) {
					rule.GCRAEmissionInterval = time.Millisecond
					rule.GCRABurstTolerance = int64Pointer(0)
				}),
		},
		{
			name: "concurrency",
			rule: baseRule(RateMetricConcurrentRequests, RateAlgorithmConcurrency, RateAccountingRequest,
				func(rule *RateLimitRule) { rule.Limit = "8" }),
		},
		{
			name: "exact cost",
			rule: baseRule(RateMetricCost, RateAlgorithmSlidingLog, RateAccountingResponseActual,
				func(rule *RateLimitRule) { rule.Limit = "12.000000000000001"; rule.Window = time.Hour }),
		},
	}
	runRuleValidationTests(t, tests)
}

func TestInvalidRateLimitRules(t *testing.T) {
	tests := []ruleValidationTest{
		{
			name: "tokens cannot use request accounting",
			rule: baseRule(RateMetricInputTokens, RateAlgorithmSlidingLog, RateAccountingRequest,
				func(rule *RateLimitRule) { rule.Limit = "100"; rule.Window = time.Minute }),
			wantErr: true,
		},
		{
			name: "actual tokens cannot use token bucket",
			rule: baseRule(RateMetricOutputTokens, RateAlgorithmTokenBucket, RateAccountingResponseActual,
				func(rule *RateLimitRule) {
					rule.BucketCapacity = "100"
					rule.RefillAmount = "10"
					rule.RefillPeriod = time.Second
				}),
			wantErr: true,
		},
		{
			name: "calendar requires IANA timezone",
			rule: baseRule(RateMetricRequests, RateAlgorithmCalendarWindow, RateAccountingRequest,
				func(rule *RateLimitRule) {
					rule.Limit = "10"
					rule.CalendarPeriod = CalendarPeriodDay
					rule.Timezone = "not-a-zone"
				}),
			wantErr: true,
		},
		{
			name: "whole value is bounded to 42 digits",
			rule: baseRule(RateMetricRequests, RateAlgorithmSlidingLog, RateAccountingRequest,
				func(rule *RateLimitRule) {
					rule.Limit = "1234567890123456789012345678901234567890123"
					rule.Window = time.Minute
				}),
			wantErr: true,
		},
		{
			name: "sliding window uses whole seconds",
			rule: baseRule(RateMetricRequests, RateAlgorithmSlidingLog, RateAccountingRequest,
				func(rule *RateLimitRule) { rule.Limit = "1"; rule.Window = time.Second + time.Nanosecond }),
			wantErr: true,
		},
	}

	runRuleValidationTests(t, tests)
}

func runRuleValidationTests(t *testing.T, tests []ruleValidationTest) {
	t.Helper()
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := test.rule.Validate()
			if test.wantErr && !errors.Is(err, ErrInvalid) {
				t.Fatalf("expected validation error, got %v", err)
			}
			if !test.wantErr && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func baseRule(metric RateMetric, algorithm RateAlgorithm, accounting RateAccounting, configure func(*RateLimitRule)) RateLimitRule {
	rule := RateLimitRule{
		ID:          "rule-1",
		PolicyID:    "rate-1",
		Metric:      metric,
		Algorithm:   algorithm,
		Accounting:  accounting,
		Enforcement: RateEnforcementEnforce,
	}
	configure(&rule)
	return rule
}

func int64Pointer(value int64) *int64 { return &value }

func TestResolveRateBindings(t *testing.T) {
	keyAllocation := validRateBinding("key-allocation", SubjectKindAPIKey, "key-1", RateBindingAllocation)
	userAllocation := validRateBinding("user-allocation", SubjectKindUser, "user-1", RateBindingAllocation)
	teamAllocation := validRateBinding("team-allocation", SubjectKindTeam, "team-1", RateBindingAllocation)
	keyCap := validRateBinding("key-cap", SubjectKindAPIKey, "key-1", RateBindingHardCap)
	userCap := validRateBinding("user-cap", SubjectKindUser, "user-1", RateBindingHardCap)
	teamCap := validRateBinding("team-cap", SubjectKindTeam, "team-1", RateBindingHardCap)

	resolved, err := ResolveRateBindings(
		[]RateLimitBinding{keyAllocation, keyCap},
		[]RateLimitBinding{userAllocation, userCap},
		[]RateLimitBinding{teamAllocation, teamCap},
	)
	if err != nil {
		t.Fatalf("ResolveRateBindings() error = %v", err)
	}
	if resolved.Allocation == nil || resolved.Allocation.Binding.ID != keyAllocation.ID || resolved.Allocation.Source != InheritanceLayerKey {
		t.Fatalf("unexpected allocation: %#v", resolved.Allocation)
	}
	if len(resolved.HardCaps) != 3 {
		t.Fatalf("hard caps = %d, want 3", len(resolved.HardCaps))
	}
	if resolved.Allocation.Binding.CounterID() != keyAllocation.ID {
		t.Fatalf("counter id must be binding id")
	}
}

func TestResolveRateBindingsRejectsMultipleAllocations(t *testing.T) {
	first := validRateBinding("allocation-1", SubjectKindUser, "user-1", RateBindingAllocation)
	second := validRateBinding("allocation-2", SubjectKindUser, "user-1", RateBindingAllocation)
	if _, err := ResolveRateBindings(nil, []RateLimitBinding{first, second}, nil); !errors.Is(err, ErrInvalid) {
		t.Fatalf("expected validation error, got %v", err)
	}
}
