package policymanagement

import (
	"errors"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

func TestCompileInlineRateLimitPolicyCanonicalizesAndPreservesRuleIDs(t *testing.T) {
	const (
		namespaceID = "11111111-1111-4111-8111-111111111111"
		policyID    = "22222222-2222-4222-8222-222222222222"
		retainedID  = "33333333-3333-4333-8333-333333333333"
		generatedID = "44444444-4444-4444-8444-444444444444"
	)
	burst := int64(0)
	input := []RateLimitRule{
		{
			ID: retainedID, Metric: accesscontrol.RateMetricRequests,
			Algorithm: accesscontrol.RateAlgorithmSlidingLog, Limit: "12", Window: ISODuration(time.Minute),
			Accounting: accesscontrol.RateAccountingRequest, Enforcement: accesscontrol.RateEnforcementEnforce,
			Ordinal: 99,
		},
		{
			Metric:    accesscontrol.RateMetricRequests,
			Algorithm: accesscontrol.RateAlgorithmGCRA, GCRAEmissionInterval: ISODuration(time.Second),
			Accounting: accesscontrol.RateAccountingRequest, Enforcement: accesscontrol.RateEnforcementShadow,
			GCRABurstTolerance: &burst, Ordinal: 88,
		},
	}
	called := 0
	policy, err := CompileInlineRateLimitPolicy(InlineRateLimitPolicySpec{
		NamespaceID: namespaceID, PolicyID: policyID, Name: "  Developer  ",
		Description: "  Interactive quota  ", Rules: input,
		Now:       time.Date(2026, 8, 22, 4, 5, 6, 123456789, time.FixedZone("test", 8*60*60)),
		NewRuleID: func() string { called++; return generatedID },
	})
	if err != nil {
		t.Fatal(err)
	}
	if called != 1 || policy.Name != "Developer" || policy.Description != "Interactive quota" ||
		policy.Status != accesscontrol.PolicyStatusActive || policy.Revision != 1 ||
		policy.CreatedAt.Location() != time.UTC || policy.CreatedAt.Nanosecond() != 123456000 ||
		!policy.CreatedAt.Equal(policy.UpdatedAt) {
		t.Fatalf("compiled policy = %#v; ID calls = %d", policy, called)
	}
	if len(policy.Rules) != 2 || policy.Rules[0].ID != retainedID || policy.Rules[0].Ordinal != 0 ||
		policy.Rules[1].ID != generatedID || policy.Rules[1].Ordinal != 1 ||
		policy.Rules[1].GCRABurstTolerance == input[1].GCRABurstTolerance {
		t.Fatalf("compiled rules = %#v", policy.Rules)
	}
	burst = 9
	if *policy.Rules[1].GCRABurstTolerance != 0 {
		t.Fatalf("compiled burst tolerance was aliased: %d", *policy.Rules[1].GCRABurstTolerance)
	}
	if input[0].Ordinal != 99 || input[1].Ordinal != 88 || input[1].ID != "" {
		t.Fatalf("compiler mutated input = %#v", input)
	}
}

func TestCompileInlineRateLimitPolicyRejectsIncompleteOrUnsafeRules(t *testing.T) {
	base := InlineRateLimitPolicySpec{
		NamespaceID: "11111111-1111-4111-8111-111111111111",
		PolicyID:    "22222222-2222-4222-8222-222222222222",
		Name:        "Developer",
		Now:         time.Date(2026, 8, 22, 0, 0, 0, 0, time.UTC),
	}
	if _, err := CompileInlineRateLimitPolicy(base); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("empty rules error = %v", err)
	}
	base.Rules = []RateLimitRule{{
		Metric:    accesscontrol.RateMetricRequests,
		Algorithm: accesscontrol.RateAlgorithmSlidingLog,
		Limit:     "1000000000000000000000000000000000000000000", Window: ISODuration(time.Minute),
		Accounting: accesscontrol.RateAccountingRequest, Enforcement: accesscontrol.RateEnforcementEnforce,
	}}
	base.NewRuleID = func() string { return "33333333-3333-4333-8333-333333333333" }
	if _, err := CompileInlineRateLimitPolicy(base); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("43-digit limit error = %v", err)
	}
	base.Rules[0].Limit = "12"
	base.NewRuleID = func() string { return "not-a-uuid" }
	if _, err := CompileInlineRateLimitPolicy(base); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("invalid ID source error = %v", err)
	}
	base.NewRuleID = func() string { return "33333333-3333-4333-8333-333333333333" }
	base.Rules[0] = RateLimitRule{
		Metric:    accesscontrol.RateMetricCost,
		Algorithm: accesscontrol.RateAlgorithmSlidingLog,
		Limit:     "1234567890123456789012345678", Window: ISODuration(time.Minute),
		Accounting:  accesscontrol.RateAccountingResponseActual,
		Enforcement: accesscontrol.RateEnforcementEnforce,
	}
	if _, err := CompileInlineRateLimitPolicy(base); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("scaled 43-digit cost error = %v", err)
	}
}

func TestCompileInlineRateLimitPolicyAcceptsEightHourActualCostWindow(t *testing.T) {
	policy, err := CompileInlineRateLimitPolicy(InlineRateLimitPolicySpec{
		NamespaceID: "11111111-1111-4111-8111-111111111111",
		PolicyID:    "22222222-2222-4222-8222-222222222222",
		Name:        "Eight hour spend",
		Rules: []RateLimitRule{{
			Metric: accesscontrol.RateMetricCost, Algorithm: accesscontrol.RateAlgorithmSlidingLog,
			Limit: "20", Window: ISODuration(8 * time.Hour),
			Accounting:  accesscontrol.RateAccountingResponseActual,
			Enforcement: accesscontrol.RateEnforcementEnforce,
		}},
		Now:       time.Date(2026, 8, 23, 0, 0, 0, 0, time.UTC),
		NewRuleID: func() string { return "33333333-3333-4333-8333-333333333333" },
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(policy.Rules) != 1 || policy.Rules[0].Limit != "20" ||
		policy.Rules[0].Window.Duration() != 8*time.Hour ||
		policy.Rules[0].Metric != accesscontrol.RateMetricCost ||
		policy.Rules[0].Accounting != accesscontrol.RateAccountingResponseActual {
		t.Fatalf("compiled policy = %#v", policy)
	}
}
