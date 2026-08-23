package postgres

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

func TestCreateAccessPolicyWritesGrantsAndOutboxAtomically(t *testing.T) {
	store, mock := newMockStore(t)
	policy := testAccessPolicy()

	mock.ExpectBegin()
	mock.ExpectQuery(queryPattern(insertAccessPolicyQuery)).
		WithArgs(
			policy.ID, policy.NamespaceID, policy.DisplayName, policy.Status,
			policy.CreatedAt, policy.UpdatedAt,
		).
		WillReturnRows(accessPolicyRows().AddRow(
			policy.ID, policy.NamespaceID, policy.DisplayName, policy.Status,
			int64(1), policy.CreatedAt, policy.UpdatedAt,
		))
	grant := policy.Grants[0]
	mock.ExpectExec(queryPattern(insertAccessPolicyGrantQuery)).
		WithArgs(policy.ID, grant.Resource.Type, grant.Resource.ID, grant.Permission, grant.Effect).
		WillReturnResult(sqlmock.NewResult(0, 1))
	expectOutbox(mock, "access_policy", string(policy.ID), 1, 6, outboxCreated, nil)
	mock.ExpectCommit()

	result, err := store.CreateAccessPolicy(context.Background(), policy, testMutationMeta())
	if err != nil {
		t.Fatalf("create access policy: %v", err)
	}
	if len(result.Value.Grants) != 1 || result.Receipt.DesiredRevision != 6 {
		t.Fatalf("unexpected access-policy result: %#v", result)
	}
	assertExpectations(t, mock)
}

func TestGetAccessPolicyReadsAggregateFromOneSnapshot(t *testing.T) {
	store, mock := newMockStore(t)
	policy := testAccessPolicy()
	grant := policy.Grants[0]

	mock.ExpectBegin()
	mock.ExpectQuery(queryPattern(getAccessPolicyQuery)).
		WithArgs(testNamespaceID, testAccessPolicyID).
		WillReturnRows(accessPolicyRows().AddRow(
			policy.ID, policy.NamespaceID, policy.DisplayName, policy.Status,
			int64(1), policy.CreatedAt, policy.UpdatedAt,
		))
	mock.ExpectQuery(queryPattern(listAccessPolicyGrantsQuery)).
		WithArgs(policy.ID).
		WillReturnRows(accessGrantRows().AddRow(
			grant.PolicyID, grant.Resource.Type, grant.Resource.ID,
			grant.Permission, grant.Effect,
		))
	mock.ExpectCommit()

	got, err := store.GetAccessPolicy(context.Background(), testNamespaceID, testAccessPolicyID)
	if err != nil {
		t.Fatalf("get access policy: %v", err)
	}
	if len(got.Grants) != 1 || got.Grants[0] != grant {
		t.Fatalf("unexpected grants: %#v", got.Grants)
	}
	assertExpectations(t, mock)
}

func TestCreateRateLimitPolicyPersistsExactCostNumerator(t *testing.T) {
	store, mock := newMockStore(t)
	policy := testRateLimitPolicy()
	rule := policy.Rules[0]

	mock.ExpectBegin()
	mock.ExpectQuery(queryPattern(insertRateLimitPolicyQuery)).
		WithArgs(
			policy.ID, policy.NamespaceID, policy.DisplayName, policy.Status,
			policy.CreatedAt, policy.UpdatedAt,
		).
		WillReturnRows(rateLimitPolicyRows().AddRow(
			policy.ID, policy.NamespaceID, policy.DisplayName, policy.Status,
			int64(1), policy.CreatedAt, policy.UpdatedAt,
		))
	mock.ExpectExec(queryPattern(insertRateLimitRuleQuery)).
		WithArgs(
			rule.ID, rule.PolicyID, rule.Metric, rule.Algorithm,
			"2500000000000000", int64(60), nil, nil, nil, nil, nil, nil, nil,
			rule.Accounting, rule.Enforcement, int64(0),
		).
		WillReturnResult(sqlmock.NewResult(0, 1))
	expectOutbox(mock, "rate_limit_policy", string(policy.ID), 1, 7, outboxCreated, nil)
	mock.ExpectCommit()

	result, err := store.CreateRateLimitPolicy(context.Background(), policy, testMutationMeta())
	if err != nil {
		t.Fatalf("create rate-limit policy: %v", err)
	}
	if result.Value.Rules[0].Limit != "2.5" || result.Receipt.DesiredRevision != 7 {
		t.Fatalf("unexpected rate-limit result: %#v", result)
	}
	assertExpectations(t, mock)
}

func TestUpdateRateLimitPolicyPreservesRuleIdentityForLimitChange(t *testing.T) {
	store, mock := newMockStore(t)
	policy := testRateLimitPolicy()
	policy.Rules[0].Limit = "3"
	desiredRule := policy.Rules[0]

	mock.ExpectBegin()
	mock.ExpectQuery(queryPattern(updateRateLimitPolicyQuery)).
		WithArgs(
			policy.NamespaceID, policy.ID, int64(1), policy.DisplayName, policy.Status,
		).
		WillReturnRows(rateLimitPolicyRows().AddRow(
			policy.ID, policy.NamespaceID, policy.DisplayName, policy.Status,
			int64(2), policy.CreatedAt, policy.UpdatedAt.Add(time.Second),
		))
	mock.ExpectQuery(queryPattern(listRateLimitRulesQuery)).
		WithArgs(policy.ID).
		WillReturnRows(rateLimitRuleRows().AddRow(
			desiredRule.ID, desiredRule.PolicyID, desiredRule.Metric, desiredRule.Algorithm,
			"2500000000000000", int64(60), nil, nil, nil, nil, nil, nil, nil,
			desiredRule.Accounting, desiredRule.Enforcement, int64(0),
		))
	mock.ExpectExec(queryPattern(setRateLimitRuleOrdinalQuery)).
		WithArgs(policy.ID, desiredRule.ID, int64(2147483647)).
		WillReturnResult(sqlmock.NewResult(0, 1))
	mock.ExpectExec(queryPattern(updateRateLimitRuleQuery)).
		WithArgs(
			policy.ID, desiredRule.ID, desiredRule.Metric, desiredRule.Algorithm,
			"3000000000000000", int64(60), nil, nil, nil, nil, nil, nil, nil,
			desiredRule.Accounting, desiredRule.Enforcement, int64(0),
		).
		WillReturnResult(sqlmock.NewResult(0, 1))
	expectOutbox(mock, "rate_limit_policy", string(policy.ID), 2, 11, outboxUpdated, nil)
	mock.ExpectCommit()

	result, err := store.UpdateRateLimitPolicy(context.Background(), policy, 1, testMutationMeta())
	if err != nil {
		t.Fatalf("update rate-limit policy: %v", err)
	}
	if result.Value.Revision != 2 || result.Value.Rules[0].Limit != "3" {
		t.Fatalf("unexpected updated policy: %#v", result)
	}
	assertExpectations(t, mock)
}

func TestCostScaleRoundTripIsExact(t *testing.T) {
	tests := []struct {
		name      string
		decimal   string
		numerator string
	}{
		{name: "whole", decimal: "5", numerator: "5000000000000000"},
		{name: "fraction", decimal: "2.5", numerator: "2500000000000000"},
		{name: "minimum", decimal: "0.000000000000001", numerator: "1"},
		{name: "fifteen digits", decimal: "12.345678901234567", numerator: "12345678901234567"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			numerator, err := scaleCostDecimal(test.decimal)
			if err != nil {
				t.Fatalf("scale cost: %v", err)
			}
			if numerator != test.numerator {
				t.Fatalf("scaled numerator = %q, want %q", numerator, test.numerator)
			}
			decimal, err := unscaleCostInteger(numerator)
			if err != nil {
				t.Fatalf("unscale cost: %v", err)
			}
			if decimal != test.decimal {
				t.Fatalf("round trip = %q, want %q", decimal, test.decimal)
			}
		})
	}
}

func TestCostScaleRejectsDatabaseOverflow(t *testing.T) {
	_, err := scaleCostDecimal(strings.Repeat("9", maxQuotaDigits))
	if err == nil {
		t.Fatal("expected scaled NUMERIC overflow")
	}
}

func testAccessPolicy() accesscontrol.AccessPolicy {
	return accesscontrol.AccessPolicy{
		NamespaceID: testNamespaceID,
		ID:          testAccessPolicyID,
		DisplayName: "models",
		Status:      accesscontrol.PolicyStatusActive,
		Revision:    1,
		CreatedAt:   testNow,
		UpdatedAt:   testNow,
		Grants: []accesscontrol.AccessPolicyGrant{{
			PolicyID: testAccessPolicyID,
			Resource: accesscontrol.GrantResource{
				Type: accesscontrol.GrantResourceModel,
				ID:   testResourceID,
			},
			Permission: accesscontrol.GrantPermissionInvoke,
			Effect:     accesscontrol.GrantEffectAllow,
		}},
	}
}

func testRateLimitPolicy() accesscontrol.RateLimitPolicy {
	return accesscontrol.RateLimitPolicy{
		NamespaceID: testNamespaceID,
		ID:          testRatePolicyID,
		DisplayName: "cost",
		Status:      accesscontrol.PolicyStatusActive,
		Revision:    1,
		CreatedAt:   testNow,
		UpdatedAt:   testNow,
		Rules: []accesscontrol.RateLimitRule{{
			ID:          testRuleID,
			PolicyID:    testRatePolicyID,
			Metric:      accesscontrol.RateMetricCost,
			Algorithm:   accesscontrol.RateAlgorithmSlidingLog,
			Limit:       "2.5",
			Window:      time.Minute,
			Accounting:  accesscontrol.RateAccountingResponseActual,
			Enforcement: accesscontrol.RateEnforcementEnforce,
			Ordinal:     0,
		}},
	}
}

func accessPolicyRows() *sqlmock.Rows {
	return sqlmock.NewRows([]string{
		"id", "namespace_id", "name", "status", "revision", "created_at", "updated_at",
	})
}

func accessGrantRows() *sqlmock.Rows {
	return sqlmock.NewRows([]string{
		"policy_id", "resource_type", "resource_id", "permission", "effect",
	})
}

func rateLimitPolicyRows() *sqlmock.Rows {
	return sqlmock.NewRows([]string{
		"id", "namespace_id", "name", "status", "revision", "created_at", "updated_at",
	})
}

func rateLimitRuleRows() *sqlmock.Rows {
	return sqlmock.NewRows([]string{
		"id", "policy_id", "metric", "algorithm", "limit_value", "window_seconds",
		"calendar_period", "timezone", "bucket_capacity", "refill_amount",
		"refill_period_milliseconds", "gcra_emission_interval_microseconds",
		"gcra_burst_tolerance", "accounting", "enforcement", "ordinal",
	})
}
