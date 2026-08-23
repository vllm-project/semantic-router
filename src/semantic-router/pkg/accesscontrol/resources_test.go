package accesscontrol

import (
	"errors"
	"testing"
	"time"
)

func TestCoreResourceValidation(t *testing.T) {
	user := validUser()
	team := validTeam()
	membership := validMembership()
	namespace := Namespace{
		ID:               "ns-1",
		Name:             "Namespace One",
		QuotaPartitionID: "partition-1",
		BillingCurrency:  "USD",
		Status:           NamespaceStatusActive,
		Revision:         1,
		RuntimeEpoch:     1,
		CreatedAt:        testTime,
		UpdatedAt:        testTime,
	}

	tests := []struct {
		name     string
		validate func() error
		wantErr  bool
	}{
		{name: "namespace", validate: namespace.Validate},
		{name: "namespace currency is canonical", validate: func() error {
			invalidNamespace := namespace
			invalidNamespace.BillingCurrency = "usd"
			return invalidNamespace.Validate()
		}, wantErr: true},
		{name: "user", validate: user.Validate},
		{name: "user email is normalized", validate: func() error {
			invalidUser := user
			invalidUser.Email = "User@example.com"
			return invalidUser.Validate()
		}, wantErr: true},
		{name: "team", validate: team.Validate},
		{name: "membership", validate: membership.Validate},
		{name: "membership role", validate: func() error {
			invalidMembership := membership
			invalidMembership.Role = "owner"
			return invalidMembership.Validate()
		}, wantErr: true},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := test.validate()
			if test.wantErr && !errors.Is(err, ErrInvalid) {
				t.Fatalf("expected validation error, got %v", err)
			}
			if !test.wantErr && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestBindingReferenceValidation(t *testing.T) {
	namespace := Namespace{
		ID:               "ns-1",
		Name:             "Namespace One",
		QuotaPartitionID: "partition-1",
		BillingCurrency:  "USD",
		Status:           NamespaceStatusActive,
		Revision:         1,
		RuntimeEpoch:     1,
		CreatedAt:        testTime,
		UpdatedAt:        testTime,
	}
	user := validUser()
	subject := Subject{NamespaceID: user.NamespaceID, ID: SubjectID(user.ID), Kind: SubjectKindUser}
	accessPolicy := AccessPolicy{
		NamespaceID: "ns-1", ID: "access-1", DisplayName: "Access",
		Status: PolicyStatusActive, Revision: 1, CreatedAt: testTime, UpdatedAt: testTime,
	}
	ratePolicy := RateLimitPolicy{
		NamespaceID: "ns-1", ID: "rate-1", DisplayName: "Rate",
		Status: PolicyStatusActive, Revision: 1, CreatedAt: testTime, UpdatedAt: testTime,
	}
	accessBinding := validAccessBinding("access-binding", SubjectKindUser, "user-1")
	rateBinding := validRateBinding("rate-binding", SubjectKindUser, "user-1", RateBindingAllocation)

	if err := ValidateAccessBindingReferences(accessBinding, accessPolicy, subject); err != nil {
		t.Fatalf("valid access references rejected: %v", err)
	}
	if err := ValidateRateBindingReferences(rateBinding, ratePolicy, subject, namespace); err != nil {
		t.Fatalf("valid rate references rejected: %v", err)
	}

	wrongPartition := rateBinding
	wrongPartition.QuotaPartitionID = "partition-2"
	if err := ValidateRateBindingReferences(wrongPartition, ratePolicy, subject, namespace); !errors.Is(err, ErrInvalid) {
		t.Fatalf("expected partition validation error, got %v", err)
	}
}

func TestPolicyValidationRejectsDuplicateChildren(t *testing.T) {
	grant := AccessPolicyGrant{
		PolicyID:   "access-1",
		Resource:   GrantResource{Type: GrantResourceModel, ID: "model-1"},
		Permission: GrantPermissionInvoke,
		Effect:     GrantEffectAllow,
	}
	accessPolicy := AccessPolicy{
		NamespaceID: "ns-1", ID: "access-1", DisplayName: "Access", Status: PolicyStatusActive,
		Revision: 1, Grants: []AccessPolicyGrant{grant, grant}, CreatedAt: testTime, UpdatedAt: testTime,
	}
	if err := accessPolicy.Validate(); !errors.Is(err, ErrInvalid) {
		t.Fatalf("expected duplicate grant error, got %v", err)
	}

	rule := baseRule(RateMetricRequests, RateAlgorithmSlidingLog, RateAccountingRequest, func(rule *RateLimitRule) {
		rule.Limit = "10"
		rule.Window = time.Minute
	})
	ratePolicy := RateLimitPolicy{
		NamespaceID: "ns-1", ID: "rate-1", DisplayName: "Rate", Status: PolicyStatusActive,
		Revision: 1, Rules: []RateLimitRule{rule, rule}, CreatedAt: testTime, UpdatedAt: testTime,
	}
	if err := ratePolicy.Validate(); !errors.Is(err, ErrInvalid) {
		t.Fatalf("expected duplicate rule error, got %v", err)
	}
}
