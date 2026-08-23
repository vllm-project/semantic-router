package postgres

import (
	"context"
	"errors"
	"testing"

	"github.com/DATA-DOG/go-sqlmock"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

func TestCreateAccessPolicyBindingChecksTypedSubjectAndPublishes(t *testing.T) {
	store, mock := newMockStore(t)
	binding := testAccessBinding()

	mock.ExpectBegin()
	mock.ExpectQuery(queryPattern(insertAccessPolicyBindingQuery)).
		WithArgs(
			binding.ID, binding.NamespaceID, binding.Subject.ID, binding.Subject.Kind,
			binding.PolicyID, binding.Status,
		).
		WillReturnRows(accessBindingRows().AddRow(
			binding.ID, binding.NamespaceID, binding.Subject.ID, binding.Subject.Kind,
			binding.PolicyID, binding.Status, int64(1),
		))
	expectOutbox(
		mock, "access_policy_binding", string(binding.ID), 1, 8, outboxCreated,
		map[string]string{
			"policyId": string(binding.PolicyID), "subjectId": string(binding.Subject.ID),
		},
	)
	mock.ExpectCommit()

	result, err := store.CreateAccessPolicyBinding(context.Background(), binding, testMutationMeta())
	if err != nil {
		t.Fatalf("create access-policy binding: %v", err)
	}
	if result.Value != binding || result.Receipt.DesiredRevision != 8 {
		t.Fatalf("unexpected binding result: %#v", result)
	}
	assertExpectations(t, mock)
}

func TestCreateRateLimitBindingPublishesCounterIdentity(t *testing.T) {
	store, mock := newMockStore(t)
	binding := testRateBinding()

	mock.ExpectBegin()
	mock.ExpectQuery(queryPattern(insertRateLimitBindingQuery)).
		WithArgs(
			binding.ID, binding.NamespaceID, binding.Subject.ID, binding.Subject.Kind,
			binding.PolicyID, binding.Mode, binding.QuotaPartitionID, binding.Status,
		).
		WillReturnRows(rateBindingRows().AddRow(
			binding.ID, binding.NamespaceID, binding.Subject.ID, binding.Subject.Kind,
			binding.PolicyID, binding.Mode, binding.QuotaPartitionID, binding.Status, int64(1),
		))
	expectOutbox(
		mock, "rate_limit_binding", string(binding.ID), 1, 9, outboxCreated,
		map[string]string{
			"policyId": string(binding.PolicyID), "subjectId": string(binding.Subject.ID),
		},
	)
	mock.ExpectCommit()

	result, err := store.CreateRateLimitBinding(context.Background(), binding, testMutationMeta())
	if err != nil {
		t.Fatalf("create rate-limit binding: %v", err)
	}
	if result.Value.CounterID() != binding.ID || result.Receipt.DesiredRevision != 9 {
		t.Fatalf("unexpected rate binding result: %#v", result)
	}
	assertExpectations(t, mock)
}

func TestSetRateLimitBindingStatusUsesCAS(t *testing.T) {
	store, mock := newMockStore(t)

	mock.ExpectBegin()
	mock.ExpectQuery(queryPattern(updateRateLimitBindingStatusQuery)).
		WithArgs(testNamespaceID, testBindingID, int64(4), accesscontrol.BindingStatusDisabled).
		WillReturnRows(rateBindingRows())
	mock.ExpectRollback()

	_, err := store.SetRateLimitBindingStatus(
		context.Background(), testNamespaceID, testBindingID, 4,
		accesscontrol.BindingStatusDisabled, testMutationMeta(),
	)
	if !errors.Is(err, ErrRevisionConflict) {
		t.Fatalf("expected revision conflict, got %v", err)
	}
	assertExpectations(t, mock)
}

func testSubjectRef() accesscontrol.SubjectRef {
	return accesscontrol.SubjectRef{
		NamespaceID: testNamespaceID,
		ID:          accesscontrol.SubjectID(testUserID),
		Kind:        accesscontrol.SubjectKindUser,
	}
}

func testAccessBinding() accesscontrol.AccessPolicyBinding {
	return accesscontrol.AccessPolicyBinding{
		ID:          testBindingID,
		NamespaceID: testNamespaceID,
		Subject:     testSubjectRef(),
		PolicyID:    testAccessPolicyID,
		Status:      accesscontrol.BindingStatusActive,
		Revision:    1,
	}
}

func testRateBinding() accesscontrol.RateLimitBinding {
	return accesscontrol.RateLimitBinding{
		ID:               testBindingID,
		NamespaceID:      testNamespaceID,
		Subject:          testSubjectRef(),
		PolicyID:         testRatePolicyID,
		Mode:             accesscontrol.RateBindingAllocation,
		QuotaPartitionID: "partition-default",
		Status:           accesscontrol.BindingStatusActive,
		Revision:         1,
	}
}

func accessBindingRows() *sqlmock.Rows {
	return sqlmock.NewRows([]string{
		"id", "namespace_id", "subject_id", "kind", "policy_id", "status", "revision",
	})
}

func rateBindingRows() *sqlmock.Rows {
	return sqlmock.NewRows([]string{
		"id", "namespace_id", "subject_id", "kind", "policy_id",
		"binding_mode", "quota_partition_id", "status", "revision",
	})
}
