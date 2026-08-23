package postgres

import (
	"context"
	"errors"
	"testing"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

func TestCreateNamespaceCommitsDesiredStateAndOutbox(t *testing.T) {
	store, mock := newMockStore(t)
	namespace := accesscontrol.Namespace{
		ID: testNamespaceID, Name: "default", QuotaPartitionID: "partition-default",
		BillingCurrency: "USD", Status: accesscontrol.NamespaceStatusActive,
		Revision: 1, RuntimeEpoch: 7, CreatedAt: testNow, UpdatedAt: testNow,
	}

	mock.ExpectBegin()
	mock.ExpectQuery(queryPattern(insertNamespaceQuery)).
		WithArgs(
			namespace.ID, namespace.Name, namespace.QuotaPartitionID, namespace.BillingCurrency,
			namespace.Status, int64(1), int64(7), testNow, testNow,
		).
		WillReturnRows(namespaceRows().AddRow(
			namespace.ID, namespace.Name, namespace.QuotaPartitionID, namespace.BillingCurrency,
			namespace.Status, int64(1), int64(7), testNow, testNow,
		))
	mock.ExpectExec(queryPattern(insertNamespaceSecurityPolicyQuery)).
		WithArgs(namespace.ID).
		WillReturnResult(sqlmock.NewResult(0, 1))
	expectOutbox(mock, "namespace", string(namespace.ID), 1, 1, outboxCreated, nil)
	mock.ExpectCommit()

	result, err := store.CreateNamespace(context.Background(), namespace, testMutationMeta())
	if err != nil {
		t.Fatalf("create namespace: %v", err)
	}
	if result.Receipt.DesiredRevision != 1 || result.Value != namespace {
		t.Fatalf("unexpected namespace result: %#v", result)
	}
	assertExpectations(t, mock)
}

func TestCreateUserCommitsSubjectAndOutboxTogether(t *testing.T) {
	store, mock := newMockStore(t)
	user := accesscontrol.User{
		NamespaceID: testNamespaceID, ID: testUserID, Email: "user@example.com",
		DisplayName: "User", Status: accesscontrol.UserStatusActive,
		CreatedAt: testNow, UpdatedAt: testNow,
	}

	mock.ExpectBegin()
	mock.ExpectExec(queryPattern(insertSubjectQuery)).
		WithArgs(testNamespaceID, testUserID, accesscontrol.SubjectKindUser, testNow).
		WillReturnResult(sqlmock.NewResult(0, 1))
	mock.ExpectQuery(queryPattern(insertUserQuery)).
		WithArgs(
			user.ID, user.NamespaceID, user.Email, user.DisplayName,
			user.Status, user.CreatedAt, user.UpdatedAt,
		).
		WillReturnRows(userRows().AddRow(
			user.ID, user.NamespaceID, user.Email, user.DisplayName,
			user.Status, int64(1), user.CreatedAt, user.UpdatedAt, nil,
		))
	expectOutbox(mock, "user", string(user.ID), 1, 2, outboxCreated, nil)
	mock.ExpectCommit()

	result, err := store.CreateUser(context.Background(), user, testMutationMeta())
	if err != nil {
		t.Fatalf("create user: %v", err)
	}
	if result.Value.User != user || result.Value.Revision != 1 || result.Receipt.DesiredRevision != 2 {
		t.Fatalf("unexpected user result: %#v", result)
	}
	assertExpectations(t, mock)
}

func TestCreateMembershipPublishesCompositeIdentity(t *testing.T) {
	store, mock := newMockStore(t)
	membership := accesscontrol.TeamMembership{
		NamespaceID: testNamespaceID, TeamID: testTeamID, UserID: testUserID,
		Role: accesscontrol.TeamRoleMember, Status: accesscontrol.MembershipStatusActive,
		CreatedAt: testNow, UpdatedAt: testNow,
	}
	eventAggregateID := membershipEventAggregateID(membership)

	mock.ExpectBegin()
	mock.ExpectQuery(queryPattern(insertMembershipQuery)).
		WithArgs(
			membership.NamespaceID, membership.TeamID, membership.UserID,
			membership.Role, membership.Status, membership.CreatedAt, membership.UpdatedAt,
		).
		WillReturnRows(membershipRows().AddRow(
			membership.NamespaceID, membership.TeamID, membership.UserID,
			membership.Role, membership.Status, int64(1), membership.CreatedAt, membership.UpdatedAt,
		))
	expectOutbox(
		mock, "team_membership", eventAggregateID, 1, 3, outboxCreated,
		map[string]string{
			"namespaceId": string(testNamespaceID),
			"teamId":      string(testTeamID),
			"userId":      string(testUserID),
			"resourceRef": membershipResourceReference(membership),
		},
	)
	mock.ExpectCommit()

	result, err := store.CreateMembership(context.Background(), membership, testMutationMeta())
	if err != nil {
		t.Fatalf("create membership: %v", err)
	}
	if result.Value.Membership != membership || result.Value.Revision != 1 {
		t.Fatalf("unexpected membership result: %#v", result)
	}
	assertExpectations(t, mock)
}

func TestUpdateTeamCASConflictRollsBackWithoutOutbox(t *testing.T) {
	store, mock := newMockStore(t)
	record := TeamRecord{
		Team: accesscontrol.Team{
			NamespaceID: testNamespaceID, ID: testTeamID, Name: "Team",
			Status: accesscontrol.TeamStatusActive, CreatedAt: testNow, UpdatedAt: testNow,
		},
		Description: "description", Revision: 5,
	}

	mock.ExpectBegin()
	mock.ExpectQuery(queryPattern(updateTeamQuery)).
		WithArgs(testNamespaceID, testTeamID, int64(5), "Team", "description", accesscontrol.TeamStatusActive).
		WillReturnRows(teamRows())
	mock.ExpectRollback()

	_, err := store.UpdateTeam(context.Background(), record, 5, testMutationMeta())
	if !errors.Is(err, ErrRevisionConflict) {
		t.Fatalf("expected revision conflict, got %v", err)
	}
	assertExpectations(t, mock)
}

func TestMembershipEventIdentityIsStableAndComposite(t *testing.T) {
	membership := accesscontrol.TeamMembership{
		NamespaceID: testNamespaceID, TeamID: testTeamID, UserID: testUserID,
	}
	first := membershipEventAggregateID(membership)
	if first != membershipEventAggregateID(membership) {
		t.Fatal("membership event aggregate ID is not deterministic")
	}
	parsed, err := uuid.Parse(first)
	if err != nil || parsed.Version() != 5 {
		t.Fatalf("membership aggregate is not UUIDv5: %q, %v", first, err)
	}
	membership.UserID = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"
	if first == membershipEventAggregateID(membership) {
		t.Fatal("different composite membership produced the same event aggregate ID")
	}
}

func namespaceRows() *sqlmock.Rows {
	return sqlmock.NewRows([]string{
		"id", "name", "quota_partition_id", "billing_currency", "status",
		"revision", "runtime_epoch", "created_at", "updated_at",
	})
}

func userRows() *sqlmock.Rows {
	return sqlmock.NewRows([]string{
		"id", "namespace_id", "email", "display_name", "status",
		"revision", "created_at", "updated_at", "deleted_at",
	})
}

func teamRows() *sqlmock.Rows {
	return sqlmock.NewRows([]string{
		"id", "namespace_id", "name", "description", "status",
		"revision", "created_at", "updated_at", "deleted_at",
	})
}

func membershipRows() *sqlmock.Rows {
	return sqlmock.NewRows([]string{
		"namespace_id", "team_id", "user_id", "role", "status",
		"revision", "created_at", "updated_at",
	})
}
