package postgres

import (
	"context"
	"errors"
	"regexp"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementpermission"
)

const (
	testPrincipalID = accesscontrol.ManagementPrincipalID("11111111-1111-4111-8111-111111111111")
	testNamespaceID = accesscontrol.NamespaceID("22222222-2222-4222-8222-222222222222")
	testUserID      = "33333333-3333-4333-8333-333333333333"
	testTeamID      = "44444444-4444-4444-8444-444444444444"
	testBindingID   = "55555555-5555-4555-8555-555555555555"
	testRoleID      = "66666666-6666-4666-8666-666666666666"
)

var authorizationTestTime = time.Date(2026, 8, 22, 10, 0, 0, 0, time.UTC)

func newAuthorizationMock(t *testing.T) (*Store, sqlmock.Sqlmock) {
	t.Helper()
	database, mock, err := sqlmock.New(sqlmock.QueryMatcherOption(sqlmock.QueryMatcherRegexp))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = database.Close() })
	store, err := New(database)
	if err != nil {
		t.Fatal(err)
	}
	return store, mock
}

func expectPrincipal(mock sqlmock.Sqlmock, status string) {
	mock.ExpectQuery(regexp.QuoteMeta(principalQuery)).WithArgs(testPrincipalID).
		WillReturnRows(sqlmock.NewRows([]string{
			"issuer", "subject", "status", "attributes", "revision", "created_at", "updated_at",
		}).AddRow(
			"https://identity.example", "subject-a", status, []byte(`{"department":"platform"}`),
			int64(3), authorizationTestTime.Add(-time.Hour), authorizationTestTime,
		))
}

func expectRoleGrant(mock sqlmock.Sqlmock) {
	permissions := []byte(`["routing.read","provider_catalog.read","agent.read","tool.read"]`)
	mock.ExpectQuery(regexp.QuoteMeta(roleGrantsQuery)).
		WithArgs(testPrincipalID, string(testNamespaceID)).
		WillReturnRows(sqlmock.NewRows([]string{
			"binding_id", "role_id", "scope_kind", "binding_namespace_id",
			"resource_type", "resource_id", "delegation_ceiling", "binding_status", "binding_revision",
			"role_namespace_id", "role_name", "role_display_name", "permissions", "builtin", "role_status", "role_revision",
		}).AddRow(
			testBindingID, testRoleID, "namespace", string(testNamespaceID),
			nil, nil, []byte(`[]`), "active", int64(4),
			nil, "viewer", "Viewer", permissions, true, "active", int64(1),
		))
}

func expectTeamGrant(mock sqlmock.Sqlmock, capabilities string, membershipRevision int64) {
	mock.ExpectQuery(regexp.QuoteMeta(principalUserLinkQuery)).
		WithArgs(testPrincipalID, testNamespaceID).
		WillReturnRows(sqlmock.NewRows([]string{"user_id", "revision", "status"}).
			AddRow(testUserID, int64(2), "active"))
	mock.ExpectQuery(regexp.QuoteMeta(selfServicePolicyQuery)).
		WithArgs(testNamespaceID).
		WillReturnRows(sqlmock.NewRows([]string{
			"allow_team_key_delegation", "team_admin_capabilities", "revision",
		}).AddRow(true, []byte(capabilities), int64(7)))
	mock.ExpectQuery(regexp.QuoteMeta(teamMembershipsQuery)).
		WithArgs(testNamespaceID, testUserID).
		WillReturnRows(sqlmock.NewRows([]string{
			"team_id", "role", "membership_status", "membership_revision",
			"created_at", "updated_at", "team_status",
		}).AddRow(
			testTeamID, "admin", "active", membershipRevision,
			authorizationTestTime.Add(-time.Hour), authorizationTestTime, "active",
		))
}

func TestLoadBuildsCurrentScopedAuthority(t *testing.T) {
	store, mock := newAuthorizationMock(t)
	mock.ExpectBegin()
	expectPrincipal(mock, "active")
	expectRoleGrant(mock)
	expectTeamGrant(mock, `["membership.manage","key.manage"]`, 5)
	mock.ExpectCommit()

	snapshot, err := store.Load(context.Background(), testPrincipalID, testNamespaceID)
	if err != nil {
		t.Fatal(err)
	}
	if len(snapshot.RoleGrants) != 1 || len(snapshot.TeamGrants) != 1 {
		t.Fatalf("unexpected grants: roles=%d teams=%d", len(snapshot.RoleGrants), len(snapshot.TeamGrants))
	}
	if !regexp.MustCompile(`^sha256:[a-f0-9]{64}$`).MatchString(snapshot.AuthorityDigest) {
		t.Fatalf("unexpected authority digest %q", snapshot.AuthorityDigest)
	}

	context := managementauthorization.EvaluationContext{
		Authenticated: true,
		RoleGrants:    snapshot.RoleGrants,
		TeamGrants:    snapshot.TeamGrants,
		Targets: map[string][]accesscontrol.ScopedTarget{
			"request_namespace": {{Scope: accesscontrol.NamespaceScope(testNamespaceID)}},
			"team":              {{Scope: accesscontrol.TeamScope(testNamespaceID, accesscontrol.TeamID(testTeamID))}},
		},
	}
	if err := managementauthorization.Evaluate(
		managementpermission.Require("provider_catalog.read", "request_namespace"), context,
	); err != nil {
		t.Fatalf("namespace role did not authorize Provider Catalog read: %v", err)
	}
	if err := managementauthorization.Evaluate(
		managementpermission.Require("key.manage", "team"), context,
	); err != nil {
		t.Fatalf("Team admin policy did not authorize Team key management: %v", err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatal(err)
	}
}

func TestLoadRejectsInactivePrincipalBeforeGrantQueries(t *testing.T) {
	store, mock := newAuthorizationMock(t)
	mock.ExpectBegin()
	expectPrincipal(mock, "disabled")
	mock.ExpectRollback()

	_, err := store.Load(context.Background(), testPrincipalID, testNamespaceID)
	if !errors.Is(err, ErrPrincipalInactive) {
		t.Fatalf("expected inactive principal, got %v", err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatal(err)
	}
}

func TestLoadRejectsUnknownTeamAdminCapability(t *testing.T) {
	store, mock := newAuthorizationMock(t)
	mock.ExpectBegin()
	expectPrincipal(mock, "active")
	expectRoleGrant(mock)
	mock.ExpectQuery(regexp.QuoteMeta(principalUserLinkQuery)).
		WithArgs(testPrincipalID, testNamespaceID).
		WillReturnRows(sqlmock.NewRows([]string{"user_id", "revision", "status"}).
			AddRow(testUserID, int64(2), "active"))
	mock.ExpectQuery(regexp.QuoteMeta(selfServicePolicyQuery)).
		WithArgs(testNamespaceID).
		WillReturnRows(sqlmock.NewRows([]string{
			"allow_team_key_delegation", "team_admin_capabilities", "revision",
		}).AddRow(false, []byte(`["role_binding.manage"]`), int64(7)))
	mock.ExpectRollback()

	_, err := store.Load(context.Background(), testPrincipalID, testNamespaceID)
	if !errors.Is(err, ErrStateInvalid) {
		t.Fatalf("expected invalid authorization state, got %v", err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatal(err)
	}
}

func TestAuthorityDigestChangesWithMembershipRevision(t *testing.T) {
	load := func(revision int64) string {
		store, mock := newAuthorizationMock(t)
		mock.ExpectBegin()
		expectPrincipal(mock, "active")
		expectRoleGrant(mock)
		expectTeamGrant(mock, `[]`, revision)
		mock.ExpectCommit()
		snapshot, err := store.Load(context.Background(), testPrincipalID, testNamespaceID)
		if err != nil {
			t.Fatal(err)
		}
		if err := mock.ExpectationsWereMet(); err != nil {
			t.Fatal(err)
		}
		return snapshot.AuthorityDigest
	}
	if first, second := load(5), load(6); first == second {
		t.Fatal("authority digest ignored Team membership revision")
	}
}
