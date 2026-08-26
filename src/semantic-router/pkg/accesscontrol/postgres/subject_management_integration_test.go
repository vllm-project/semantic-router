package postgres_test

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"net/netip"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	accesspostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol/postgres"
	controlpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/subjectmanagement"
)

const (
	subjectTestNamespaceID      = "11111111-1111-4111-8111-111111111111"
	subjectTestPrincipalID      = "22222222-2222-4222-8222-222222222222"
	subjectTestAccessID         = "33333333-3333-4333-8333-333333333333"
	subjectTestRateID           = "44444444-4444-4444-8444-444444444444"
	subjectTestAccessID2        = "55555555-5555-4555-8555-555555555555"
	subjectTestOtherNamespaceID = "66666666-6666-4666-8666-666666666666"
	subjectTestCrossAccessID    = "77777777-7777-4777-8777-777777777777"
	subjectTestDisabledAccessID = "88888888-8888-4888-8888-888888888888"
	subjectTestCrossRateID      = "99999999-9999-4999-8999-999999999999"
	subjectTestDisabledRateID   = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
)

func TestSubjectManagementPostgresEndToEnd(t *testing.T) {
	dsn := os.Getenv("VLLM_SR_CONTROL_PLANE_TEST_DATABASE_URL")
	if dsn == "" {
		dsn = os.Getenv("VLLM_SR_ACCESS_CONTROL_TEST_DATABASE_URL")
	}
	if dsn == "" {
		t.Skip("PostgreSQL subject Management test database is not configured")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	db := isolatedSubjectDatabase(t, ctx, dsn)
	if err := (controlpostgres.Migrator{DB: db}).Apply(ctx); err != nil {
		t.Fatal(err)
	}

	service := newSubjectManagementService(t, ctx, db)
	actor := subjectmanagement.Actor{
		PrincipalID: subjectTestPrincipalID, ActorChain: []string{subjectTestPrincipalID},
		RequestID: "request-1", SourceIP: netip.MustParseAddr("192.0.2.10"),
	}

	allResults := accesscontrol.ResultScope{NamespaceID: subjectTestNamespaceID, All: true}
	userID := assertSubjectUserLifecycle(t, ctx, service, actor, allResults)
	teamID := assertSubjectTeamLifecycle(t, ctx, db, service, actor, allResults)
	assertSubjectTeamPolicyValidation(t, ctx, db, service, actor)
	assertSubjectMembershipAndUpdate(t, ctx, service, actor, allResults, userID, teamID)
	assertSubjectDurableAccounting(t, ctx, db)
}

func newSubjectManagementService(
	t *testing.T,
	ctx context.Context,
	db *sql.DB,
) *subjectmanagement.Service {
	t.Helper()
	tx, beginErr := db.BeginTx(ctx, nil)
	if beginErr != nil {
		t.Fatal(beginErr)
	}
	statements := []struct {
		query string
		args  []any
	}{
		{`INSERT INTO access_namespaces
  (id,name,quota_partition_id,billing_currency,status) VALUES ($1,'test','test-partition','USD','active')`, []any{subjectTestNamespaceID}},
		{`INSERT INTO access_namespaces
  (id,name,quota_partition_id,billing_currency,status) VALUES ($1,'other','other-partition','USD','active')`, []any{subjectTestOtherNamespaceID}},
		{`INSERT INTO management_principals
  (id,issuer,subject,display_name,status) VALUES ($1,'test','actor','Actor','active')`, []any{subjectTestPrincipalID}},
		{`INSERT INTO access_policies
  (id,namespace_id,name,status) VALUES ($1,$2,'default-access','active')`, []any{subjectTestAccessID, subjectTestNamespaceID}},
		{`INSERT INTO access_policies
  (id,namespace_id,name,status) VALUES ($1,$2,'extended-access','active')`, []any{subjectTestAccessID2, subjectTestNamespaceID}},
		{`INSERT INTO access_policies
  (id,namespace_id,name,status) VALUES ($1,$2,'cross-access','active')`, []any{subjectTestCrossAccessID, subjectTestOtherNamespaceID}},
		{`INSERT INTO access_policies
  (id,namespace_id,name,status) VALUES ($1,$2,'disabled-access','disabled')`, []any{subjectTestDisabledAccessID, subjectTestNamespaceID}},
		{`INSERT INTO rate_limit_policies
  (id,namespace_id,name,status) VALUES ($1,$2,'default-rate','active')`, []any{subjectTestRateID, subjectTestNamespaceID}},
		{`INSERT INTO rate_limit_policies
  (id,namespace_id,name,status) VALUES ($1,$2,'cross-rate','active')`, []any{subjectTestCrossRateID, subjectTestOtherNamespaceID}},
		{`INSERT INTO rate_limit_policies
  (id,namespace_id,name,status) VALUES ($1,$2,'disabled-rate','disabled')`, []any{subjectTestDisabledRateID, subjectTestNamespaceID}},
		{`INSERT INTO self_service_policies
  (namespace_id,default_access_policy_id,default_rate_limit_policy_id,seed_version)
VALUES ($1,$2,$3,1)`, []any{subjectTestNamespaceID, subjectTestAccessID, subjectTestRateID}},
	}
	for _, statement := range statements {
		if _, err := tx.ExecContext(ctx, statement.query, statement.args...); err != nil {
			_ = tx.Rollback()
			t.Fatal(err)
		}
	}
	if err := tx.Commit(); err != nil {
		t.Fatal(err)
	}
	store, err := accesspostgres.New(db)
	if err != nil {
		t.Fatal(err)
	}
	repository, err := accesspostgres.NewSubjectRepository(store)
	if err != nil {
		t.Fatal(err)
	}
	commandCodec, err := managementcommand.NewCodec(securitykeyring.Symmetric{
		ActiveVersion: "v1", Keys: map[string][]byte{"v1": []byte(strings.Repeat("c", 32))},
	})
	if err != nil {
		t.Fatal(err)
	}
	service, err := subjectmanagement.NewService(subjectmanagement.Options{
		Repository: repository, CommandCodec: commandCodec,
		CursorKeyring: securitykeyring.Symmetric{ActiveVersion: "v1", Keys: map[string][]byte{
			"v1": []byte(strings.Repeat("p", 32)),
		}}, IdempotencyTTL: time.Hour,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(service.Close)
	return service
}

func assertSubjectUserLifecycle(
	t *testing.T,
	ctx context.Context,
	service *subjectmanagement.Service,
	actor subjectmanagement.Actor,
	scope accesscontrol.ResultScope,
) string {
	t.Helper()
	created := assertSubjectUserCreateAndReplay(t, ctx, service, actor)
	assertSecondSubjectUser(t, ctx, service, actor, created.ID)
	assertSubjectUserPagination(t, ctx, service, scope)
	assertSubjectUserScopeAndSearch(t, ctx, service, scope, created.ID)
	assertDeletedSubjectUserVisibility(t, ctx, service, actor, scope)
	return created.ID
}

func assertSubjectUserCreateAndReplay(
	t *testing.T,
	ctx context.Context,
	service *subjectmanagement.Service,
	actor subjectmanagement.Actor,
) subjectmanagement.MutationResult {
	t.Helper()
	request := subjectmanagement.CreateUserRequest{
		NamespaceID: subjectTestNamespaceID, Email: "FIRST@Example.COM", DisplayName: "First User",
		IdempotencyKey: "create-user-0123456789", Actor: actor,
	}
	created, createErr := service.CreateUser(ctx, request)
	if createErr != nil || created.Replayed || created.Revision != 1 {
		t.Fatalf("create User = %#v, %v", created, createErr)
	}
	request.Email = "first@example.com"
	replayed, replayErr := service.CreateUser(ctx, request)
	if replayErr != nil || !replayed.Replayed || replayed.ID != created.ID {
		t.Fatalf("replay User = %#v, %v", replayed, replayErr)
	}
	request.Email, request.DisplayName = "different@example.com", "Different"
	if _, err := service.CreateUser(ctx, request); !errors.Is(err, managementcommand.ErrConflict) {
		t.Fatalf("conflicting replay error = %v", err)
	}
	return created
}

func assertSecondSubjectUser(
	t *testing.T,
	ctx context.Context,
	service *subjectmanagement.Service,
	actor subjectmanagement.Actor,
	firstUserID string,
) {
	t.Helper()
	second, err := service.CreateUser(ctx, subjectmanagement.CreateUserRequest{
		NamespaceID: subjectTestNamespaceID, Email: "second@example.com", DisplayName: "Second User",
		IdempotencyKey: "create-user-9876543210", Actor: actor,
	})
	if err != nil || second.ID == firstUserID {
		t.Fatalf("create second User = %#v, %v", second, err)
	}
}

func assertSubjectUserPagination(
	t *testing.T,
	ctx context.Context,
	service *subjectmanagement.Service,
	scope accesscontrol.ResultScope,
) {
	t.Helper()
	firstPage, err := service.ListUsers(ctx, subjectmanagement.ListRequest{
		NamespaceID: subjectTestNamespaceID, PageSize: 1, Scope: scope,
	})
	if err != nil || len(firstPage.Items) != 1 || !firstPage.HasMore || firstPage.NextCursor == "" {
		t.Fatalf("first User page = %#v, %v", firstPage, err)
	}
	secondPage, err := service.ListUsers(ctx, subjectmanagement.ListRequest{
		NamespaceID: subjectTestNamespaceID, PageSize: 1, Cursor: firstPage.NextCursor, Scope: scope,
	})
	if err != nil || len(secondPage.Items) != 1 || secondPage.Items[0].ID == firstPage.Items[0].ID {
		t.Fatalf("second User page = %#v, %v", secondPage, err)
	}
}

func assertSubjectUserScopeAndSearch(
	t *testing.T,
	ctx context.Context,
	service *subjectmanagement.Service,
	scope accesscontrol.ResultScope,
	userID string,
) {
	t.Helper()
	narrow, err := service.ListUsers(ctx, subjectmanagement.ListRequest{
		NamespaceID: subjectTestNamespaceID, PageSize: 10,
		Scope: accesscontrol.ResultScope{
			NamespaceID: subjectTestNamespaceID,
			UserIDs:     []accesscontrol.UserID{accesscontrol.UserID(userID)},
		},
	})
	if err != nil || len(narrow.Items) != 1 || narrow.Items[0].ID != userID || narrow.HasMore {
		t.Fatalf("exact User scope page = %#v, %v", narrow, err)
	}
	searched, err := service.ListUsers(ctx, subjectmanagement.ListRequest{
		NamespaceID: subjectTestNamespaceID, Search: "FIRST@", PageSize: 1, Scope: scope,
	})
	if err != nil || len(searched.Items) != 1 || searched.Items[0].ID != userID || searched.HasMore {
		t.Fatalf("searched User page = %#v, %v", searched, err)
	}
}

func assertDeletedSubjectUserVisibility(
	t *testing.T,
	ctx context.Context,
	service *subjectmanagement.Service,
	actor subjectmanagement.Actor,
	scope accesscontrol.ResultScope,
) {
	t.Helper()
	deleted, err := service.CreateUser(ctx, subjectmanagement.CreateUserRequest{
		NamespaceID: subjectTestNamespaceID, Email: "deleted@example.com", DisplayName: "Deleted User",
		IdempotencyKey: "create-user-deleted-0123456789", Actor: actor,
	})
	if err != nil {
		t.Fatalf("create User to delete = %#v, %v", deleted, err)
	}
	if _, err := service.DeleteUser(ctx, subjectmanagement.DeleteUserRequest{
		NamespaceID: subjectTestNamespaceID, UserID: deleted.ID, ExpectedRevision: deleted.Revision, Actor: actor,
	}); err != nil {
		t.Fatalf("delete User = %v", err)
	}
	assertDefaultSubjectUsersExclude(t, ctx, service, scope, deleted.ID)
	assertDeletedSubjectUserSearch(t, ctx, service, scope, deleted.ID)
}

func assertDefaultSubjectUsersExclude(
	t *testing.T,
	ctx context.Context,
	service *subjectmanagement.Service,
	scope accesscontrol.ResultScope,
	excludedID string,
) {
	t.Helper()
	visible, err := service.ListUsers(ctx, subjectmanagement.ListRequest{
		NamespaceID: subjectTestNamespaceID, PageSize: 10, Scope: scope,
	})
	if err != nil || len(visible.Items) != 2 {
		t.Fatalf("default User page after delete = %#v, %v", visible, err)
	}
	for _, item := range visible.Items {
		if item.ID == excludedID {
			t.Fatalf("deleted User remained in default page: %#v", visible)
		}
	}
}

func assertDeletedSubjectUserSearch(
	t *testing.T,
	ctx context.Context,
	service *subjectmanagement.Service,
	scope accesscontrol.ResultScope,
	deletedID string,
) {
	t.Helper()
	deletedPage, err := service.ListUsers(ctx, subjectmanagement.ListRequest{
		NamespaceID: subjectTestNamespaceID, Status: string(accesscontrol.UserStatusDeleted),
		Search: "deleted@", PageSize: 10, Scope: scope,
	})
	if err != nil || len(deletedPage.Items) != 1 || deletedPage.Items[0].ID != deletedID ||
		deletedPage.Items[0].Status != accesscontrol.UserStatusDeleted || deletedPage.Items[0].DeletedAt == nil {
		t.Fatalf("explicitly deleted User page = %#v, %v", deletedPage, err)
	}
}

func assertSubjectTeamLifecycle(
	t *testing.T,
	ctx context.Context,
	db *sql.DB,
	service *subjectmanagement.Service,
	actor subjectmanagement.Actor,
	scope accesscontrol.ResultScope,
) string {
	t.Helper()
	created := assertSubjectTeamCreateAndReplay(t, ctx, service, actor)
	assertSubjectTeamSearchAndState(t, ctx, db, service, scope, created.ID)
	assertSubjectTeamPolicyBindings(t, ctx, db, created.ID)
	return created.ID
}

func assertSubjectTeamCreateAndReplay(
	t *testing.T,
	ctx context.Context,
	service *subjectmanagement.Service,
	actor subjectmanagement.Actor,
) subjectmanagement.MutationResult {
	t.Helper()
	request := subjectmanagement.CreateTeamRequest{
		NamespaceID: subjectTestNamespaceID, Name: "Platform", Description: "Platform team",
		AccessPolicyIDs:   []string{subjectTestAccessID2, subjectTestAccessID},
		RateLimitPolicyID: subjectTestRateID, IdempotencyKey: "create-team-0123456789", Actor: actor,
	}
	created, createErr := service.CreateTeam(ctx, request)
	if createErr != nil {
		t.Fatal(createErr)
	}
	request.AccessPolicyIDs = []string{subjectTestAccessID, subjectTestAccessID2}
	replayed, replayErr := service.CreateTeam(ctx, request)
	if replayErr != nil || !replayed.Replayed || replayed.ID != created.ID {
		t.Fatalf("replay Team with canonical policy order = %#v, %v", replayed, replayErr)
	}
	request.AccessPolicyIDs = []string{subjectTestAccessID}
	if _, err := service.CreateTeam(ctx, request); !errors.Is(err, managementcommand.ErrConflict) {
		t.Fatalf("conflicting Team policy selection error = %v", err)
	}
	return created
}

func assertSubjectTeamSearchAndState(
	t *testing.T,
	ctx context.Context,
	db *sql.DB,
	service *subjectmanagement.Service,
	scope accesscontrol.ResultScope,
	teamID string,
) {
	t.Helper()
	searched, searchErr := service.ListTeams(ctx, subjectmanagement.ListRequest{
		NamespaceID: subjectTestNamespaceID, Search: "plat", PageSize: 1, Scope: scope,
	})
	if searchErr != nil || len(searched.Items) != 1 || searched.Items[0].ID != teamID {
		t.Fatalf("searched Team page = %#v, %v", searched, searchErr)
	}
	var status string
	if err := db.QueryRowContext(ctx, `SELECT status FROM access_teams WHERE id=$1`, teamID).Scan(&status); err != nil || status != "active" {
		t.Fatalf("Team status = %q, %v", status, err)
	}
}

func assertSubjectTeamPolicyBindings(t *testing.T, ctx context.Context, db *sql.DB, teamID string) {
	t.Helper()
	var accessBindings, rateBindings int
	bindingErr := db.QueryRowContext(ctx, `SELECT
  (SELECT count(*) FROM access_policy_bindings
   WHERE subject_id=$1 AND policy_id IN ($2,$3) AND status='active'),
  (SELECT count(*) FROM rate_limit_bindings
   WHERE subject_id=$1 AND policy_id=$4 AND binding_mode='allocation' AND status='active')`,
		teamID, subjectTestAccessID, subjectTestAccessID2, subjectTestRateID,
	).Scan(&accessBindings, &rateBindings)
	if bindingErr != nil || accessBindings != 2 || rateBindings != 1 {
		t.Fatalf("Team policy bindings = %d/%d, %v", accessBindings, rateBindings, bindingErr)
	}
}

func assertSubjectTeamPolicyValidation(
	t *testing.T,
	ctx context.Context,
	db *sql.DB,
	service *subjectmanagement.Service,
	actor subjectmanagement.Actor,
) {
	t.Helper()
	selections := []struct {
		name      string
		accessIDs []string
		rateID    string
	}{
		{name: "Invalid missing", accessIDs: []string{uuid.NewString()}, rateID: subjectTestRateID},
		{name: "Invalid cross access", accessIDs: []string{subjectTestCrossAccessID}, rateID: subjectTestRateID},
		{name: "Invalid disabled access", accessIDs: []string{subjectTestDisabledAccessID}, rateID: subjectTestRateID},
		{name: "Invalid wrong access kind", accessIDs: []string{subjectTestRateID}, rateID: subjectTestRateID},
		{name: "Invalid cross rate", accessIDs: []string{subjectTestAccessID}, rateID: subjectTestCrossRateID},
		{name: "Invalid disabled rate", accessIDs: []string{subjectTestAccessID}, rateID: subjectTestDisabledRateID},
		{name: "Invalid wrong rate kind", accessIDs: []string{subjectTestAccessID}, rateID: subjectTestAccessID},
	}
	for index, selection := range selections {
		_, err := service.CreateTeam(ctx, subjectmanagement.CreateTeamRequest{
			NamespaceID: subjectTestNamespaceID, Name: selection.name, AccessPolicyIDs: selection.accessIDs,
			RateLimitPolicyID: selection.rateID,
			IdempotencyKey:    fmt.Sprintf("create-team-invalid-%02d-0123456789", index), Actor: actor,
		})
		if !errors.Is(err, subjectmanagement.ErrPolicySelectionUnavailable) {
			t.Fatalf("%s policy selection error = %v", selection.name, err)
		}
	}
	var unavailableTeams int
	if err := db.QueryRowContext(ctx, `SELECT count(*) FROM access_teams WHERE namespace_id=$1 AND name LIKE 'Invalid %'`,
		subjectTestNamespaceID,
	).Scan(&unavailableTeams); err != nil || unavailableTeams != 0 {
		t.Fatalf("partially created Teams = %d, %v", unavailableTeams, err)
	}
	defaults, defaultsErr := service.ResolveTeamDefaults(ctx, subjectTestNamespaceID)
	if defaultsErr != nil {
		t.Fatal(defaultsErr)
	}
	if _, err := service.CreateTeam(ctx, subjectmanagement.CreateTeamRequest{
		NamespaceID: subjectTestNamespaceID, Name: "Default Team",
		AccessPolicyIDs: []string{defaults.AccessPolicyID}, RateLimitPolicyID: defaults.RateLimitPolicyID,
		NamespaceDefaults: &defaults, UseDefaultAccessPolicy: true, UseDefaultRateLimitPolicy: true,
		IdempotencyKey: "create-team-defaults-0123456789", Actor: actor,
	}); err != nil {
		t.Fatalf("create Team from namespace defaults = %v", err)
	}
	staleDefaults, staleDefaultsErr := service.ResolveTeamDefaults(ctx, subjectTestNamespaceID)
	if staleDefaultsErr != nil {
		t.Fatal(staleDefaultsErr)
	}
	if _, err := db.ExecContext(ctx, `UPDATE self_service_policies SET revision=revision+1 WHERE namespace_id=$1`, subjectTestNamespaceID); err != nil {
		t.Fatal(err)
	}
	_, staleCreateErr := service.CreateTeam(ctx, subjectmanagement.CreateTeamRequest{
		NamespaceID: subjectTestNamespaceID, Name: "Stale defaults",
		AccessPolicyIDs: []string{staleDefaults.AccessPolicyID}, RateLimitPolicyID: staleDefaults.RateLimitPolicyID,
		NamespaceDefaults: &staleDefaults, UseDefaultAccessPolicy: true, UseDefaultRateLimitPolicy: true,
		IdempotencyKey: "create-team-stale-defaults-0123456789", Actor: actor,
	})
	if !errors.Is(staleCreateErr, subjectmanagement.ErrDefaultsUnavailable) {
		t.Fatalf("stale Team defaults error = %v", staleCreateErr)
	}
	var staleTeams int
	if err := db.QueryRowContext(ctx, `SELECT count(*) FROM access_teams WHERE namespace_id=$1 AND name=$2`,
		subjectTestNamespaceID, "Stale defaults",
	).Scan(&staleTeams); err != nil || staleTeams != 0 {
		t.Fatalf("Team created from stale defaults = %d, %v", staleTeams, err)
	}
}

func assertSubjectMembershipAndUpdate(
	t *testing.T,
	ctx context.Context,
	service *subjectmanagement.Service,
	actor subjectmanagement.Actor,
	scope accesscontrol.ResultScope,
	userID, teamID string,
) {
	t.Helper()
	assertSubjectMembershipCreated(t, ctx, service, actor, userID, teamID)
	assertSubjectUserMemberships(t, ctx, service, scope, userID)
	assertSubjectTeamMembers(t, ctx, service, scope, teamID)
	assertSubjectMembershipScope(t, ctx, service, userID)
	assertSubjectTeamUpdate(t, ctx, service, actor, teamID)
}

func assertSubjectMembershipCreated(
	t *testing.T,
	ctx context.Context,
	service *subjectmanagement.Service,
	actor subjectmanagement.Actor,
	userID, teamID string,
) {
	t.Helper()
	membership, err := service.PutMembership(ctx, subjectmanagement.PutMembershipRequest{
		NamespaceID: subjectTestNamespaceID, TeamID: teamID, UserID: userID,
		Role: accesscontrol.TeamRoleAdmin, IdempotencyKey: "membership-0123456789", Actor: actor,
	})
	if err != nil || membership.Revision != 1 {
		t.Fatalf("put membership = %#v, %v", membership, err)
	}
}

func assertSubjectUserMemberships(
	t *testing.T,
	ctx context.Context,
	service *subjectmanagement.Service,
	scope accesscontrol.ResultScope,
	userID string,
) {
	t.Helper()
	userMemberships, err := service.ListUserMemberships(ctx, subjectmanagement.MembershipListRequest{
		NamespaceID: subjectTestNamespaceID, UserID: userID, PageSize: 10,
		IncludeTotal: true, Scope: scope,
	})
	if err != nil || len(userMemberships.Items) != 1 || userMemberships.Items[0].TeamName != "Platform" ||
		userMemberships.TotalCount == nil || *userMemberships.TotalCount != 1 {
		t.Fatalf("User memberships = %#v, %v", userMemberships, err)
	}
}

func assertSubjectTeamMembers(
	t *testing.T,
	ctx context.Context,
	service *subjectmanagement.Service,
	scope accesscontrol.ResultScope,
	teamID string,
) {
	t.Helper()
	teamMembers, err := service.ListTeamMembers(ctx, subjectmanagement.MembershipListRequest{
		NamespaceID: subjectTestNamespaceID, TeamID: teamID, PageSize: 10,
		IncludeTotal: true, Scope: scope,
	})
	if err != nil || len(teamMembers.Items) != 1 || teamMembers.Items[0].DisplayName != "First User" ||
		teamMembers.Items[0].Email != "first@example.com" || teamMembers.TotalCount == nil ||
		*teamMembers.TotalCount != 1 {
		t.Fatalf("Team members = %#v, %v", teamMembers, err)
	}
}

func assertSubjectMembershipScope(
	t *testing.T,
	ctx context.Context,
	service *subjectmanagement.Service,
	userID string,
) {
	t.Helper()
	hidden, err := service.ListUserMemberships(ctx, subjectmanagement.MembershipListRequest{
		NamespaceID: subjectTestNamespaceID, UserID: userID, PageSize: 10, IncludeTotal: true,
		Scope: accesscontrol.ResultScope{NamespaceID: subjectTestNamespaceID},
	})
	if err != nil || len(hidden.Items) != 0 || hidden.TotalCount == nil || *hidden.TotalCount != 0 {
		t.Fatalf("permission-filtered User memberships = %#v, %v", hidden, err)
	}
}

func assertSubjectTeamUpdate(
	t *testing.T,
	ctx context.Context,
	service *subjectmanagement.Service,
	actor subjectmanagement.Actor,
	teamID string,
) {
	t.Helper()
	newName := "Platform Engineering"
	request := subjectmanagement.UpdateTeamRequest{
		NamespaceID: subjectTestNamespaceID, TeamID: teamID, ExpectedRevision: 1,
		Name: &newName, Actor: actor,
	}
	updated, err := service.UpdateTeam(ctx, request)
	if err != nil || updated.Revision != 2 {
		t.Fatalf("update Team = %#v, %v", updated, err)
	}
	if _, err := service.UpdateTeam(ctx, request); !errors.Is(err, subjectmanagement.ErrRevisionConflict) {
		t.Fatalf("stale Team CAS error = %v", err)
	}
}

func assertSubjectDurableAccounting(t *testing.T, ctx context.Context, db *sql.DB) {
	t.Helper()
	var commands, audits, outbox int
	if err := db.QueryRowContext(ctx, `SELECT
  (SELECT count(*) FROM management_idempotency),
  (SELECT count(*) FROM access_audit_events),
  (SELECT count(*) FROM policy_outbox)`).Scan(&commands, &audits, &outbox); err != nil {
		t.Fatal(err)
	}
	if commands != 7 || audits < 8 || outbox != audits {
		t.Fatalf("durable accounting commands/audits/outbox = %d/%d/%d", commands, audits, outbox)
	}
}

func isolatedSubjectDatabase(t *testing.T, ctx context.Context, dsn string) *sql.DB {
	t.Helper()
	admin, isolatedSubjectDatabaseErr := sql.Open("postgres", dsn)
	if isolatedSubjectDatabaseErr != nil {
		t.Fatal(isolatedSubjectDatabaseErr)
	}
	t.Cleanup(func() { _ = admin.Close() })
	if err := admin.PingContext(ctx); err != nil {
		t.Fatal(err)
	}
	schema := "vsr_subject_" + strings.ReplaceAll(uuid.NewString(), "-", "")
	if _, err := admin.ExecContext(ctx, "CREATE SCHEMA "+pq.QuoteIdentifier(schema)); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cleanup, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_, _ = admin.ExecContext(cleanup, "DROP SCHEMA "+pq.QuoteIdentifier(schema)+" CASCADE")
	})
	parsed, isolatedSubjectDatabaseErr := url.Parse(dsn)
	if isolatedSubjectDatabaseErr != nil {
		t.Fatal(isolatedSubjectDatabaseErr)
	}
	query := parsed.Query()
	query.Set("search_path", schema)
	parsed.RawQuery = query.Encode()
	db, isolatedSubjectDatabaseErr := sql.Open("postgres", parsed.String())
	if isolatedSubjectDatabaseErr != nil {
		t.Fatal(isolatedSubjectDatabaseErr)
	}
	t.Cleanup(func() { _ = db.Close() })
	return db
}
