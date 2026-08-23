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

	const (
		namespaceID      = "11111111-1111-4111-8111-111111111111"
		principalID      = "22222222-2222-4222-8222-222222222222"
		accessID         = "33333333-3333-4333-8333-333333333333"
		rateID           = "44444444-4444-4444-8444-444444444444"
		accessID2        = "55555555-5555-4555-8555-555555555555"
		otherNamespaceID = "66666666-6666-4666-8666-666666666666"
		crossAccessID    = "77777777-7777-4777-8777-777777777777"
		disabledAccessID = "88888888-8888-4888-8888-888888888888"
		crossRateID      = "99999999-9999-4999-8999-999999999999"
		disabledRateID   = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
	)
	seedTx, testSubjectManagementPostgresEndToEndErr := db.BeginTx(ctx, nil)
	if testSubjectManagementPostgresEndToEndErr != nil {
		t.Fatal(testSubjectManagementPostgresEndToEndErr)
	}
	seedStatements := []struct {
		query string
		args  []any
	}{
		{`INSERT INTO access_namespaces
  (id,name,quota_partition_id,billing_currency,status) VALUES ($1,'test','test-partition','USD','active')`, []any{namespaceID}},
		{`INSERT INTO access_namespaces
  (id,name,quota_partition_id,billing_currency,status) VALUES ($1,'other','other-partition','USD','active')`, []any{otherNamespaceID}},
		{`INSERT INTO management_principals
  (id,issuer,subject,display_name,status) VALUES ($1,'test','actor','Actor','active')`, []any{principalID}},
		{`INSERT INTO access_policies
  (id,namespace_id,name,status) VALUES ($1,$2,'default-access','active')`, []any{accessID, namespaceID}},
		{`INSERT INTO access_policies
  (id,namespace_id,name,status) VALUES ($1,$2,'extended-access','active')`, []any{accessID2, namespaceID}},
		{`INSERT INTO access_policies
  (id,namespace_id,name,status) VALUES ($1,$2,'cross-access','active')`, []any{crossAccessID, otherNamespaceID}},
		{`INSERT INTO access_policies
  (id,namespace_id,name,status) VALUES ($1,$2,'disabled-access','disabled')`, []any{disabledAccessID, namespaceID}},
		{`INSERT INTO rate_limit_policies
  (id,namespace_id,name,status) VALUES ($1,$2,'default-rate','active')`, []any{rateID, namespaceID}},
		{`INSERT INTO rate_limit_policies
  (id,namespace_id,name,status) VALUES ($1,$2,'cross-rate','active')`, []any{crossRateID, otherNamespaceID}},
		{`INSERT INTO rate_limit_policies
  (id,namespace_id,name,status) VALUES ($1,$2,'disabled-rate','disabled')`, []any{disabledRateID, namespaceID}},
		{`INSERT INTO self_service_policies
  (namespace_id,default_access_policy_id,default_rate_limit_policy_id,seed_version)
VALUES ($1,$2,$3,1)`, []any{namespaceID, accessID, rateID}},
	}
	for _, statement := range seedStatements {
		if _, err := seedTx.ExecContext(ctx, statement.query, statement.args...); err != nil {
			_ = seedTx.Rollback()
			t.Fatal(err)
		}
	}
	if err := seedTx.Commit(); err != nil {
		t.Fatal(err)
	}
	store, testSubjectManagementPostgresEndToEndErr := accesspostgres.New(db)
	if testSubjectManagementPostgresEndToEndErr != nil {
		t.Fatal(testSubjectManagementPostgresEndToEndErr)
	}
	repository, testSubjectManagementPostgresEndToEndErr := accesspostgres.NewSubjectRepository(store)
	if testSubjectManagementPostgresEndToEndErr != nil {
		t.Fatal(testSubjectManagementPostgresEndToEndErr)
	}
	commandCodec, testSubjectManagementPostgresEndToEndErr := managementcommand.NewCodec(securitykeyring.Symmetric{
		ActiveVersion: "v1", Keys: map[string][]byte{"v1": []byte(strings.Repeat("c", 32))},
	})
	if testSubjectManagementPostgresEndToEndErr != nil {
		t.Fatal(testSubjectManagementPostgresEndToEndErr)
	}
	service, testSubjectManagementPostgresEndToEndErr := subjectmanagement.NewService(subjectmanagement.Options{
		Repository: repository, CommandCodec: commandCodec,
		CursorKeyring: securitykeyring.Symmetric{ActiveVersion: "v1", Keys: map[string][]byte{
			"v1": []byte(strings.Repeat("p", 32)),
		}}, IdempotencyTTL: time.Hour,
	})
	if testSubjectManagementPostgresEndToEndErr != nil {
		t.Fatal(testSubjectManagementPostgresEndToEndErr)
	}
	t.Cleanup(service.Close)
	actor := subjectmanagement.Actor{
		PrincipalID: principalID, ActorChain: []string{principalID},
		RequestID: "request-1", SourceIP: netip.MustParseAddr("192.0.2.10"),
	}

	userResult, testSubjectManagementPostgresEndToEndErr := service.CreateUser(ctx, subjectmanagement.CreateUserRequest{
		NamespaceID: namespaceID, Email: "FIRST@Example.COM", DisplayName: "First User",
		IdempotencyKey: "create-user-0123456789", Actor: actor,
	})
	if testSubjectManagementPostgresEndToEndErr != nil || userResult.Replayed || userResult.Revision != 1 {
		t.Fatalf("create User = %#v, %v", userResult, testSubjectManagementPostgresEndToEndErr)
	}
	replayed, testSubjectManagementPostgresEndToEndErr := service.CreateUser(ctx, subjectmanagement.CreateUserRequest{
		NamespaceID: namespaceID, Email: "first@example.com", DisplayName: "First User",
		IdempotencyKey: "create-user-0123456789", Actor: actor,
	})
	if testSubjectManagementPostgresEndToEndErr != nil || !replayed.Replayed || replayed.ID != userResult.ID {
		t.Fatalf("replay User = %#v, %v", replayed, testSubjectManagementPostgresEndToEndErr)
	}
	if _, err := service.CreateUser(ctx, subjectmanagement.CreateUserRequest{
		NamespaceID: namespaceID, Email: "different@example.com", DisplayName: "Different",
		IdempotencyKey: "create-user-0123456789", Actor: actor,
	}); !errors.Is(err, managementcommand.ErrConflict) {
		t.Fatalf("conflicting replay error = %v", err)
	}

	secondResult, testSubjectManagementPostgresEndToEndErr := service.CreateUser(ctx, subjectmanagement.CreateUserRequest{
		NamespaceID: namespaceID, Email: "second@example.com", DisplayName: "Second User",
		IdempotencyKey: "create-user-9876543210", Actor: actor,
	})
	if testSubjectManagementPostgresEndToEndErr != nil || secondResult.ID == userResult.ID {
		t.Fatalf("create second User = %#v, %v", secondResult, testSubjectManagementPostgresEndToEndErr)
	}
	allResults := accesscontrol.ResultScope{NamespaceID: namespaceID, All: true}
	firstPage, testSubjectManagementPostgresEndToEndErr := service.ListUsers(ctx, subjectmanagement.ListRequest{
		NamespaceID: namespaceID, PageSize: 1, Scope: allResults,
	})
	if testSubjectManagementPostgresEndToEndErr != nil || len(firstPage.Items) != 1 || !firstPage.HasMore || firstPage.NextCursor == "" {
		t.Fatalf("first User page = %#v, %v", firstPage, testSubjectManagementPostgresEndToEndErr)
	}
	secondPage, testSubjectManagementPostgresEndToEndErr := service.ListUsers(ctx, subjectmanagement.ListRequest{
		NamespaceID: namespaceID, PageSize: 1, Cursor: firstPage.NextCursor, Scope: allResults,
	})
	if testSubjectManagementPostgresEndToEndErr != nil || len(secondPage.Items) != 1 || secondPage.Items[0].ID == firstPage.Items[0].ID {
		t.Fatalf("second User page = %#v, %v", secondPage, testSubjectManagementPostgresEndToEndErr)
	}
	narrowUsers, testSubjectManagementPostgresEndToEndErr := service.ListUsers(ctx, subjectmanagement.ListRequest{
		NamespaceID: namespaceID, PageSize: 10,
		Scope: accesscontrol.ResultScope{
			NamespaceID: namespaceID,
			UserIDs:     []accesscontrol.UserID{accesscontrol.UserID(userResult.ID)},
		},
	})
	if testSubjectManagementPostgresEndToEndErr != nil || len(narrowUsers.Items) != 1 || narrowUsers.Items[0].ID != userResult.ID || narrowUsers.HasMore {
		t.Fatalf("exact User scope page = %#v, %v", narrowUsers, testSubjectManagementPostgresEndToEndErr)
	}
	searchedUsers, testSubjectManagementPostgresEndToEndErr := service.ListUsers(ctx, subjectmanagement.ListRequest{
		NamespaceID: namespaceID, Search: "FIRST@", PageSize: 1, Scope: allResults,
	})
	if testSubjectManagementPostgresEndToEndErr != nil || len(searchedUsers.Items) != 1 || searchedUsers.Items[0].ID != userResult.ID || searchedUsers.HasMore {
		t.Fatalf("searched User page = %#v, %v", searchedUsers, testSubjectManagementPostgresEndToEndErr)
	}

	teamResult, testSubjectManagementPostgresEndToEndErr := service.CreateTeam(ctx, subjectmanagement.CreateTeamRequest{
		NamespaceID: namespaceID, Name: "Platform", Description: "Platform team",
		AccessPolicyIDs: []string{accessID2, accessID}, RateLimitPolicyID: rateID,
		IdempotencyKey: "create-team-0123456789", Actor: actor,
	})
	if testSubjectManagementPostgresEndToEndErr != nil {
		t.Fatal(testSubjectManagementPostgresEndToEndErr)
	}
	replayedTeam, testSubjectManagementPostgresEndToEndErr := service.CreateTeam(ctx, subjectmanagement.CreateTeamRequest{
		NamespaceID: namespaceID, Name: "Platform", Description: "Platform team",
		AccessPolicyIDs: []string{accessID, accessID2}, RateLimitPolicyID: rateID,
		IdempotencyKey: "create-team-0123456789", Actor: actor,
	})
	if testSubjectManagementPostgresEndToEndErr != nil || !replayedTeam.Replayed || replayedTeam.ID != teamResult.ID {
		t.Fatalf("replay Team with canonical policy order = %#v, %v", replayedTeam, testSubjectManagementPostgresEndToEndErr)
	}
	if _, err := service.CreateTeam(ctx, subjectmanagement.CreateTeamRequest{
		NamespaceID: namespaceID, Name: "Platform", Description: "Platform team",
		AccessPolicyIDs: []string{accessID}, RateLimitPolicyID: rateID,
		IdempotencyKey: "create-team-0123456789", Actor: actor,
	}); !errors.Is(err, managementcommand.ErrConflict) {
		t.Fatalf("conflicting Team policy selection error = %v", err)
	}
	searchedTeams, testSubjectManagementPostgresEndToEndErr := service.ListTeams(ctx, subjectmanagement.ListRequest{
		NamespaceID: namespaceID, Search: "plat", PageSize: 1, Scope: allResults,
	})
	if testSubjectManagementPostgresEndToEndErr != nil || len(searchedTeams.Items) != 1 || searchedTeams.Items[0].ID != teamResult.ID {
		t.Fatalf("searched Team page = %#v, %v", searchedTeams, testSubjectManagementPostgresEndToEndErr)
	}
	var teamStatus string
	if err := db.QueryRowContext(ctx, `SELECT status FROM access_teams WHERE id=$1`, teamResult.ID).Scan(&teamStatus); err != nil || teamStatus != "active" {
		t.Fatalf("Team status = %q, %v", teamStatus, err)
	}
	var accessBindings, rateBindings int
	if err := db.QueryRowContext(ctx, `SELECT
  (SELECT count(*) FROM access_policy_bindings
   WHERE subject_id=$1 AND policy_id IN ($2,$3) AND status='active'),
  (SELECT count(*) FROM rate_limit_bindings
   WHERE subject_id=$1 AND policy_id=$4 AND binding_mode='allocation' AND status='active')`,
		teamResult.ID, accessID, accessID2, rateID).Scan(&accessBindings, &rateBindings); err != nil ||
		accessBindings != 2 || rateBindings != 1 {
		t.Fatalf("Team policy bindings = %d/%d, %v", accessBindings, rateBindings, err)
	}
	invalidSelections := []struct {
		name      string
		accessIDs []string
		rateID    string
	}{
		{name: "Invalid missing", accessIDs: []string{uuid.NewString()}, rateID: rateID},
		{name: "Invalid cross access", accessIDs: []string{crossAccessID}, rateID: rateID},
		{name: "Invalid disabled access", accessIDs: []string{disabledAccessID}, rateID: rateID},
		{name: "Invalid wrong access kind", accessIDs: []string{rateID}, rateID: rateID},
		{name: "Invalid cross rate", accessIDs: []string{accessID}, rateID: crossRateID},
		{name: "Invalid disabled rate", accessIDs: []string{accessID}, rateID: disabledRateID},
		{name: "Invalid wrong rate kind", accessIDs: []string{accessID}, rateID: accessID},
	}
	for index, selection := range invalidSelections {
		if _, err := service.CreateTeam(ctx, subjectmanagement.CreateTeamRequest{
			NamespaceID: namespaceID, Name: selection.name, AccessPolicyIDs: selection.accessIDs,
			RateLimitPolicyID: selection.rateID,
			IdempotencyKey:    fmt.Sprintf("create-team-invalid-%02d-0123456789", index), Actor: actor,
		}); !errors.Is(err, subjectmanagement.ErrPolicySelectionUnavailable) {
			t.Fatalf("%s policy selection error = %v", selection.name, err)
		}
	}
	var unavailableTeams int
	if err := db.QueryRowContext(ctx, `SELECT count(*) FROM access_teams WHERE namespace_id=$1 AND name LIKE 'Invalid %'`,
		namespaceID).Scan(&unavailableTeams); err != nil || unavailableTeams != 0 {
		t.Fatalf("partially created Teams = %d, %v", unavailableTeams, err)
	}
	defaults, testSubjectManagementPostgresEndToEndErr := service.ResolveTeamDefaults(ctx, namespaceID)
	if testSubjectManagementPostgresEndToEndErr != nil {
		t.Fatal(testSubjectManagementPostgresEndToEndErr)
	}
	if _, err := service.CreateTeam(ctx, subjectmanagement.CreateTeamRequest{
		NamespaceID: namespaceID, Name: "Default Team", AccessPolicyIDs: []string{defaults.AccessPolicyID},
		RateLimitPolicyID: defaults.RateLimitPolicyID, NamespaceDefaults: &defaults,
		UseDefaultAccessPolicy: true, UseDefaultRateLimitPolicy: true,
		IdempotencyKey: "create-team-defaults-0123456789", Actor: actor,
	}); err != nil {
		t.Fatalf("create Team from namespace defaults = %v", err)
	}
	staleDefaults, testSubjectManagementPostgresEndToEndErr := service.ResolveTeamDefaults(ctx, namespaceID)
	if testSubjectManagementPostgresEndToEndErr != nil {
		t.Fatal(testSubjectManagementPostgresEndToEndErr)
	}
	if _, err := db.ExecContext(ctx, `UPDATE self_service_policies SET revision=revision+1 WHERE namespace_id=$1`, namespaceID); err != nil {
		t.Fatal(err)
	}
	if _, err := service.CreateTeam(ctx, subjectmanagement.CreateTeamRequest{
		NamespaceID: namespaceID, Name: "Stale defaults", AccessPolicyIDs: []string{staleDefaults.AccessPolicyID},
		RateLimitPolicyID: staleDefaults.RateLimitPolicyID, NamespaceDefaults: &staleDefaults,
		UseDefaultAccessPolicy: true, UseDefaultRateLimitPolicy: true,
		IdempotencyKey: "create-team-stale-defaults-0123456789", Actor: actor,
	}); !errors.Is(err, subjectmanagement.ErrDefaultsUnavailable) {
		t.Fatalf("stale Team defaults error = %v", err)
	}
	var staleTeams int
	if err := db.QueryRowContext(ctx, `SELECT count(*) FROM access_teams WHERE namespace_id=$1 AND name=$2`,
		namespaceID, "Stale defaults").Scan(&staleTeams); err != nil || staleTeams != 0 {
		t.Fatalf("Team created from stale defaults = %d, %v", staleTeams, err)
	}

	membership, testSubjectManagementPostgresEndToEndErr := service.PutMembership(ctx, subjectmanagement.PutMembershipRequest{
		NamespaceID: namespaceID, TeamID: teamResult.ID, UserID: userResult.ID,
		Role: accesscontrol.TeamRoleAdmin, IdempotencyKey: "membership-0123456789", Actor: actor,
	})
	if testSubjectManagementPostgresEndToEndErr != nil || membership.Revision != 1 {
		t.Fatalf("put membership = %#v, %v", membership, testSubjectManagementPostgresEndToEndErr)
	}
	userMemberships, testSubjectManagementPostgresEndToEndErr := service.ListUserMemberships(ctx, subjectmanagement.MembershipListRequest{
		NamespaceID: namespaceID, UserID: userResult.ID, PageSize: 10, Scope: allResults,
	})
	if testSubjectManagementPostgresEndToEndErr != nil || len(userMemberships.Items) != 1 || userMemberships.Items[0].TeamName != "Platform" {
		t.Fatalf("User memberships = %#v, %v", userMemberships, testSubjectManagementPostgresEndToEndErr)
	}
	teamMembers, testSubjectManagementPostgresEndToEndErr := service.ListTeamMembers(ctx, subjectmanagement.MembershipListRequest{
		NamespaceID: namespaceID, TeamID: teamResult.ID, PageSize: 10, Scope: allResults,
	})
	if testSubjectManagementPostgresEndToEndErr != nil || len(teamMembers.Items) != 1 || teamMembers.Items[0].DisplayName != "First User" {
		t.Fatalf("Team members = %#v, %v", teamMembers, testSubjectManagementPostgresEndToEndErr)
	}

	newName := "Platform Engineering"
	updated, testSubjectManagementPostgresEndToEndErr := service.UpdateTeam(ctx, subjectmanagement.UpdateTeamRequest{
		NamespaceID: namespaceID, TeamID: teamResult.ID, ExpectedRevision: 1,
		Name: &newName, Actor: actor,
	})
	if testSubjectManagementPostgresEndToEndErr != nil || updated.Revision != 2 {
		t.Fatalf("update Team = %#v, %v", updated, testSubjectManagementPostgresEndToEndErr)
	}
	if _, err := service.UpdateTeam(ctx, subjectmanagement.UpdateTeamRequest{
		NamespaceID: namespaceID, TeamID: teamResult.ID, ExpectedRevision: 1,
		Name: &newName, Actor: actor,
	}); !errors.Is(err, subjectmanagement.ErrRevisionConflict) {
		t.Fatalf("stale Team CAS error = %v", err)
	}

	var commands, audits, outbox int
	if err := db.QueryRowContext(ctx, `SELECT
  (SELECT count(*) FROM management_idempotency),
  (SELECT count(*) FROM access_audit_events),
  (SELECT count(*) FROM policy_outbox)`).Scan(&commands, &audits, &outbox); err != nil {
		t.Fatal(err)
	}
	if commands != 5 || audits < 6 || outbox != audits {
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
