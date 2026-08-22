package accesscontrol

import (
	"context"
	"errors"
	"os"
	"reflect"
	"testing"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
)

type teamPolicyFixture struct {
	userID, teamID, teamGroupID, userGroupID, keyID, teamBudgetID, userBudgetID, digest string
}

func openIntegrationStore(t *testing.T) (context.Context, *Store) {
	t.Helper()
	databaseURL := os.Getenv("VLLM_SR_ACCESS_CONTROL_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("VLLM_SR_ACCESS_CONTROL_TEST_DATABASE_URL is not configured")
	}
	ctx := context.Background()
	store, err := OpenStore(ctx, databaseURL)
	if err != nil {
		t.Fatalf("OpenStore() error = %v", err)
	}
	t.Cleanup(store.Close)
	return ctx, store
}

func seedTeamPolicyFixture(t *testing.T, ctx context.Context, store *Store) teamPolicyFixture {
	t.Helper()
	suffix := uuid.NewString()
	fixture := teamPolicyFixture{
		userID: "user-" + suffix, teamID: "team-" + suffix,
		teamGroupID: "team-group-" + suffix, userGroupID: "user-group-" + suffix,
		keyID: "key-" + suffix, teamBudgetID: "team-budget-" + suffix,
		userBudgetID: "user-budget-" + suffix, digest: "digest-" + suffix,
	}
	t.Cleanup(func() {
		_, _ = store.pool.Exec(ctx, `DELETE FROM access_api_keys WHERE id=$1`, fixture.keyID)
		_ = store.DeleteTeam(ctx, fixture.teamID)
		_ = store.DeleteUser(ctx, fixture.userID)
		_ = store.DeleteAccessGroup(ctx, fixture.teamGroupID)
		_ = store.DeleteAccessGroup(ctx, fixture.userGroupID)
		_ = store.DeleteBudget(ctx, fixture.teamBudgetID)
		_ = store.DeleteBudget(ctx, fixture.userBudgetID)
	})
	for _, budget := range []Budget{
		{ID: fixture.teamBudgetID, Name: "Team quota " + suffix, RPM: 12, TPM: 30000, DailyTokens: 250000, Enabled: true},
		{ID: fixture.userBudgetID, Name: "User quota " + suffix, RPM: 24, TPM: 60000, DailyTokens: 500000, Enabled: true},
	} {
		if _, err := store.SaveBudget(ctx, budget); err != nil {
			t.Fatalf("SaveBudget() error = %v", err)
		}
	}
	if _, err := store.SaveUser(ctx, User{ID: fixture.userID, Email: suffix + "@example.test", Name: "Invited User", Status: StatusActive}); err != nil {
		t.Fatalf("SaveUser() error = %v", err)
	}
	if _, err := store.SaveAccessGroup(ctx, AccessGroup{ID: fixture.teamGroupID, Name: "Team " + suffix, ModelPatterns: []string{"team/*"}}); err != nil {
		t.Fatalf("SaveAccessGroup(team) error = %v", err)
	}
	if _, err := store.SaveTeam(ctx, Team{
		ID: fixture.teamID, Name: "Team " + suffix, Status: StatusActive,
		Members:        []TeamMembership{{UserID: fixture.userID, Role: TeamRoleMember}},
		AccessGroupIDs: []string{fixture.teamGroupID}, BudgetID: fixture.teamBudgetID,
	}); err != nil {
		t.Fatalf("SaveTeam() error = %v", err)
	}
	if _, err := store.CreateSelfAPIKey(ctx, APIKey{
		ID: fixture.keyID, Name: "Invited User", Prefix: "vllm_test", UserID: fixture.userID,
		ContextTeamID: fixture.teamID, Status: StatusActive,
	}, fixture.digest, "ciphertext"); err != nil {
		t.Fatalf("CreateSelfAPIKey() error = %v", err)
	}
	return fixture
}

func TestTeamPolicyInheritanceIntegration(t *testing.T) {
	ctx, store := openIntegrationStore(t)
	fixture := seedTeamPolicyFixture(t, ctx, store)

	team, err := store.GetTeam(ctx, fixture.teamID)
	if err != nil || !reflect.DeepEqual(team.AccessGroupIDs, []string{fixture.teamGroupID}) || team.BudgetID != fixture.teamBudgetID {
		t.Fatalf("GetTeam() = %#v, %v; want reusable Team policy", team, err)
	}
	principal, err := store.PrincipalByDigest(ctx, fixture.digest)
	if err != nil || !reflect.DeepEqual(principal.ModelPatterns, []string{"team/*"}) || len(principal.Budgets) != 1 || principal.Budgets[0].ID != fixture.teamBudgetID {
		t.Fatalf("PrincipalByDigest(team) = %#v, %v", principal, err)
	}
	key, _ := store.GetAPIKey(ctx, fixture.keyID)
	patterns, modelSource, err := store.ModelPolicyForKey(ctx, key)
	if err != nil || !reflect.DeepEqual(patterns, []string{"team/*"}) || modelSource != "team" {
		t.Fatalf("ModelPolicyForKey(team) = %#v, %q, %v", patterns, modelSource, err)
	}
	budgetID, budgetSource, err := store.BudgetPolicyForKey(ctx, key)
	if err != nil || budgetID != fixture.teamBudgetID || budgetSource != "team" {
		t.Fatalf("BudgetPolicyForKey(team) = %q, %q, %v", budgetID, budgetSource, err)
	}
	service := &Service{store: store}
	keys, total, err := service.ListAPIKeys(ctx, ListFilter{KeyID: fixture.keyID, Limit: 10})
	if err != nil || total != 1 || len(keys) != 1 ||
		!reflect.DeepEqual(keys[0].ModelPatterns, []string{"team/*"}) ||
		keys[0].ModelPolicySource != "team" || keys[0].EffectiveBudgetID != fixture.teamBudgetID ||
		keys[0].BudgetPolicySource != "team" {
		t.Fatalf("ListAPIKeys(effective Team policy) = %#v, %d, %v", keys, total, err)
	}

	if _, err = store.SaveAccessGroup(ctx, AccessGroup{ID: fixture.userGroupID, Name: "User " + fixture.userID, ModelPatterns: []string{"user/*"}}); err != nil {
		t.Fatalf("SaveAccessGroup(user) error = %v", err)
	}
	user, _ := store.GetUser(ctx, fixture.userID)
	user.AccessGroupIDs, user.BudgetID = []string{fixture.userGroupID}, fixture.userBudgetID
	if _, err = store.SaveUser(ctx, user); err != nil {
		t.Fatalf("SaveUser(policy) error = %v", err)
	}
	principal, err = store.PrincipalByDigest(ctx, fixture.digest)
	if err != nil || !reflect.DeepEqual(principal.ModelPatterns, []string{"user/*"}) || len(principal.Budgets) != 1 || principal.Budgets[0].ID != fixture.userBudgetID {
		t.Fatalf("PrincipalByDigest(user) = %#v, %v", principal, err)
	}
	assertTeamPolicyInvariants(t, ctx, store, fixture)
}

func TestTeamAdminSelfServiceScopeIntegration(t *testing.T) {
	ctx, store := openIntegrationStore(t)
	fixture := seedTeamPolicyFixture(t, ctx, store)
	if err := store.SetUserTeamMembership(ctx, fixture.userID, fixture.teamID, TeamRoleAdmin); err != nil {
		t.Fatalf("SetUserTeamMembership(admin) error = %v", err)
	}
	allowed, err := store.IsTeamAdmin(ctx, fixture.userID, fixture.teamID)
	if err != nil || !allowed {
		t.Fatalf("IsTeamAdmin() = %v, %v; want true", allowed, err)
	}
	teamKeyID := "team-key-" + uuid.NewString()
	t.Cleanup(func() { _, _ = store.pool.Exec(ctx, `DELETE FROM access_api_keys WHERE id=$1`, teamKeyID) })
	if _, err = store.CreateAPIKey(ctx, APIKey{
		ID: teamKeyID, Name: "Shared Team key", Prefix: "vllm_team", TeamID: fixture.teamID,
		OwnerType: "team", OwnerID: fixture.teamID, ContextTeamID: fixture.teamID, Status: StatusActive,
	}, "digest-"+teamKeyID, "ciphertext"); err != nil {
		t.Fatalf("CreateAPIKey(team) error = %v", err)
	}
	keys, err := store.ListAPIKeysForUser(ctx, fixture.userID)
	if err != nil || len(keys) != 2 {
		t.Fatalf("ListAPIKeysForUser() = %#v, %v; want personal and Team keys", keys, err)
	}
	members, err := store.ListUsersSharingTeam(ctx, fixture.userID)
	if err != nil || len(members) != 1 || members[0].ID != fixture.userID {
		t.Fatalf("ListUsersSharingTeam() = %#v, %v", members, err)
	}
	groups, budgets, err := store.ListPoliciesForUserTeams(ctx, fixture.userID)
	if err != nil || len(groups) != 1 || len(budgets) != 1 {
		t.Fatalf("ListPoliciesForUserTeams() = %#v, %#v, %v", groups, budgets, err)
	}
	overview, err := store.OverviewForUser(ctx, fixture.userID)
	if err != nil || overview.Users != 1 || overview.Teams != 1 || overview.ActiveKeys != 2 ||
		overview.AccessGroups != 1 || overview.EnabledBudgets != 1 {
		t.Fatalf("OverviewForUser() = %#v, %v", overview, err)
	}
}

func TestDeleteAPIKeyPreservesUsageAndUnblocksIdentityDeletion(t *testing.T) {
	ctx, store := openIntegrationStore(t)
	fixture := seedTeamPolicyFixture(t, ctx, store)
	requestID := "request-" + uuid.NewString()
	t.Cleanup(func() {
		_, _ = store.pool.Exec(ctx, `DELETE FROM access_usage_events WHERE request_id=$1`, requestID)
	})
	if err := store.InsertUsage(ctx, UsageEvent{
		ID: uuid.NewString(), RequestID: requestID, KeyID: fixture.keyID,
		UserID: fixture.userID, TeamID: fixture.teamID, Model: "team/model",
		StatusCode: 200, TotalTokens: 42,
	}); err != nil {
		t.Fatalf("InsertUsage() error = %v", err)
	}
	if err := store.DeleteAPIKey(ctx, fixture.keyID); err != nil {
		t.Fatalf("DeleteAPIKey() error = %v", err)
	}
	if _, err := store.GetAPIKey(ctx, fixture.keyID); !errors.Is(err, pgx.ErrNoRows) {
		t.Fatalf("GetAPIKey(deleted) error = %v, want pgx.ErrNoRows", err)
	}
	if total, err := store.CountUsage(ctx, ListFilter{KeyID: fixture.keyID}); err != nil || total != 1 {
		t.Fatalf("CountUsage(deleted key) = %d, %v; want preserved ledger", total, err)
	}
	if err := store.DeleteTeam(ctx, fixture.teamID); err != nil {
		t.Fatalf("DeleteTeam() after key deletion error = %v", err)
	}
	if err := store.DeleteUser(ctx, fixture.userID); err != nil {
		t.Fatalf("DeleteUser() after key deletion error = %v", err)
	}
}

func TestDeletingIdentityRemovesItsPolicyAssignments(t *testing.T) {
	ctx, store := openIntegrationStore(t)
	suffix := uuid.NewString()
	groupID, budgetID := "group-"+suffix, "budget-"+suffix
	userID, teamID := "user-"+suffix, "team-"+suffix
	t.Cleanup(func() {
		_ = store.DeleteTeam(ctx, teamID)
		_ = store.DeleteUser(ctx, userID)
		_ = store.DeleteAccessGroup(ctx, groupID)
		_ = store.DeleteBudget(ctx, budgetID)
	})
	if _, err := store.SaveAccessGroup(ctx, AccessGroup{
		ID: groupID, Name: "Group " + suffix, ModelPatterns: []string{"model/*"},
	}); err != nil {
		t.Fatalf("SaveAccessGroup() error = %v", err)
	}
	if _, err := store.SaveBudget(ctx, Budget{
		ID: budgetID, Name: "Budget " + suffix, RPM: 10, Enabled: true,
	}); err != nil {
		t.Fatalf("SaveBudget() error = %v", err)
	}
	if _, err := store.SaveUser(ctx, User{
		ID: userID, Email: suffix + "@example.test", Name: "User " + suffix,
		Status: StatusActive, AccessGroupIDs: []string{groupID},
	}); err != nil {
		t.Fatalf("SaveUser() error = %v", err)
	}
	if _, err := store.SaveTeam(ctx, Team{
		ID: teamID, Name: "Team " + suffix, Status: StatusActive, BudgetID: budgetID,
		Members:        []TeamMembership{{UserID: userID, Role: TeamRoleAdmin}},
		AccessGroupIDs: []string{groupID},
	}); err != nil {
		t.Fatalf("SaveTeam() error = %v", err)
	}
	if err := store.DeleteTeam(ctx, teamID); err != nil {
		t.Fatalf("DeleteTeam() error = %v", err)
	}
	group, err := store.GetAccessGroup(ctx, groupID)
	if err != nil || group.AssignmentCount != 1 {
		t.Fatalf("GetAccessGroup() after Team delete = %#v, %v", group, err)
	}
	if err = store.DeleteUser(ctx, userID); err != nil {
		t.Fatalf("DeleteUser() error = %v", err)
	}
	group, err = store.GetAccessGroup(ctx, groupID)
	if err != nil || group.AssignmentCount != 0 {
		t.Fatalf("GetAccessGroup() after User delete = %#v, %v", group, err)
	}
}

func TestAssignedBudgetsCannotBeDisabledOrReusedWhileInactive(t *testing.T) {
	ctx, store := openIntegrationStore(t)
	fixture := seedTeamPolicyFixture(t, ctx, store)
	service := &Service{store: store}

	assigned, err := store.GetBudget(ctx, fixture.teamBudgetID)
	if err != nil {
		t.Fatalf("GetBudget(assigned) error = %v", err)
	}
	assigned.Enabled = false
	if _, err = service.SaveBudget(ctx, Actor{}, assigned); err == nil {
		t.Fatal("SaveBudget() disabled an assigned budget")
	}

	inactiveID := "inactive-budget-" + uuid.NewString()
	t.Cleanup(func() { _ = store.DeleteBudget(ctx, inactiveID) })
	if _, err = store.SaveBudget(ctx, Budget{
		ID: inactiveID, Name: "Inactive quota " + inactiveID, RPM: 1, Enabled: false,
	}); err != nil {
		t.Fatalf("SaveBudget(inactive) error = %v", err)
	}
	user, err := store.GetUser(ctx, fixture.userID)
	if err != nil {
		t.Fatalf("GetUser() error = %v", err)
	}
	user.BudgetID = inactiveID
	if _, err = service.SaveUser(ctx, Actor{}, user); err == nil {
		t.Fatal("SaveUser() assigned an inactive budget")
	}
}

func assertTeamPolicyInvariants(t *testing.T, ctx context.Context, store *Store, fixture teamPolicyFixture) {
	t.Helper()
	if err := store.DeleteAccessGroup(ctx, fixture.teamGroupID); err == nil {
		t.Fatal("DeleteAccessGroup() deleted an assigned policy")
	}
	if err := store.DeleteBudget(ctx, fixture.teamBudgetID); err == nil {
		t.Fatal("DeleteBudget() deleted an assigned policy")
	}
	if !errors.Is(store.SetUserTeamMembership(ctx, fixture.userID, "missing-team", TeamRoleMember), pgx.ErrNoRows) {
		t.Fatal("SetUserTeamMembership() accepted an unknown Team")
	}
	team, _ := store.GetTeam(ctx, fixture.teamID)
	team.Members = nil
	if _, err := store.SaveTeam(ctx, team); err != nil {
		t.Fatalf("SaveTeam(remove members) error = %v", err)
	}
	selfTeams, err := store.ListTeamsForUser(ctx, fixture.userID)
	if err != nil || len(selfTeams) != 0 {
		t.Fatalf("ListTeamsForUser(after removal) = %#v, %v; want no Team", selfTeams, err)
	}
}
