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
	userID, teamID, teamGroupID, userGroupID, keyID, userBudgetID, digest string
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
		keyID: "key-" + suffix, userBudgetID: "user-budget-" + suffix,
		digest: "digest-" + suffix,
	}
	t.Cleanup(func() {
		_, _ = store.pool.Exec(ctx, `DELETE FROM access_api_keys WHERE id=$1`, fixture.keyID)
		_ = store.DeleteTeam(ctx, fixture.teamID)
		_ = store.DeleteAccessGroup(ctx, fixture.teamGroupID)
		_ = store.DeleteAccessGroup(ctx, fixture.userGroupID)
		_ = store.DeleteBudget(ctx, fixture.userBudgetID)
		_ = store.DeleteUser(ctx, fixture.userID)
	})
	if _, err := store.SaveUser(ctx, User{
		ID: fixture.userID, Email: suffix + "@example.test", Name: "Invited User", Status: StatusActive,
	}); err != nil {
		t.Fatalf("SaveUser() error = %v", err)
	}
	if _, err := store.SaveAccessGroup(ctx, AccessGroup{
		ID: fixture.teamGroupID, Name: "Team " + suffix, ModelPatterns: []string{"team/*"},
	}); err != nil {
		t.Fatalf("SaveAccessGroup(team) error = %v", err)
	}
	if _, err := store.SaveTeam(ctx, Team{
		ID: fixture.teamID, Name: "Team " + suffix, Status: StatusActive,
		UserIDs: []string{fixture.userID}, AccessGroupIDs: []string{fixture.teamGroupID},
		Budget: &KeyBudget{RPM: 12, TPM: 30000, DailyTokens: 250000},
	}); err != nil {
		t.Fatalf("SaveTeam() error = %v", err)
	}
	if _, err := store.CreateSelfAPIKey(ctx, APIKey{
		ID: fixture.keyID, Name: "Invited User", Prefix: "vllm_test", UserID: fixture.userID, Status: StatusActive,
	}, fixture.digest, "ciphertext"); err != nil {
		t.Fatalf("CreateSelfAPIKey() error = %v", err)
	}
	return fixture
}

func TestTeamPolicyInheritanceIntegration(t *testing.T) {
	ctx, store := openIntegrationStore(t)
	fixture := seedTeamPolicyFixture(t, ctx, store)

	team, err := store.GetTeam(ctx, fixture.teamID)
	if err != nil || !reflect.DeepEqual(team.AccessGroupIDs, []string{fixture.teamGroupID}) || team.Budget == nil {
		t.Fatalf("GetTeam() = %#v, %v; want group and budget", team, err)
	}
	selfTeams, err := store.ListTeamsForUser(ctx, fixture.userID)
	if err != nil || len(selfTeams) != 1 || selfTeams[0].Budget == nil {
		t.Fatalf("ListTeamsForUser() = %#v, %v; want inherited Team policy", selfTeams, err)
	}
	principal, err := store.PrincipalByDigest(ctx, fixture.digest)
	if err != nil || !reflect.DeepEqual(principal.ModelPatterns, []string{"team/*"}) {
		t.Fatalf("PrincipalByDigest(team) = %#v, %v; want Team patterns", principal, err)
	}
	if got := budgetIDs(principal.Budgets); !reflect.DeepEqual(got, []string{"team-budget-" + fixture.teamID}) {
		t.Fatalf("team budgets = %#v, want Team budget", got)
	}

	if _, err = store.SaveAccessGroup(ctx, AccessGroup{
		ID: fixture.userGroupID, Name: "User " + fixture.userID, ModelPatterns: []string{"user/*"},
		Bindings: []Binding{{SubjectType: "user", SubjectID: fixture.userID}},
	}); err != nil {
		t.Fatalf("SaveAccessGroup(user) error = %v", err)
	}
	if _, err = store.SaveBudget(ctx, Budget{
		ID: fixture.userBudgetID, Name: "User " + fixture.userID, ScopeType: "user", ScopeID: fixture.userID,
		RPM: 24, TPM: 60000, DailyTokens: 500000, Enabled: true,
	}); err != nil {
		t.Fatalf("SaveBudget(user) error = %v", err)
	}
	principal, err = store.PrincipalByDigest(ctx, fixture.digest)
	if err != nil || !reflect.DeepEqual(principal.ModelPatterns, []string{"user/*"}) {
		t.Fatalf("PrincipalByDigest(user) = %#v, %v; want User override", principal, err)
	}
	if got := budgetIDs(principal.Budgets); !reflect.DeepEqual(got, []string{fixture.userBudgetID}) {
		t.Fatalf("user budgets = %#v, want User override", got)
	}
	assertTeamPolicyInvariants(t, ctx, store, fixture)
}

func assertTeamPolicyInvariants(t *testing.T, ctx context.Context, store *Store, fixture teamPolicyFixture) {
	t.Helper()
	teamGroup, err := store.GetAccessGroup(ctx, fixture.teamGroupID)
	if err != nil {
		t.Fatalf("GetAccessGroup(team) error = %v", err)
	}
	teamGroup.Bindings = nil
	if _, err = store.SaveAccessGroup(ctx, teamGroup); err == nil {
		t.Fatal("SaveAccessGroup() removed a Team's last access group")
	}
	if err = store.DeleteAccessGroup(ctx, fixture.teamGroupID); err == nil {
		t.Fatal("DeleteAccessGroup() deleted a Team's last access group")
	}
	if err = store.DeleteBudget(ctx, "team-budget-"+fixture.teamID); err == nil {
		t.Fatal("DeleteBudget() deleted a required Team budget")
	}
	if !errors.Is(store.SetUserTeam(ctx, fixture.userID, "missing-team"), pgx.ErrNoRows) {
		t.Fatal("SetUserTeam() accepted an unknown Team")
	}
	if _, err = store.SaveTeam(ctx, Team{
		ID: fixture.teamID, Name: "Team " + fixture.teamID, Status: StatusActive,
		AccessGroupIDs: []string{fixture.teamGroupID},
		Budget:         &KeyBudget{RPM: 12, TPM: 30000, DailyTokens: 250000},
	}); err != nil {
		t.Fatalf("SaveTeam(remove members) error = %v", err)
	}
	selfTeams, err := store.ListTeamsForUser(ctx, fixture.userID)
	if err != nil || len(selfTeams) != 0 {
		t.Fatalf("ListTeamsForUser(after removal) = %#v, %v; want no Team", selfTeams, err)
	}
}
