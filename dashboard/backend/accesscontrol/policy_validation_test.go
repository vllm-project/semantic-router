package accesscontrol

import "testing"

func TestNormalizeTeamMembershipsRejectsDuplicateUsers(t *testing.T) {
	_, err := normalizeTeamMemberships("team-a", []TeamMembership{
		{UserID: "user-a", Role: TeamRoleMember},
		{UserID: "user-a", Role: TeamRoleAdmin},
	})
	if err == nil {
		t.Fatal("normalizeTeamMemberships() accepted a duplicate user")
	}
}

func TestNormalizeTeamMembershipsNormalizesRoleAndTeam(t *testing.T) {
	memberships, err := normalizeTeamMemberships("team-a", []TeamMembership{
		{UserID: " user-a ", Role: " ADMIN "},
	})
	if err != nil {
		t.Fatalf("normalizeTeamMemberships() error = %v", err)
	}
	if len(memberships) != 1 || memberships[0].TeamID != "team-a" ||
		memberships[0].UserID != "user-a" || memberships[0].Role != TeamRoleAdmin {
		t.Fatalf("normalizeTeamMemberships() = %#v", memberships)
	}
}
