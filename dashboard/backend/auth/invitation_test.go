package auth

import (
	"errors"
	"testing"
)

func TestInvitationLifecycleCreatesDashboardIdentityOnce(t *testing.T) {
	t.Parallel()
	svc := newTestAuthService(t)
	admin := newTestUser(t, svc, "admin@example.com", RoleAdmin, "active")

	invitation, token, err := svc.CreateInvitation(t.Context(), InvitationSpec{
		Kind: InvitationPersonal, Email: " Builder@Example.com ", Name: " Ada Builder ", Role: RoleWrite,
	}, admin.ID)
	if err != nil {
		t.Fatalf("CreateInvitation() error = %v", err)
	}
	if token == "" || invitation.Email != "builder@example.com" || invitation.Name != "Ada Builder" || invitation.Role != RoleWrite {
		t.Fatalf("created invitation = %#v token=%q", invitation, token)
	}
	if info, infoErr := svc.InvitationInfo(t.Context(), token); infoErr != nil || info.ID != invitation.ID {
		t.Fatalf("InvitationInfo() = %#v, %v", info, infoErr)
	}

	_, user, err := svc.AcceptInvitation(t.Context(), token, "", "", "fresh-password")
	if err != nil {
		t.Fatalf("AcceptInvitation() error = %v", err)
	}
	if user.Email != invitation.Email || user.Name != invitation.Name || user.Role != RoleWrite {
		t.Fatalf("accepted user = %#v", user)
	}
	if _, err := svc.InvitationInfo(t.Context(), token); !errors.Is(err, ErrInvitationUnavailable) {
		t.Fatalf("used invitation error = %v, want unavailable", err)
	}
	if _, _, err := svc.AcceptInvitation(t.Context(), token, "", "", "another-password"); !errors.Is(err, ErrInvitationUnavailable) {
		t.Fatalf("second acceptance error = %v, want unavailable", err)
	}
}

func TestInvitationRotationInvalidatesPreviousLink(t *testing.T) {
	t.Parallel()
	svc := newTestAuthService(t)
	admin := newTestUser(t, svc, "admin@example.com", RoleAdmin, "active")

	invitation, originalToken, err := svc.CreateInvitation(t.Context(), InvitationSpec{
		Kind: InvitationPersonal, Email: "reader@example.com", Name: "Reader", Role: RoleRead,
	}, admin.ID)
	if err != nil {
		t.Fatal(err)
	}
	_, replacementToken, err := svc.RotateInvitation(t.Context(), invitation.ID)
	if err != nil {
		t.Fatal(err)
	}
	if originalToken == replacementToken {
		t.Fatal("rotation reused the previous token")
	}
	if _, err := svc.InvitationInfo(t.Context(), originalToken); !errors.Is(err, ErrInvitationUnavailable) {
		t.Fatalf("original token error = %v, want unavailable", err)
	}
	if _, err := svc.InvitationInfo(t.Context(), replacementToken); err != nil {
		t.Fatalf("replacement token error = %v", err)
	}
}

func TestInvitationRejectsExistingDashboardUser(t *testing.T) {
	t.Parallel()
	svc := newTestAuthService(t)
	admin := newTestUser(t, svc, "admin@example.com", RoleAdmin, "active")
	newTestUser(t, svc, "existing@example.com", RoleRead, "active")

	if _, _, err := svc.CreateInvitation(t.Context(), InvitationSpec{
		Kind: InvitationPersonal, Email: "existing@example.com", Name: "Existing", Role: RoleRead,
	}, admin.ID); !errors.Is(err, ErrInvitationUserExists) {
		t.Fatalf("CreateInvitation() error = %v, want existing user", err)
	}
}

func TestSharedInvitationAllocatesDistinctUsersUntilCapacity(t *testing.T) {
	t.Parallel()
	svc := newTestAuthService(t)
	admin := newTestUser(t, svc, "admin@example.com", RoleAdmin, "active")

	invitation, token, err := svc.CreateInvitation(t.Context(), InvitationSpec{
		Kind: InvitationShared, Role: RoleRead, MaxUses: 2,
	}, admin.ID)
	if err != nil {
		t.Fatalf("CreateInvitation() error = %v", err)
	}
	if invitation.Kind != InvitationShared || invitation.MaxUses != 2 || invitation.RemainingUses != 2 {
		t.Fatalf("shared invitation = %#v", invitation)
	}

	_, first, err := svc.AcceptInvitation(t.Context(), token, "first@example.com", "First User", "fresh-password")
	if err != nil {
		t.Fatalf("first AcceptInvitation() error = %v", err)
	}
	if first.Role != RoleRead || first.Email != "first@example.com" {
		t.Fatalf("first user = %#v", first)
	}
	info, err := svc.InvitationInfo(t.Context(), token)
	if err != nil {
		t.Fatalf("InvitationInfo() after first use error = %v", err)
	}
	if info.UsedCount != 1 || info.RemainingUses != 1 || info.Status != InvitationPending {
		t.Fatalf("invitation after first use = %#v", info)
	}

	_, second, err := svc.AcceptInvitation(t.Context(), token, "second@example.com", "Second User", "fresh-password")
	if err != nil {
		t.Fatalf("second AcceptInvitation() error = %v", err)
	}
	if second.Email != "second@example.com" {
		t.Fatalf("second user = %#v", second)
	}
	if _, err := svc.InvitationInfo(t.Context(), token); !errors.Is(err, ErrInvitationUnavailable) {
		t.Fatalf("exhausted invitation error = %v, want unavailable", err)
	}
	if _, _, err := svc.AcceptInvitation(t.Context(), token, "third@example.com", "Third User", "fresh-password"); !errors.Is(err, ErrInvitationUnavailable) {
		t.Fatalf("over-capacity acceptance error = %v, want unavailable", err)
	}
}
