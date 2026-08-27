package auth

import (
	"errors"
	"testing"
)

func TestInvitationLifecycleCreatesDashboardIdentityOnce(t *testing.T) {
	t.Parallel()
	svc := newTestAuthService(t)
	admin := newTestUser(t, svc, "admin@example.com", RoleAdmin, "active")

	invitation, token, err := svc.CreateInvitation(t.Context(), " Builder@Example.com ", " Ada Builder ", RoleWrite, admin.ID)
	if err != nil {
		t.Fatalf("CreateInvitation() error = %v", err)
	}
	if token == "" || invitation.Email != "builder@example.com" || invitation.Name != "Ada Builder" || invitation.Role != RoleWrite {
		t.Fatalf("created invitation = %#v token=%q", invitation, token)
	}
	if info, infoErr := svc.InvitationInfo(t.Context(), token); infoErr != nil || info.ID != invitation.ID {
		t.Fatalf("InvitationInfo() = %#v, %v", info, infoErr)
	}

	_, user, err := svc.AcceptInvitation(t.Context(), token, "fresh-password")
	if err != nil {
		t.Fatalf("AcceptInvitation() error = %v", err)
	}
	if user.Email != invitation.Email || user.Name != invitation.Name || user.Role != RoleWrite {
		t.Fatalf("accepted user = %#v", user)
	}
	if _, err := svc.InvitationInfo(t.Context(), token); !errors.Is(err, ErrInvitationUnavailable) {
		t.Fatalf("used invitation error = %v, want unavailable", err)
	}
	if _, _, err := svc.AcceptInvitation(t.Context(), token, "another-password"); !errors.Is(err, ErrInvitationUnavailable) {
		t.Fatalf("second acceptance error = %v, want unavailable", err)
	}
}

func TestInvitationRotationInvalidatesPreviousLink(t *testing.T) {
	t.Parallel()
	svc := newTestAuthService(t)
	admin := newTestUser(t, svc, "admin@example.com", RoleAdmin, "active")

	invitation, originalToken, err := svc.CreateInvitation(t.Context(), "reader@example.com", "Reader", RoleRead, admin.ID)
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

	if _, _, err := svc.CreateInvitation(t.Context(), "existing@example.com", "Existing", RoleRead, admin.ID); !errors.Is(err, ErrInvitationUserExists) {
		t.Fatalf("CreateInvitation() error = %v, want existing user", err)
	}
}
