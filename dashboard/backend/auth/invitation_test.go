package auth

import (
	"context"
	"errors"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
)

type modelUserRecorder struct {
	id            string
	email         string
	name          string
	teamID        string
	teamRole      string
	missingTeamID string
}

func (r *modelUserRecorder) EnsureModelUser(_ context.Context, id, email, name string) error {
	r.id, r.email, r.name = id, email, name
	return nil
}

func (r *modelUserRecorder) AssignModelUserTeam(_ context.Context, id, teamID, role string) error {
	r.id, r.teamID, r.teamRole = id, teamID, role
	return nil
}

func (r *modelUserRecorder) ModelTeamName(_ context.Context, teamID string) (string, error) {
	if teamID == r.missingTeamID {
		return "", nil
	}
	return teamID, nil
}

func (r *modelUserRecorder) RemoveModelUser(_ context.Context, _ string) error {
	return nil
}

func TestDashboardMemberInvitationAcceptsExactlyOnce(t *testing.T) {
	svc := newTestAuthService(t)
	provisioner := &modelUserRecorder{}
	svc.ConfigureModelUsers(provisioner)
	admin := newTestUser(t, svc, "admin-invite@example.com", RoleAdmin, "active")

	invitation, err := svc.CreateInvitation(t.Context(), invitationInput{
		Email: "operator@example.com", Name: "Operator", Role: RoleWrite,
		TeamID: "team-42", TeamRole: "admin", CreatedBy: admin.ID, ExpiresInHours: 24,
	})
	if err != nil {
		t.Fatalf("CreateInvitation() error = %v", err)
	}
	if invitation.InvitationToken == "" || invitation.InvitationPath == "" {
		t.Fatalf("invitation did not return its one-time onboarding secret: %#v", invitation)
	}
	if invitation.InvitationPath != "/login?invite=1&token="+url.QueryEscape(invitation.InvitationToken) {
		t.Fatalf("invitation path = %q, want login onboarding route", invitation.InvitationPath)
	}

	_, user, err := svc.AcceptInvitation(t.Context(), invitation.InvitationToken, "Accepted Operator", "a-secure-password")
	if err != nil {
		t.Fatalf("AcceptInvitation() error = %v", err)
	}
	if user.Role != RoleWrite {
		t.Fatalf("accepted user = %#v", user)
	}
	if provisioner.id != user.ID || provisioner.email != user.Email || provisioner.teamID != "team-42" || provisioner.teamRole != "admin" {
		t.Fatalf("model user provisioning = %#v", provisioner)
	}
	if _, _, secondErr := svc.AcceptInvitation(t.Context(), invitation.InvitationToken, "Again", "another-secure-password"); !errors.Is(secondErr, ErrInvitationUnavailable) {
		t.Fatalf("second acceptance error = %v, want ErrInvitationUnavailable", secondErr)
	}
}

func TestDashboardMemberInvitationResendRotatesSecret(t *testing.T) {
	svc := newTestAuthService(t)
	admin := newTestUser(t, svc, "admin-resend@example.com", RoleAdmin, "active")
	created, err := svc.CreateInvitation(context.Background(), invitationInput{
		Email: "viewer@example.com", Role: RoleRead, CreatedBy: admin.ID,
	})
	if err != nil {
		t.Fatal(err)
	}
	resent, err := svc.ResendInvitation(context.Background(), created.ID, false)
	if err != nil {
		t.Fatal(err)
	}
	if resent.InvitationToken == created.InvitationToken {
		t.Fatal("resend reused the previous invitation token")
	}
	if _, err := svc.InvitationInfo(context.Background(), created.InvitationToken); !errors.Is(err, ErrInvitationUnavailable) {
		t.Fatalf("old token error = %v, want unavailable", err)
	}
	if _, err := svc.InvitationInfo(context.Background(), resent.InvitationToken); err != nil {
		t.Fatalf("rotated token should be valid: %v", err)
	}
}

func TestDashboardMemberInvitationRejectsUnknownTeam(t *testing.T) {
	svc := newTestAuthService(t)
	svc.ConfigureModelUsers(&modelUserRecorder{})
	admin := newTestUser(t, svc, "admin-team@example.com", RoleAdmin, "active")
	if _, err := svc.CreateInvitation(t.Context(), invitationInput{
		Email: "member@example.com", Name: "Member", Role: RoleRead,
		CreatedBy: admin.ID,
	}); err != nil {
		t.Fatalf("CreateInvitation() without Team error = %v", err)
	}

	svc.ConfigureModelUsers(&modelUserRecorder{missingTeamID: "missing"})
	if _, err := svc.CreateInvitation(t.Context(), invitationInput{
		Email: "other@example.com", Name: "Other", Role: RoleRead,
		TeamID: "missing", TeamRole: "member", CreatedBy: admin.ID,
	}); err == nil {
		t.Fatal("CreateInvitation() accepted an unknown Team")
	}
}

func TestDashboardMemberInvitationRejectsWeakPassword(t *testing.T) {
	svc := newTestAuthService(t)
	admin := newTestUser(t, svc, "admin-password@example.com", RoleAdmin, "active")
	invitation, err := svc.CreateInvitation(context.Background(), invitationInput{
		Email: "new-member@example.com", Role: RoleRead, CreatedBy: admin.ID,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := svc.AcceptInvitation(context.Background(), invitation.InvitationToken, "New Member", "short123"); err == nil {
		t.Fatal("AcceptInvitation() accepted a password shorter than 9 characters")
	}
	if _, err := svc.InvitationInfo(context.Background(), invitation.InvitationToken); err != nil {
		t.Fatalf("failed acceptance consumed the one-time invitation: %v", err)
	}
}

func TestInvitationRequestDecoderRejectsAmbiguousBodies(t *testing.T) {
	t.Parallel()
	for _, body := range []string{
		`{"email":"member@example.com","unexpected":true}`,
		`{"email":"member@example.com"}{"email":"other@example.com"}`,
	} {
		request := httptest.NewRequest("POST", "/api/admin/invitations", strings.NewReader(body))
		response := httptest.NewRecorder()
		var value struct {
			Email string `json:"email"`
		}
		if err := decodeInvitationRequest(response, request, &value); err == nil {
			t.Fatalf("decodeInvitationRequest(%q) unexpectedly succeeded", body)
		}
	}
}
