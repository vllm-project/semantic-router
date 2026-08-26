package auth

import (
	"context"
	"database/sql"
	"errors"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
)

const (
	testInvitationNamespace = "20000000-0000-4000-8000-000000000001"
	testInvitationID        = "20000000-0000-4000-8000-000000000002"
	testInvitationUserID    = "20000000-0000-4000-8000-000000000003"
	testInvitationKeyID     = "20000000-0000-4000-8000-000000000004"
	testInvitationPrincipal = "20000000-0000-4000-8000-000000000005"
	testInvitationTeamID    = "20000000-0000-4000-8000-000000000006"
)

type fakeInvitationAuthority struct {
	createdRequest  managementapi.InvitationCreateRequest
	createdActor    AuthContext
	createErr       error
	acceptErr       error
	acceptCalls     int
	acceptedRequest RouterInvitationAcceptance
	revoked         bool
	invitation      managementapi.Invitation
	token           string
	onboardingKey   string
	withoutFirstKey bool
	disableAutomaticFirstKey bool
}

func (authority *fakeInvitationAuthority) ListInvitations(context.Context, AuthContext, string) ([]managementapi.Invitation, error) {
	if authority.invitation.InvitationID == "" {
		return []managementapi.Invitation{}, nil
	}
	return []managementapi.Invitation{authority.invitation}, nil
}

func (authority *fakeInvitationAuthority) CreateInvitation(_ context.Context, actor AuthContext, namespaceID, _ string, request managementapi.InvitationCreateRequest) (managementapi.InvitationIssuedSecret, error) {
	if authority.createErr != nil {
		return managementapi.InvitationIssuedSecret{}, authority.createErr
	}
	authority.createdActor, authority.createdRequest = actor, request
	grants := make([]managementapi.InvitationRoleGrant, len(request.RoleGrants))
	for index, grant := range request.RoleGrants {
		grants[index] = managementapi.InvitationRoleGrant{
			RoleID: grant.RoleID, ScopeKind: grant.ScopeKind,
			RoleRevision: 1, SourceBindingRevision: 1,
		}
	}
	now := time.Now().UTC()
	authority.invitation = managementapi.Invitation{
		InvitationID: testInvitationID, NamespaceID: namespaceID,
		ExpectedIdentity: request.ExpectedIdentity, DisplayName: request.DisplayName,
		Onboarding: managementapi.InvitationOnboardingSnapshot{
			RoleGrants: grants, Team: request.Team,
			AutomaticFirstKey: !authority.disableAutomaticFirstKey,
		},
		ExpiresAt: request.ExpiresAt, Status: InvitationPending, Revision: 1, CreatedAt: now, UpdatedAt: now,
	}
	authority.token = "router-invitation-token"
	return managementapi.InvitationIssuedSecret{
		Data: authority.invitation, Token: authority.token,
		DeliveryExpiresAt: now.Add(time.Hour),
	}, nil
}

func (authority *fakeInvitationAuthority) RotateInvitation(_ context.Context, _ AuthContext, _ string, _ string, revision uint64, _ string, expiresAt *time.Time) (managementapi.InvitationIssuedSecret, error) {
	authority.invitation.Revision = revision + 1
	authority.invitation.UpdatedAt = time.Now().UTC()
	if expiresAt != nil {
		authority.invitation.ExpiresAt = *expiresAt
	}
	authority.token = "rotated-router-invitation-token"
	return managementapi.InvitationIssuedSecret{
		Data: authority.invitation, Token: authority.token,
		DeliveryExpiresAt: time.Now().UTC().Add(time.Hour),
	}, nil
}

func (authority *fakeInvitationAuthority) RevokeInvitation(context.Context, AuthContext, string, string, uint64) (uint64, error) {
	authority.revoked = true
	authority.invitation.Status = InvitationRevoked
	authority.invitation.Revision++
	return authority.invitation.Revision, nil
}

func (authority *fakeInvitationAuthority) AcceptInvitation(_ context.Context, request RouterInvitationAcceptance) (RouterInvitationAcceptanceResult, error) {
	authority.acceptCalls++
	authority.acceptedRequest = request
	if authority.acceptErr != nil {
		return RouterInvitationAcceptanceResult{}, authority.acceptErr
	}
	if authority.onboardingKey == "" && !authority.withoutFirstKey {
		authority.onboardingKey = "test-onboarding-key"
	}
	deliveryExpiresAt := time.Time{}
	onboardingKeyID := ""
	if authority.onboardingKey != "" {
		deliveryExpiresAt = time.Now().UTC().Add(time.Hour)
		onboardingKeyID = testInvitationKeyID
	}
	return RouterInvitationAcceptanceResult{DashboardRole: RoleWrite, Onboarding: managementapi.OnboardingResult{
		InvitationID: testInvitationID, PrincipalID: testInvitationPrincipal, UserID: testInvitationUserID,
		APIKeyID: onboardingKeyID, APIKey: authority.onboardingKey,
		DeliveryExpiresAt: deliveryExpiresAt,
	}}, nil
}

func configuredInvitationService(t *testing.T) (*Service, *fakeInvitationAuthority, *User) {
	t.Helper()
	svc := newTestAuthService(t)
	authority := &fakeInvitationAuthority{}
	svc.ConfigureInvitations(authority, nil, "", "https://dashboard.example.test")
	admin := newTestUser(t, svc, "admin-invite@example.com", RoleAdmin, "active")
	return svc, authority, admin
}

func createRouterInvitation(t *testing.T, svc *Service, admin *User, input invitationInput) *DashboardMemberInvitation {
	t.Helper()
	input.NamespaceID = testInvitationNamespace
	input.IdempotencyKey = "invite-create-00000001"
	input.CreatedBy = admin.ID
	item, err := svc.CreateInvitation(t.Context(), AuthContext{UserID: admin.ID, SessionID: "admin-session"}, input)
	if err != nil {
		t.Fatalf("CreateInvitation() error = %v", err)
	}
	return item
}

func TestDashboardInvitationMapsRoleAndOptionalTeamIntoRouterAuthority(t *testing.T) {
	svc, authority, admin := configuredInvitationService(t)
	created := createRouterInvitation(t, svc, admin, invitationInput{
		Email: "operator@example.com", Name: "Operator", Role: RoleWrite,
		TeamID: testInvitationTeamID, TeamRole: "admin", ExpiresInHours: 24,
	})
	if created.Role != RoleWrite || created.TeamID != testInvitationTeamID || created.TeamRole != "admin" {
		t.Fatalf("created invitation = %#v", created)
	}
	request := authority.createdRequest
	if request.ExpectedIdentity.Subject == "" || request.ExpectedIdentity.Issuer != "https://dashboard.example.test" ||
		len(request.RoleGrants) != 2 || request.RoleGrants[0].RoleID != routerOperatorRoleID ||
		request.RoleGrants[0].ScopeKind != "namespace" || request.Team == nil ||
		request.RoleGrants[1].RoleID != routerConsumerRoleID || request.RoleGrants[1].ScopeKind != "user" ||
		request.Team.TeamID != testInvitationTeamID || request.Team.Role != "admin" {
		t.Fatalf("Router invitation request = %#v", request)
	}
}

func TestInvitationCreationDoesNotRequireAutomaticFirstKey(t *testing.T) {
	svc, authority, admin := configuredInvitationService(t)
	authority.disableAutomaticFirstKey = true

	created := createRouterInvitation(t, svc, admin, invitationInput{
		Email: "manual-key@example.com", Name: "Manual Key", Role: RoleRead,
	})
	if created.ID != testInvitationID || created.Status != InvitationPending || authority.revoked {
		t.Fatalf("created invitation = %#v, revoked = %v", created, authority.revoked)
	}
}

func TestInvitationAcceptanceAllowsPolicyWithoutAutomaticFirstKey(t *testing.T) {
	svc, authority, admin := configuredInvitationService(t)
	authority.disableAutomaticFirstKey = true
	authority.withoutFirstKey = true
	created := createRouterInvitation(t, svc, admin, invitationInput{
		Email: "manual-key-accept@example.com", Name: "Manual Key", Role: RoleRead,
	})

	accepted, err := svc.AcceptInvitation(
		t.Context(), created.InvitationToken, "Manual Key", "a-secure-password",
	)
	if err != nil {
		t.Fatalf("AcceptInvitation() error = %v", err)
	}
	if accepted.User == nil || accepted.User.Email != "manual-key-accept@example.com" ||
		accepted.Onboarding.APIKeyID != "" || accepted.Onboarding.APIKey != "" {
		t.Fatalf("accepted invitation = %#v", accepted)
	}
}

func TestDashboardInvitationRoleMappingIsExact(t *testing.T) {
	for role, roleID := range map[string]string{
		RoleAdmin: routerPlatformAdminRoleID,
		RoleWrite: routerOperatorRoleID,
	} {
		grants, err := invitationRoleGrants(role)
		if err != nil || len(grants) != 2 || grants[0].RoleID != roleID || grants[0].ScopeKind != "namespace" ||
			grants[1].RoleID != routerConsumerRoleID || grants[1].ScopeKind != "user" {
			t.Fatalf("invitationRoleGrants(%q) = %#v, %v", role, grants, err)
		}
	}
	grants, err := invitationRoleGrants(RoleRead)
	if err != nil || len(grants) != 1 || grants[0].RoleID != routerConsumerRoleID ||
		grants[0].ScopeKind != "user" {
		t.Fatalf("invitationRoleGrants(%q) = %#v, %v", RoleRead, grants, err)
	}
	resolved, err := dashboardRoleFromGrants([]managementapi.InvitationRoleGrant{{
		RoleID: routerConsumerRoleID, ScopeKind: "user",
	}})
	if err != nil || resolved != RoleRead {
		t.Fatalf("dashboardRoleFromGrants(consumer) = %q, %v", resolved, err)
	}
}

func TestDashboardReadRoleRequiresConsumerBindingForTheInvitedUser(t *testing.T) {
	now := time.Now().UTC()
	binding := managementapi.ManagementRoleBinding{
		BindingID: "binding", PrincipalID: testInvitationPrincipal, RoleID: routerConsumerRoleID,
		Scope: managementapi.ManagementScope{
			Kind: "user", NamespaceID: testInvitationNamespace, UserID: testInvitationUserID,
		},
		Status: "active", Revision: 1, CreatedAt: now, UpdatedAt: now,
	}
	role, err := DashboardRoleFromManagementBindings(
		[]managementapi.ManagementRoleBinding{binding}, testInvitationNamespace, testInvitationUserID,
	)
	if err != nil || role != RoleRead {
		t.Fatalf("owned consumer role = %q, %v", role, err)
	}
	binding.Scope.UserID = "20000000-0000-4000-8000-000000000099"
	if _, err := DashboardRoleFromManagementBindings(
		[]managementapi.ManagementRoleBinding{binding}, testInvitationNamespace, testInvitationUserID,
	); !errors.Is(err, ErrInvitationAuthorityUnavailable) {
		t.Fatalf("cross-user consumer role error = %v", err)
	}
}

func TestDashboardInvitationStoreContainsPresentationMetadataOnly(t *testing.T) {
	svc, _, admin := configuredInvitationService(t)
	created := createRouterInvitation(t, svc, admin, invitationInput{
		Email: "viewer@example.com", Name: "Viewer", Role: RoleRead,
	})
	columns := map[string]bool{}
	rows, err := svc.store.db.Query(`PRAGMA table_info(dashboard_member_invitations)`)
	if err != nil {
		t.Fatal(err)
	}
	for rows.Next() {
		var cid, notNull, primaryKey int
		var name, columnType string
		var defaultValue any
		if err := rows.Scan(&cid, &name, &columnType, &notNull, &defaultValue, &primaryKey); err != nil {
			t.Fatal(err)
		}
		columns[name] = true
	}
	_ = rows.Close()
	for _, forbidden := range []string{"role", "team_id", "team_role", "api_key", "invitation_token"} {
		if columns[forbidden] {
			t.Fatalf("local invitation schema retained authority column %q", forbidden)
		}
	}
	var digest, subject string
	if err := svc.store.db.QueryRow(`SELECT token_digest,planned_subject_id FROM dashboard_member_invitations WHERE router_invitation_id=?`, created.ID).Scan(&digest, &subject); err != nil {
		t.Fatal(err)
	}
	if digest == "router-invitation-token" || digest == "" || subject == "" {
		t.Fatalf("stored digest=%q subject=%q", digest, subject)
	}
}

func TestRouterOnboardingPrecedesAtomicLocalCommitAndCanReplay(t *testing.T) {
	svc, authority, admin := configuredInvitationService(t)
	created := createRouterInvitation(t, svc, admin, invitationInput{
		Email: "retry@example.com", Name: "Retry User", Role: RoleWrite,
	})
	commitErr := errors.New("injected local commit failure")
	svc.store.invitationBeforeCommit = func() error { return commitErr }
	if _, err := svc.AcceptInvitation(t.Context(), created.InvitationToken, "Retry User", "a-secure-password"); !errors.Is(err, commitErr) {
		t.Fatalf("first AcceptInvitation() error = %v", err)
	}
	if authority.acceptCalls != 1 {
		t.Fatalf("Router accept calls = %d", authority.acceptCalls)
	}
	if _, err := svc.store.GetUserByID(t.Context(), authority.createdRequest.ExpectedIdentity.Subject); !errors.Is(err, sql.ErrNoRows) {
		t.Fatalf("user was committed before local transaction succeeded: %v", err)
	}
	svc.store.invitationBeforeCommit = nil
	accepted, err := svc.AcceptInvitation(t.Context(), created.InvitationToken, "Retry User", "a-secure-password")
	if err != nil {
		t.Fatalf("replayed AcceptInvitation() error = %v", err)
	}
	if authority.acceptCalls != 2 || accepted.Onboarding.APIKey != authority.onboardingKey ||
		accepted.User.ID != authority.createdRequest.ExpectedIdentity.Subject || accepted.User.Role != RoleWrite {
		t.Fatalf("replayed acceptance = %#v calls=%d", accepted, authority.acceptCalls)
	}
	claims, err := svc.ParseToken(accepted.AccessToken)
	if err != nil || claims.ExpiresAt == nil ||
		!claims.ExpiresAt.Time.UTC().Equal(authority.acceptedRequest.SessionExpiresAt) {
		t.Fatalf("accepted session claims=%#v request=%#v error=%v", claims, authority.acceptedRequest, err)
	}
	var plaintextMatches int
	if err := svc.store.db.QueryRow(`SELECT COUNT(*) FROM dashboard_member_invitations
WHERE token_digest IN (?,?) OR email IN (?,?)`, created.InvitationToken, authority.onboardingKey,
		created.InvitationToken, authority.onboardingKey).Scan(&plaintextMatches); err != nil {
		t.Fatal(err)
	}
	if plaintextMatches != 0 {
		t.Fatal("SQLite invitation presentation persisted a plaintext secret")
	}
}

func TestLocalValidationNeverConsumesRouterInvitation(t *testing.T) {
	svc, authority, admin := configuredInvitationService(t)
	created := createRouterInvitation(t, svc, admin, invitationInput{
		Email: "password@example.com", Name: "Password", Role: RoleRead,
	})
	if _, err := svc.AcceptInvitation(t.Context(), created.InvitationToken, "Password", "short123"); err == nil {
		t.Fatal("AcceptInvitation() accepted a weak password")
	}
	if authority.acceptCalls != 0 {
		t.Fatalf("Router invitation was consumed before local validation: %d", authority.acceptCalls)
	}
}

func TestRouterInvitationErrorsDoNotCreateLocalAuthority(t *testing.T) {
	svc, authority, admin := configuredInvitationService(t)
	authority.createErr = &InvitationAuthorityError{Status: 403}
	_, err := svc.CreateInvitation(t.Context(), AuthContext{UserID: admin.ID}, invitationInput{
		Email: "denied@example.com", Name: "Denied", Role: RoleAdmin,
		NamespaceID: testInvitationNamespace, IdempotencyKey: "denied-invite-0001", CreatedBy: admin.ID,
	})
	if err == nil {
		t.Fatal("CreateInvitation() unexpectedly succeeded")
	}
	var count int
	if err := svc.store.db.QueryRow(`SELECT COUNT(*) FROM dashboard_member_invitations`).Scan(&count); err != nil || count != 0 {
		t.Fatalf("local invitation rows=%d error=%v", count, err)
	}
}

func TestReadOnlyDashboardMemberCannotManageInvitations(t *testing.T) {
	svc, _, _ := configuredInvitationService(t)
	request := httptest.NewRequest("GET", "/api/admin/invitations", nil)
	request.Header.Set(managementapi.HeaderNamespaceID, testInvitationNamespace)
	request = request.WithContext(WithAuthContext(request.Context(), AuthContext{
		UserID: "read-only-user", Role: RoleRead, Perms: map[string]bool{PermUsersView: true},
	}))
	response := httptest.NewRecorder()
	adminInvitationsHandler(svc).ServeHTTP(response, request)
	if response.Code != 403 {
		t.Fatalf("read-only invitation list status = %d", response.Code)
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
