package managementserver

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/invitationmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
)

const (
	invitationTestID      = "99999999-9999-4999-8999-999999999999"
	invitationTeamID      = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
	invitationRoleID      = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
	invitationPrincipalID = "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"
)

func TestInvitationCreateAuthorizesTeamAndDeliversCanonicalToken(t *testing.T) {
	now := time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC)
	invitation := testInvitation(now)
	canonical := `{"data":{"invitationId":"` + invitationTestID + `"},"token":"vsi_secret","deliveryExpiresAt":"2026-08-22T12:05:00Z"}`
	service := &invitationServiceStub{createResult: invitationmanagement.SecretResult{
		Invitation: invitation, Token: "vsi_secret", CanonicalJSON: []byte(canonical), Replayed: true,
	}}
	var authorized AuthorizationRequest
	routes := newTestInvitationRoutes(t, service, AuthorizerFunc(func(_ context.Context, request AuthorizationRequest) (AuthorizationDecision, error) {
		authorized = request
		return AuthorizationDecision{}, nil
	}))
	request := authorizedRequest(t, http.MethodPost, invitationsPath, strings.NewReader(`{
  "expectedIdentity":{"issuer":"https://issuer.example","email":"invited@example.com"},
  "displayName":"Invited User",
  "roleGrants":[{"roleId":"`+invitationRoleID+`","scopeKind":"user"}],
  "team":{"teamId":"`+invitationTeamID+`","role":"member"},
  "expiresAt":"2026-08-22T13:00:00Z"
}`))
	request.Header.Set("Content-Type", managementapi.JSONMediaType)
	request.Header.Set(managementapi.HeaderIdempotencyKey, "invitation-create-000001")
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, request)
	if response.Code != http.StatusCreated || response.Body.String() != canonical+"\n" ||
		response.Header().Get(managementapi.HeaderETag) != `"invitation:1"` ||
		response.Header().Get(managementapi.HeaderIdempotencyReplayed) != "true" ||
		response.Header().Get("Cache-Control") != "no-store" {
		t.Fatalf("create status=%d headers=%#v body=%s", response.Code, response.Header(), response.Body.String())
	}
	teamTargets := authorized.Targets["team"]
	if !authorized.Conditions["team_role_requested"] || len(teamTargets) != 1 ||
		teamTargets[0].Scope.Kind != accesscontrol.ScopeKindTeam || string(teamTargets[0].Scope.TeamID) != invitationTeamID {
		t.Fatalf("create authorization = conditions %#v targets %#v", authorized.Conditions, authorized.Targets)
	}
	if service.lastCreate.Team == nil || service.lastCreate.Team.TeamID != invitationTeamID ||
		service.lastCreate.Actor.Reason != "Create invitation." {
		t.Fatalf("create domain request = %#v", service.lastCreate)
	}
}

func TestTeamOnboardingUsesInheritedPoliciesForAuthorizationAndWrite(t *testing.T) {
	snapshot := invitationmanagement.OnboardingSnapshot{
		RoleGrants:                []invitationmanagement.RoleGrant{{RoleID: invitationRoleID, ScopeKind: "user"}},
		Team:                      &invitationmanagement.TeamAssignment{TeamID: invitationTeamID, Role: accesscontrol.TeamRoleMember},
		SelfServicePolicyRevision: 7, AutomaticFirstKey: true,
	}
	canonical := `{"principalId":"` + invitationPrincipalID + `","userId":"` + subjectUserOne + `","deliveryExpiresAt":"2026-08-22T12:05:00Z"}`
	service := &invitationServiceStub{
		snapshot:      snapshot,
		onboardResult: invitationmanagement.PrivilegedOnboardingResult{CanonicalJSON: []byte(canonical), Replayed: true},
	}
	var authorized AuthorizationRequest
	routes := newTestInvitationRoutes(t, service, AuthorizerFunc(func(_ context.Context, request AuthorizationRequest) (AuthorizationDecision, error) {
		authorized = request
		return AuthorizationDecision{}, nil
	}))
	request := authorizedRequest(t, http.MethodPost, onboardingPath, strings.NewReader(`{
  "principalId":"`+invitationPrincipalID+`","email":"invited@example.com","displayName":"Invited User",
  "roleGrants":[{"roleId":"`+invitationRoleID+`","scopeKind":"user"}],
  "team":{"teamId":"`+invitationTeamID+`","role":"member"},"createFirstKey":true
}`))
	request.Header.Set("Content-Type", managementapi.JSONMediaType)
	request.Header.Set(managementapi.HeaderIdempotencyKey, "onboarding-create-000001")
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, request)
	if response.Code != http.StatusCreated || response.Body.String() != canonical+"\n" ||
		response.Header().Get(managementapi.HeaderIdempotencyReplayed) != "true" ||
		response.Header().Get(managementapi.HeaderETag) != "" {
		t.Fatalf("onboarding status=%d headers=%#v body=%s", response.Code, response.Header(), response.Body.String())
	}
	accessTargets, rateTargets := authorized.Targets["access_policy"], authorized.Targets["rate_policy"]
	if len(accessTargets) != 0 || len(rateTargets) != 0 ||
		authorized.Conditions["access_binding_requested"] || authorized.Conditions["rate_binding_requested"] ||
		!authorized.Conditions["first_key_requested"] || !authorized.Conditions["team_membership_requested"] {
		t.Fatalf("onboarding authorization = conditions %#v targets %#v", authorized.Conditions, authorized.Targets)
	}
	if service.lastOnboard.PreparedSnapshot == nil ||
		service.lastOnboard.PreparedSnapshot.Team == nil ||
		service.lastOnboard.PreparedSnapshot.Team.TeamID != invitationTeamID {
		t.Fatalf("onboarding did not reuse prepared snapshot: %#v", service.lastOnboard)
	}
}

func TestInvitationMutationRequiresCASAfterNondisclosingAuthorization(t *testing.T) {
	service := &invitationServiceStub{invitation: testInvitation(time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC))}
	routes := newTestInvitationRoutes(t, service, AuthorizerFunc(func(context.Context, AuthorizationRequest) (AuthorizationDecision, error) {
		return AuthorizationDecision{}, nil
	}))
	request := authorizedRequest(t, http.MethodPost, invitationsPath+"/"+invitationTestID+":rotate-token", strings.NewReader(`{}`))
	request.Header.Set("Content-Type", managementapi.JSONMediaType)
	request.Header.Set(managementapi.HeaderIdempotencyKey, "invitation-rotate-000001")
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, request)
	if response.Code != http.StatusPreconditionRequired || service.rotateCalls != 0 {
		t.Fatalf("rotate without CAS status=%d calls=%d body=%s", response.Code, service.rotateCalls, response.Body.String())
	}
}

type invitationServiceStub struct {
	invitation    invitationmanagement.Invitation
	createResult  invitationmanagement.SecretResult
	onboardResult invitationmanagement.PrivilegedOnboardingResult
	snapshot      invitationmanagement.OnboardingSnapshot
	lastCreate    invitationmanagement.CreateRequest
	lastOnboard   invitationmanagement.PrivilegedOnboardingRequest
	rotateCalls   int
}

func (service *invitationServiceStub) Ready(context.Context) error { return nil }
func (service *invitationServiceStub) Get(context.Context, string, string) (invitationmanagement.Invitation, error) {
	if service.invitation.ID == "" {
		return invitationmanagement.Invitation{}, invitationmanagement.ErrNotFound
	}
	return service.invitation, nil
}

func (service *invitationServiceStub) List(context.Context, invitationmanagement.ListRequest) (invitationmanagement.Page, error) {
	return invitationmanagement.Page{}, nil
}

func (service *invitationServiceStub) Create(_ context.Context, request invitationmanagement.CreateRequest) (invitationmanagement.SecretResult, error) {
	service.lastCreate = request
	return service.createResult, nil
}

func (service *invitationServiceStub) Rotate(context.Context, invitationmanagement.RotateRequest) (invitationmanagement.SecretResult, error) {
	service.rotateCalls++
	return invitationmanagement.SecretResult{}, errors.New("unexpected Rotate")
}

func (service *invitationServiceStub) Revoke(context.Context, invitationmanagement.RevokeRequest) (invitationmanagement.MutationResult, error) {
	return invitationmanagement.MutationResult{}, errors.New("unexpected Revoke")
}

func (service *invitationServiceStub) PrepareOnboarding(context.Context, string, string,
	[]invitationmanagement.RequestedRoleGrant, *invitationmanagement.TeamAssignment,
) (invitationmanagement.OnboardingSnapshot, error) {
	return service.snapshot, nil
}

func (service *invitationServiceStub) Onboard(_ context.Context, request invitationmanagement.PrivilegedOnboardingRequest) (invitationmanagement.PrivilegedOnboardingResult, error) {
	service.lastOnboard = request
	return service.onboardResult, nil
}

func newTestInvitationRoutes(t *testing.T, service InvitationManagementService, authorizer Authorizer) *InvitationRoutes {
	t.Helper()
	routes, err := NewInvitationRoutes(InvitationRoutesOptions{
		Service:    service,
		Namespaces: NamespaceResolverFunc(func(context.Context, *http.Request) (string, error) { return testNamespaceID, nil }),
		Sessions:   sessionStub{}, Authorization: authorizer,
		Now: func() time.Time { return time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC) },
	})
	if err != nil {
		t.Fatal(err)
	}
	return routes
}

func testInvitation(now time.Time) invitationmanagement.Invitation {
	return invitationmanagement.Invitation{
		ID: invitationTestID, NamespaceID: testNamespaceID,
		CreatedByPrincipalID: testPrincipalID, DisplayName: "Invited User",
		Expected: invitationmanagement.ExpectedIdentity{Issuer: "https://issuer.example", Email: "invited@example.com"},
		Snapshot: invitationmanagement.OnboardingSnapshot{
			RoleGrants:                []invitationmanagement.RoleGrant{{RoleID: invitationRoleID, ScopeKind: "user"}},
			Team:                      &invitationmanagement.TeamAssignment{TeamID: invitationTeamID, Role: accesscontrol.TeamRoleMember},
			SelfServicePolicyRevision: 1, AutomaticFirstKey: true,
		},
		ExpiresAt: now.Add(time.Hour), Status: invitationmanagement.StatusPending, Revision: 1,
		CreatedAt: now, UpdatedAt: now,
	}
}

var _ InvitationManagementService = (*invitationServiceStub)(nil)
