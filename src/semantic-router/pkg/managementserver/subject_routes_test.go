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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/subjectmanagement"
)

const (
	subjectUserOne = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
	subjectUserTwo = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
	subjectTeamID  = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
	subjectAccess  = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"
	subjectRate    = "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"
	subjectAccess2 = "ffffffff-ffff-4fff-8fff-ffffffffffff"
)

func TestSubjectRoutesPushAuthorizedScopeBeforePagination(t *testing.T) {
	now := time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC)
	service := &subjectServiceStub{userPage: subjectmanagement.Page[subjectmanagement.User]{
		Items: []subjectmanagement.User{{
			ID: subjectUserOne, NamespaceID: testNamespaceID,
			Email: "one@example.com", DisplayName: "One", Status: accesscontrol.UserStatusActive,
			Revision: 1, CreatedAt: now, UpdatedAt: now,
		}},
		NextCursor: "opaque-next", HasMore: true, PageSize: 2,
	}}
	authorizationCalls := 0
	authorizer := subjectAuthorizerFunc(func(context.Context, AuthorizationRequest) (AuthorizationDecision, error) {
		authorizationCalls++
		return AuthorizationDecision{}, managementauthorization.ErrDenied
	})
	routes := newTestSubjectRoutes(t, service, authorizer)
	routes.scopes = resultScopeResolverFunc(func(_ context.Context, _ accesscontrol.ManagementPrincipalID,
		namespaceID accesscontrol.NamespaceID, permission accesscontrol.Permission,
	) (managementauthorization.ResultScope, error) {
		if permission != accesscontrol.PermissionUserRead {
			t.Fatalf("permission = %q", permission)
		}
		return managementauthorization.ResultScope{
			NamespaceID: namespaceID,
			UserIDs:     []accesscontrol.UserID{subjectUserOne},
		}, nil
	})
	request := authorizedRequest(t, http.MethodGet, usersPath+"?pageSize=2&search=One", nil)
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, request)
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), subjectUserOne) ||
		strings.Contains(response.Body.String(), subjectUserTwo) || !strings.Contains(response.Body.String(), "opaque-next") {
		t.Fatalf("scoped User page status=%d body=%s", response.Code, response.Body.String())
	}
	if authorizationCalls != 0 || service.lastUserList.Scope.All ||
		len(service.lastUserList.Scope.UserIDs) != 1 || service.lastUserList.Scope.UserIDs[0] != subjectUserOne ||
		service.lastUserList.Search != "One" {
		t.Fatalf("list authorization calls=%d request=%#v", authorizationCalls, service.lastUserList)
	}
}

func TestSubjectRelationshipRouteForwardsExactTotalRequest(t *testing.T) {
	totalCount := uint64(42)
	service := &subjectServiceStub{membershipPage: subjectmanagement.Page[subjectmanagement.UserMembership]{
		Items: []subjectmanagement.UserMembership{}, PageSize: 3, TotalCount: &totalCount,
	}}
	routes := newTestSubjectRoutes(t, service, &authorizerStub{})
	request := authorizedRequest(t, http.MethodGet,
		usersPath+"/"+subjectUserOne+"/memberships?pageSize=3&includeTotal=true", nil)
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, request)
	if response.Code != http.StatusOK || !service.lastMembershipList.IncludeTotal ||
		service.lastMembershipList.PageSize != 3 ||
		!strings.Contains(response.Body.String(), `"totalCount":"42"`) {
		t.Fatalf("status=%d request=%#v body=%s", response.Code, service.lastMembershipList, response.Body.String())
	}
}

func TestSubjectRoutesTeamCreateAuthorizesExactDefaultPolicies(t *testing.T) {
	defaults := subjectmanagement.TeamDefaults{
		NamespaceID: testNamespaceID, SelfServiceRevision: 4,
		AccessPolicyID: subjectAccess, AccessPolicyRevision: 2,
		RateLimitPolicyID: subjectRate, RateLimitPolicyRevision: 3,
	}
	service := &subjectServiceStub{defaults: defaults, createTeamResult: subjectmanagement.MutationResult{
		Kind: "team", ID: subjectTeamID, Revision: 1, Idempotent: true, HTTPStatus: http.StatusCreated,
	}}
	var authorized AuthorizationRequest
	authorizer := subjectAuthorizerFunc(func(_ context.Context, request AuthorizationRequest) (AuthorizationDecision, error) {
		authorized = request
		return AuthorizationDecision{}, nil
	})
	routes := newTestSubjectRoutes(t, service, authorizer)
	request := authorizedRequest(t, http.MethodPost, teamsPath, strings.NewReader(`{"name":"Platform"}`))
	request.Header.Set("Content-Type", managementapi.JSONMediaType)
	request.Header.Set(managementapi.HeaderIdempotencyKey, "create-team-0123456789")
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, request)
	if response.Code != http.StatusCreated || response.Header().Get(managementapi.HeaderETag) != `"team:1"` {
		t.Fatalf("Team create status=%d headers=%#v body=%s", response.Code, response.Header(), response.Body.String())
	}
	accessTargets, rateTargets := authorized.Targets["access_policy"], authorized.Targets["rate_policy"]
	if len(accessTargets) != 1 || len(rateTargets) != 1 ||
		string(accessTargets[0].Scope.ResourceID) != subjectAccess ||
		accessTargets[0].Scope.ResourceType != accesscontrol.ScopeResourceAccessPolicy ||
		string(rateTargets[0].Scope.ResourceID) != subjectRate ||
		rateTargets[0].Scope.ResourceType != accesscontrol.ScopeResourceRateLimitPolicy {
		t.Fatalf("Team create authorization targets = %#v", authorized.Targets)
	}
	if service.lastCreateTeam.NamespaceDefaults == nil || *service.lastCreateTeam.NamespaceDefaults != defaults ||
		!service.lastCreateTeam.UseDefaultAccessPolicy || !service.lastCreateTeam.UseDefaultRateLimitPolicy ||
		len(service.lastCreateTeam.AccessPolicyIDs) != 1 ||
		service.lastCreateTeam.AccessPolicyIDs[0] != subjectAccess ||
		service.lastCreateTeam.RateLimitPolicyID != subjectRate ||
		service.lastCreateTeam.IdempotencyKey != "create-team-0123456789" ||
		service.resolveDefaultsCalls != 1 {
		t.Fatalf("Team create request = %#v", service.lastCreateTeam)
	}
}

func TestSubjectRoutesTeamCreateAuthorizesEveryExplicitPolicy(t *testing.T) {
	service := &subjectServiceStub{createTeamResult: subjectmanagement.MutationResult{
		Kind: "team", ID: subjectTeamID, Revision: 1, Idempotent: true, HTTPStatus: http.StatusCreated,
	}}
	var authorized AuthorizationRequest
	routes := newTestSubjectRoutes(t, service, subjectAuthorizerFunc(func(
		_ context.Context,
		request AuthorizationRequest,
	) (AuthorizationDecision, error) {
		authorized = request
		return AuthorizationDecision{}, nil
	}))
	request := authorizedRequest(t, http.MethodPost, teamsPath, strings.NewReader(
		`{"name":"Platform","accessPolicyIds":["`+subjectAccess2+`","`+subjectAccess+`"],"rateLimitPolicyId":"`+subjectRate+`"}`))
	request.Header.Set("Content-Type", managementapi.JSONMediaType)
	request.Header.Set(managementapi.HeaderIdempotencyKey, "create-team-explicit-0123456789")
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, request)
	if response.Code != http.StatusCreated {
		t.Fatalf("Team create status=%d body=%s", response.Code, response.Body.String())
	}
	accessTargets := authorized.Targets["access_policy"]
	rateTargets := authorized.Targets["rate_policy"]
	if len(accessTargets) != 2 || string(accessTargets[0].Scope.ResourceID) != subjectAccess ||
		string(accessTargets[1].Scope.ResourceID) != subjectAccess2 || len(rateTargets) != 1 ||
		string(rateTargets[0].Scope.ResourceID) != subjectRate || service.resolveDefaultsCalls != 0 {
		t.Fatalf("Team create authorization targets=%#v default resolutions=%d",
			authorized.Targets, service.resolveDefaultsCalls)
	}
	if service.lastCreateTeam.NamespaceDefaults != nil || service.lastCreateTeam.UseDefaultAccessPolicy ||
		service.lastCreateTeam.UseDefaultRateLimitPolicy ||
		len(service.lastCreateTeam.AccessPolicyIDs) != 2 ||
		service.lastCreateTeam.AccessPolicyIDs[0] != subjectAccess ||
		service.lastCreateTeam.AccessPolicyIDs[1] != subjectAccess2 ||
		service.lastCreateTeam.RateLimitPolicyID != subjectRate {
		t.Fatalf("Team create request = %#v", service.lastCreateTeam)
	}
}

func TestSubjectRoutesTeamCreateRejectsExplicitEmptyPolicySelection(t *testing.T) {
	for _, test := range []struct {
		name string
		body string
	}{
		{name: "AccessPolicy list", body: `{"name":"Platform","accessPolicyIds":[],"rateLimitPolicyId":"` + subjectRate + `"}`},
		{name: "RateLimitPolicy id", body: `{"name":"Platform","accessPolicyIds":["` + subjectAccess + `"],"rateLimitPolicyId":""}`},
	} {
		t.Run(test.name, func(t *testing.T) {
			service := &subjectServiceStub{}
			authorizationCalls := 0
			routes := newTestSubjectRoutes(t, service, subjectAuthorizerFunc(func(
				context.Context,
				AuthorizationRequest,
			) (AuthorizationDecision, error) {
				authorizationCalls++
				return AuthorizationDecision{}, nil
			}))
			request := authorizedRequest(t, http.MethodPost, teamsPath, strings.NewReader(test.body))
			request.Header.Set("Content-Type", managementapi.JSONMediaType)
			request.Header.Set(managementapi.HeaderIdempotencyKey, "create-team-empty-0123456789")
			response := httptest.NewRecorder()
			routes.ServeHTTP(response, request)
			if response.Code != http.StatusBadRequest || authorizationCalls != 0 {
				t.Fatalf("Team create status=%d authorization calls=%d body=%s",
					response.Code, authorizationCalls, response.Body.String())
			}
		})
	}
}

func TestSubjectRoutesMembershipRequiresCASAndIdempotency(t *testing.T) {
	service := &subjectServiceStub{putMembershipResult: subjectmanagement.MutationResult{
		Kind: "team_membership", ID: subjectUserOne, Revision: 1, Idempotent: true, HTTPStatus: http.StatusOK,
	}, deleteMembershipResult: subjectmanagement.MutationResult{
		Kind: "team_membership", ID: subjectUserOne, Revision: 2, HTTPStatus: http.StatusNoContent,
	}}
	routes := newTestSubjectRoutes(t, service, subjectAuthorizerFunc(func(context.Context, AuthorizationRequest) (AuthorizationDecision, error) {
		return AuthorizationDecision{}, nil
	}))

	put := authorizedRequest(t, http.MethodPut, teamsPath+"/"+subjectTeamID+"/members/"+subjectUserOne, strings.NewReader(`{"role":"admin"}`))
	put.Header.Set("Content-Type", managementapi.JSONMediaType)
	put.Header.Set(managementapi.HeaderIdempotencyKey, "membership-0123456789")
	putResponse := httptest.NewRecorder()
	routes.ServeHTTP(putResponse, put)
	if putResponse.Code != http.StatusOK || putResponse.Header().Get(managementapi.HeaderETag) != `"membership:1"` {
		t.Fatalf("membership PUT status=%d body=%s", putResponse.Code, putResponse.Body.String())
	}

	deleteRequest := authorizedRequest(t, http.MethodDelete, teamsPath+"/"+subjectTeamID+"/members/"+subjectUserOne, nil)
	withoutCAS := httptest.NewRecorder()
	routes.ServeHTTP(withoutCAS, deleteRequest)
	if withoutCAS.Code != http.StatusPreconditionRequired {
		t.Fatalf("membership delete without CAS status=%d body=%s", withoutCAS.Code, withoutCAS.Body.String())
	}
	deleteRequest = authorizedRequest(t, http.MethodDelete, teamsPath+"/"+subjectTeamID+"/members/"+subjectUserOne, nil)
	deleteRequest.Header.Set(managementapi.HeaderIfMatch, `"membership:1"`)
	deleted := httptest.NewRecorder()
	routes.ServeHTTP(deleted, deleteRequest)
	if deleted.Code != http.StatusNoContent || service.lastDeleteMembership.ExpectedRevision != 1 {
		t.Fatalf("membership delete status=%d request=%#v", deleted.Code, service.lastDeleteMembership)
	}
}

type subjectAuthorizerFunc func(context.Context, AuthorizationRequest) (AuthorizationDecision, error)

func (function subjectAuthorizerFunc) Authorize(ctx context.Context, request AuthorizationRequest) (AuthorizationDecision, error) {
	return function(ctx, request)
}

type subjectServiceStub struct {
	userPage               subjectmanagement.Page[subjectmanagement.User]
	defaults               subjectmanagement.TeamDefaults
	createTeamResult       subjectmanagement.MutationResult
	putMembershipResult    subjectmanagement.MutationResult
	deleteMembershipResult subjectmanagement.MutationResult
	lastCreateTeam         subjectmanagement.CreateTeamRequest
	lastDeleteMembership   subjectmanagement.DeleteMembershipRequest
	lastUserList           subjectmanagement.ListRequest
	membershipPage         subjectmanagement.Page[subjectmanagement.UserMembership]
	lastMembershipList     subjectmanagement.MembershipListRequest
	resolveDefaultsCalls   int
}

func (service *subjectServiceStub) Ready(context.Context) error { return nil }
func (service *subjectServiceStub) ResolveTeamDefaults(context.Context, string) (subjectmanagement.TeamDefaults, error) {
	service.resolveDefaultsCalls++
	return service.defaults, nil
}

func (service *subjectServiceStub) GetUser(context.Context, string, string) (subjectmanagement.User, error) {
	return subjectmanagement.User{}, subjectmanagement.ErrNotFound
}

func (service *subjectServiceStub) ListUsers(_ context.Context, request subjectmanagement.ListRequest) (subjectmanagement.Page[subjectmanagement.User], error) {
	service.lastUserList = request
	return service.userPage, nil
}

func (service *subjectServiceStub) CreateUser(context.Context, subjectmanagement.CreateUserRequest) (subjectmanagement.MutationResult, error) {
	return subjectmanagement.MutationResult{}, errors.New("unexpected CreateUser")
}

func (service *subjectServiceStub) UpdateUser(context.Context, subjectmanagement.UpdateUserRequest) (subjectmanagement.MutationResult, error) {
	return subjectmanagement.MutationResult{}, errors.New("unexpected UpdateUser")
}

func (service *subjectServiceStub) DeleteUser(context.Context, subjectmanagement.DeleteUserRequest) (subjectmanagement.MutationResult, error) {
	return subjectmanagement.MutationResult{}, errors.New("unexpected DeleteUser")
}

func (service *subjectServiceStub) GetTeam(context.Context, string, string) (subjectmanagement.Team, error) {
	return subjectmanagement.Team{}, subjectmanagement.ErrNotFound
}

func (service *subjectServiceStub) ListTeams(context.Context, subjectmanagement.ListRequest) (subjectmanagement.Page[subjectmanagement.Team], error) {
	return subjectmanagement.Page[subjectmanagement.Team]{}, nil
}

func (service *subjectServiceStub) CreateTeam(_ context.Context, request subjectmanagement.CreateTeamRequest) (subjectmanagement.MutationResult, error) {
	service.lastCreateTeam = request
	return service.createTeamResult, nil
}

func (service *subjectServiceStub) UpdateTeam(context.Context, subjectmanagement.UpdateTeamRequest) (subjectmanagement.MutationResult, error) {
	return subjectmanagement.MutationResult{}, errors.New("unexpected UpdateTeam")
}

func (service *subjectServiceStub) DeleteTeam(context.Context, subjectmanagement.DeleteTeamRequest) (subjectmanagement.MutationResult, error) {
	return subjectmanagement.MutationResult{}, errors.New("unexpected DeleteTeam")
}

func (service *subjectServiceStub) ListUserMemberships(_ context.Context, request subjectmanagement.MembershipListRequest) (subjectmanagement.Page[subjectmanagement.UserMembership], error) {
	service.lastMembershipList = request
	return service.membershipPage, nil
}

func (service *subjectServiceStub) ListTeamMembers(context.Context, subjectmanagement.MembershipListRequest) (subjectmanagement.Page[subjectmanagement.TeamMember], error) {
	return subjectmanagement.Page[subjectmanagement.TeamMember]{}, nil
}

func (service *subjectServiceStub) PutMembership(context.Context, subjectmanagement.PutMembershipRequest) (subjectmanagement.MutationResult, error) {
	return service.putMembershipResult, nil
}

func (service *subjectServiceStub) UpdateMembership(context.Context, subjectmanagement.UpdateMembershipRequest) (subjectmanagement.MutationResult, error) {
	return subjectmanagement.MutationResult{}, errors.New("unexpected UpdateMembership")
}

func (service *subjectServiceStub) DeleteMembership(_ context.Context, request subjectmanagement.DeleteMembershipRequest) (subjectmanagement.MutationResult, error) {
	service.lastDeleteMembership = request
	return service.deleteMembershipResult, nil
}

func newTestSubjectRoutes(t *testing.T, service SubjectManagementService, authorizer Authorizer) *SubjectRoutes {
	t.Helper()
	routes, err := NewSubjectRoutes(SubjectRoutesOptions{
		Service:    service,
		Namespaces: NamespaceResolverFunc(func(context.Context, *http.Request) (string, error) { return testNamespaceID, nil }),
		Sessions:   sessionStub{}, Authorization: authorizer, Scopes: allowAllResultScopes(),
		Now: func() time.Time { return time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC) },
	})
	if err != nil {
		t.Fatal(err)
	}
	return routes
}

var _ SubjectManagementService = (*subjectServiceStub)(nil)
