package managementserver

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/subjectmanagement"
)

const (
	usersPath               = managementapi.BasePath + "/users"
	teamsPath               = managementapi.BasePath + "/teams"
	maximumSubjectBodyBytes = 128 << 10
)

type SubjectRoutes struct {
	service       SubjectManagementService
	namespaces    NamespaceResolver
	sessions      SessionAuthenticator
	authorization Authorizer
	scopes        ResultScopeResolver
	now           func() time.Time
	operations    map[string]managementapi.OperationContract
}

func NewSubjectRoutes(options SubjectRoutesOptions) (*SubjectRoutes, error) {
	scopes := configuredResultScopes(options.Scopes, options.Authorization)
	if options.Service == nil || options.Namespaces == nil || options.Sessions == nil || options.Authorization == nil || scopes == nil {
		return nil, errors.New("subject Management routes require service, namespace, session, and authorization dependencies")
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	routes := &SubjectRoutes{
		service: options.Service, namespaces: options.Namespaces,
		sessions: options.Sessions, authorization: options.Authorization, scopes: scopes, now: now,
		operations: make(map[string]managementapi.OperationContract),
	}
	for _, contract := range subjectHTTPContracts() {
		operation, found := managementapi.LookupOperation(contract.method, contract.path)
		if !found {
			return nil, fmt.Errorf("subject Management operation contract %s %s is unavailable", contract.method, contract.path)
		}
		routes.operations[subjectOperationKey(contract.method, contract.path)] = operation
	}
	return routes, nil
}

func (routes *SubjectRoutes) Register(mux *http.ServeMux) {
	if routes == nil || mux == nil {
		panic("Subject Management routes and mux are required")
	}
	for _, pattern := range []string{
		"GET " + usersPath, "POST " + usersPath,
		"GET " + usersPath + "/", "PATCH " + usersPath + "/", "DELETE " + usersPath + "/",
		"GET " + teamsPath, "POST " + teamsPath,
		"GET " + teamsPath + "/", "PATCH " + teamsPath + "/", "DELETE " + teamsPath + "/",
		"PUT " + teamsPath + "/",
	} {
		mux.Handle(pattern, routes)
	}
}

func (routes *SubjectRoutes) Ready(ctx context.Context) error {
	if routes == nil || routes.service == nil {
		return subjectmanagement.ErrUnavailable
	}
	return routes.service.Ready(ctx)
}

func (routes *SubjectRoutes) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	requestID := managementRequestID(request)
	setProviderResponseHeaders(response, requestID)
	if routes == nil || request == nil || request.URL == nil || request.URL.EscapedPath() != request.URL.Path {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	switch {
	case request.URL.Path == usersPath && request.Method == http.MethodGet:
		routes.listUsers(response, request, requestID)
	case request.URL.Path == usersPath && request.Method == http.MethodPost:
		routes.createUser(response, request, requestID)
	case strings.HasPrefix(request.URL.Path, usersPath+"/"):
		routes.userResource(response, request, requestID)
	case request.URL.Path == teamsPath && request.Method == http.MethodGet:
		routes.listTeams(response, request, requestID)
	case request.URL.Path == teamsPath && request.Method == http.MethodPost:
		routes.createTeam(response, request, requestID)
	case strings.HasPrefix(request.URL.Path, teamsPath+"/"):
		routes.teamResource(response, request, requestID)
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
}

func (routes *SubjectRoutes) listUsers(response http.ResponseWriter, request *http.Request, requestID string) {
	query, pageSize, ok := subjectListQuery(response, request, requestID, map[string]bool{
		"": true, "active": true, "disabled": true, "deleted": true,
	})
	if !ok {
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	operation := routes.operation(managementapi.MethodGET, usersPath)
	permission, valid := listPermission(operation)
	if !valid {
		writeProviderError(response, http.StatusServiceUnavailable, "authorization_unavailable", "Authorization state is unavailable.", requestID)
		return
	}
	scope, err := resolveListResultScope(request.Context(), routes.scopes, session, namespaceID, permission)
	if err != nil {
		writeResultScopeError(response, err, requestID)
		return
	}
	page, err := routes.service.ListUsers(request.Context(), subjectmanagement.ListRequest{
		NamespaceID: namespaceID, Status: query.Get("status"), Search: query.Get("search"),
		Cursor: query.Get("cursor"), PageSize: pageSize,
		Scope: scope,
	})
	if err != nil {
		writeSubjectError(response, err, requestID)
		return
	}
	writeProviderJSON(response, http.StatusOK, newUserPage(page), requestID)
}

func (routes *SubjectRoutes) createUser(response http.ResponseWriter, request *http.Request, requestID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "User create does not accept query parameters.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok || !routes.authorize(response, request, requestID, session, namespaceID,
		routes.operation(managementapi.MethodPOST, usersPath), subjectUserNamespaceTargets(namespaceID), nil, false) {
		return
	}
	key, ok := requireIdempotencyKey(response, request, requestID)
	if !ok {
		return
	}
	var body managementapi.UserCreateRequest
	if !decodeSubjectBody(response, request, requestID, &body) {
		return
	}
	result, err := routes.service.CreateUser(request.Context(), subjectmanagement.CreateUserRequest{
		NamespaceID: namespaceID, Email: body.Email, DisplayName: body.DisplayName,
		IdempotencyKey: string(key), Actor: routes.actor(request, session, requestID),
	})
	if err != nil {
		writeSubjectError(response, err, requestID)
		return
	}
	routes.writeMutation(response, result, requestID)
}

func (routes *SubjectRoutes) userResource(response http.ResponseWriter, request *http.Request, requestID string) {
	userID, memberships, ok := userPathValue(request.URL.Path)
	if !ok {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	if memberships {
		if request.Method != http.MethodGet {
			writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
			return
		}
		routes.listUserMemberships(response, request, requestID, userID)
		return
	}
	switch request.Method {
	case http.MethodGet:
		routes.getUser(response, request, requestID, userID)
	case http.MethodPatch:
		routes.patchUser(response, request, requestID, userID)
	case http.MethodDelete:
		routes.deleteUser(response, request, requestID, userID)
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
}

func (routes *SubjectRoutes) getUser(response http.ResponseWriter, request *http.Request, requestID, userID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "User detail does not accept query parameters.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok || !routes.authorize(response, request, requestID, session, namespaceID,
		routes.operation(managementapi.MethodGET, usersPath+"/{userId}"), subjectUserTargets(namespaceID, userID), nil, true) {
		return
	}
	user, err := routes.service.GetUser(request.Context(), namespaceID, userID)
	if err != nil {
		writeSubjectError(response, err, requestID)
		return
	}
	response.Header().Set(managementapi.HeaderETag, subjectETag("user", user.Revision))
	writeProviderJSON(response, http.StatusOK, managementapi.UserDetail{Data: newUserView(user)}, requestID)
}

func (routes *SubjectRoutes) patchUser(response http.ResponseWriter, request *http.Request, requestID, userID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "User update does not accept query parameters.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok || !routes.authorize(response, request, requestID, session, namespaceID,
		routes.operation(managementapi.MethodPATCH, usersPath+"/{userId}"), subjectUserTargets(namespaceID, userID), nil, false) {
		return
	}
	revision, ok := requireSubjectRevision(response, request, requestID, "user")
	if !ok {
		return
	}
	var body managementapi.UserPatchRequest
	if !decodeSubjectBody(response, request, requestID, &body) {
		return
	}
	var status *accesscontrol.UserStatus
	if body.Status != nil {
		value := accesscontrol.UserStatus(*body.Status)
		status = &value
	}
	result, err := routes.service.UpdateUser(request.Context(), subjectmanagement.UpdateUserRequest{
		NamespaceID: namespaceID, UserID: userID, ExpectedRevision: revision,
		Email: body.Email, DisplayName: body.DisplayName, Status: status,
		Actor: routes.actor(request, session, requestID),
	})
	if err != nil {
		writeSubjectError(response, err, requestID)
		return
	}
	routes.writeMutation(response, result, requestID)
}

func (routes *SubjectRoutes) deleteUser(response http.ResponseWriter, request *http.Request, requestID, userID string) {
	if !subjectDeleteRequest(response, request, requestID) {
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok || !routes.authorize(response, request, requestID, session, namespaceID,
		routes.operation(managementapi.MethodDELETE, usersPath+"/{userId}"), subjectUserTargets(namespaceID, userID), nil, false) {
		return
	}
	revision, ok := requireSubjectRevision(response, request, requestID, "user")
	if !ok {
		return
	}
	result, err := routes.service.DeleteUser(request.Context(), subjectmanagement.DeleteUserRequest{
		NamespaceID: namespaceID, UserID: userID, ExpectedRevision: revision,
		Actor: routes.actor(request, session, requestID),
	})
	if err != nil {
		writeSubjectError(response, err, requestID)
		return
	}
	routes.writeMutation(response, result, requestID)
}

func (routes *SubjectRoutes) listUserMemberships(response http.ResponseWriter, request *http.Request, requestID, userID string) {
	query, pageSize, includeTotal, ok := subjectRelationshipListQuery(response, request, requestID, map[string]bool{"": true, "active": true, "disabled": true})
	if !ok {
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	operation := routes.operation(managementapi.MethodGET, usersPath+"/{userId}/memberships")
	if !ok || !routes.authorize(response, request, requestID, session, namespaceID, operation,
		subjectUserTargets(namespaceID, userID), map[string]bool{"user_membership_row": false}, true) {
		return
	}
	teamScope, err := resolveListResultScope(request.Context(), routes.scopes, session, namespaceID, accesscontrol.PermissionTeamRead)
	if err != nil {
		writeResultScopeError(response, err, requestID)
		return
	}
	page, err := routes.service.ListUserMemberships(request.Context(), subjectmanagement.MembershipListRequest{
		NamespaceID: namespaceID, UserID: userID, Status: accesscontrol.MembershipStatus(query.Get("status")),
		Cursor: query.Get("cursor"), PageSize: pageSize, IncludeTotal: includeTotal, Scope: teamScope,
	})
	if err != nil {
		writeSubjectError(response, err, requestID)
		return
	}
	writeProviderJSON(response, http.StatusOK, newUserMembershipPage(page), requestID)
}

func (routes *SubjectRoutes) listTeams(response http.ResponseWriter, request *http.Request, requestID string) {
	query, pageSize, ok := subjectListQuery(response, request, requestID, map[string]bool{"": true, "active": true, "disabled": true})
	if !ok {
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	operation := routes.operation(managementapi.MethodGET, teamsPath)
	permission, valid := listPermission(operation)
	if !valid {
		writeProviderError(response, http.StatusServiceUnavailable, "authorization_unavailable", "Authorization state is unavailable.", requestID)
		return
	}
	scope, err := resolveListResultScope(request.Context(), routes.scopes, session, namespaceID, permission)
	if err != nil {
		writeResultScopeError(response, err, requestID)
		return
	}
	page, err := routes.service.ListTeams(request.Context(), subjectmanagement.ListRequest{
		NamespaceID: namespaceID, Status: query.Get("status"), Search: query.Get("search"),
		Cursor: query.Get("cursor"), PageSize: pageSize,
		Scope: scope,
	})
	if err != nil {
		writeSubjectError(response, err, requestID)
		return
	}
	writeProviderJSON(response, http.StatusOK, newTeamPage(page), requestID)
}

func (routes *SubjectRoutes) createTeam(response http.ResponseWriter, request *http.Request, requestID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Team create does not accept query parameters.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	var body managementapi.TeamCreateRequest
	if !decodeSubjectBody(response, request, requestID, &body) {
		return
	}
	accessPolicyIDs, rateLimitPolicyID, defaults, useDefaultAccess, useDefaultRate, ok := routes.resolveTeamPolicySelection(response, request, requestID, namespaceID, body)
	if !ok {
		return
	}
	targets := subjectTeamNamespaceTargets(namespaceID)
	targets["access_policy"] = make([]accesscontrol.ScopedTarget, 0, len(accessPolicyIDs))
	for _, policyID := range accessPolicyIDs {
		targets["access_policy"] = append(targets["access_policy"], subjectResourceTarget(namespaceID,
			accesscontrol.ScopeResourceAccessPolicy, policyID))
	}
	targets["rate_policy"] = []accesscontrol.ScopedTarget{subjectResourceTarget(namespaceID,
		accesscontrol.ScopeResourceRateLimitPolicy, rateLimitPolicyID)}
	if !routes.authorize(response, request, requestID, session, namespaceID,
		routes.operation(managementapi.MethodPOST, teamsPath), targets, nil, false) {
		return
	}
	key, ok := requireIdempotencyKey(response, request, requestID)
	if !ok {
		return
	}
	result, err := routes.service.CreateTeam(request.Context(), subjectmanagement.CreateTeamRequest{
		NamespaceID: namespaceID, Name: body.Name, Description: body.Description,
		AccessPolicyIDs: accessPolicyIDs, RateLimitPolicyID: rateLimitPolicyID, NamespaceDefaults: defaults,
		UseDefaultAccessPolicy: useDefaultAccess, UseDefaultRateLimitPolicy: useDefaultRate,
		IdempotencyKey: string(key), Actor: routes.actor(request, session, requestID),
	})
	if err != nil {
		writeSubjectError(response, err, requestID)
		return
	}
	routes.writeMutation(response, result, requestID)
}

func (routes *SubjectRoutes) teamResource(response http.ResponseWriter, request *http.Request, requestID string) {
	teamID, userID, kind, ok := teamPathValue(request.URL.Path)
	if !ok {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	switch kind {
	case "team":
		switch request.Method {
		case http.MethodGet:
			routes.getTeam(response, request, requestID, teamID)
		case http.MethodPatch:
			routes.patchTeam(response, request, requestID, teamID)
		case http.MethodDelete:
			routes.deleteTeam(response, request, requestID, teamID)
		default:
			writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		}
	case "members":
		if request.Method == http.MethodGet {
			routes.listTeamMembers(response, request, requestID, teamID)
		} else {
			writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		}
	case "membership":
		routes.membershipMutation(response, request, requestID, teamID, userID)
	}
}

func (routes *SubjectRoutes) getTeam(response http.ResponseWriter, request *http.Request, requestID, teamID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Team detail does not accept query parameters.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok || !routes.authorize(response, request, requestID, session, namespaceID,
		routes.operation(managementapi.MethodGET, teamsPath+"/{teamId}"), subjectTeamTargets(namespaceID, teamID), nil, true) {
		return
	}
	team, err := routes.service.GetTeam(request.Context(), namespaceID, teamID)
	if err != nil {
		writeSubjectError(response, err, requestID)
		return
	}
	response.Header().Set(managementapi.HeaderETag, subjectETag("team", team.Revision))
	writeProviderJSON(response, http.StatusOK, managementapi.TeamDetail{Data: newTeamView(team)}, requestID)
}

func (routes *SubjectRoutes) patchTeam(response http.ResponseWriter, request *http.Request, requestID, teamID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Team update does not accept query parameters.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok || !routes.authorize(response, request, requestID, session, namespaceID,
		routes.operation(managementapi.MethodPATCH, teamsPath+"/{teamId}"), subjectTeamTargets(namespaceID, teamID), nil, false) {
		return
	}
	revision, ok := requireSubjectRevision(response, request, requestID, "team")
	if !ok {
		return
	}
	var body managementapi.TeamPatchRequest
	if !decodeSubjectBody(response, request, requestID, &body) {
		return
	}
	var status *accesscontrol.TeamStatus
	if body.Status != nil {
		value := accesscontrol.TeamStatus(*body.Status)
		status = &value
	}
	result, err := routes.service.UpdateTeam(request.Context(), subjectmanagement.UpdateTeamRequest{
		NamespaceID: namespaceID, TeamID: teamID, ExpectedRevision: revision,
		Name: body.Name, Description: body.Description, Status: status,
		Actor: routes.actor(request, session, requestID),
	})
	if err != nil {
		writeSubjectError(response, err, requestID)
		return
	}
	routes.writeMutation(response, result, requestID)
}

func (routes *SubjectRoutes) deleteTeam(response http.ResponseWriter, request *http.Request, requestID, teamID string) {
	if !subjectDeleteRequest(response, request, requestID) {
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok || !routes.authorize(response, request, requestID, session, namespaceID,
		routes.operation(managementapi.MethodDELETE, teamsPath+"/{teamId}"), subjectTeamTargets(namespaceID, teamID), nil, false) {
		return
	}
	revision, ok := requireSubjectRevision(response, request, requestID, "team")
	if !ok {
		return
	}
	result, err := routes.service.DeleteTeam(request.Context(), subjectmanagement.DeleteTeamRequest{
		NamespaceID: namespaceID, TeamID: teamID, ExpectedRevision: revision,
		Actor: routes.actor(request, session, requestID),
	})
	if err != nil {
		writeSubjectError(response, err, requestID)
		return
	}
	routes.writeMutation(response, result, requestID)
}

func (routes *SubjectRoutes) listTeamMembers(response http.ResponseWriter, request *http.Request, requestID, teamID string) {
	query, pageSize, includeTotal, ok := subjectRelationshipListQuery(response, request, requestID, map[string]bool{"": true, "active": true, "disabled": true})
	if !ok {
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok || !routes.authorize(response, request, requestID, session, namespaceID,
		routes.operation(managementapi.MethodGET, teamsPath+"/{teamId}/members"), subjectTeamTargets(namespaceID, teamID), nil, true) {
		return
	}
	page, err := routes.service.ListTeamMembers(request.Context(), subjectmanagement.MembershipListRequest{
		NamespaceID: namespaceID, TeamID: teamID, Status: accesscontrol.MembershipStatus(query.Get("status")),
		Cursor: query.Get("cursor"), PageSize: pageSize, IncludeTotal: includeTotal,
		Scope: managementauthorization.ResultScope{NamespaceID: accesscontrol.NamespaceID(namespaceID), All: true},
	})
	if err != nil {
		writeSubjectError(response, err, requestID)
		return
	}
	writeProviderJSON(response, http.StatusOK, newTeamMemberPage(page), requestID)
}

func (routes *SubjectRoutes) membershipMutation(response http.ResponseWriter, request *http.Request, requestID, teamID, userID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Membership mutation does not accept query parameters.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	method := managementapi.HTTPMethod(request.Method)
	operation := routes.operation(method, teamsPath+"/{teamId}/members/{userId}")
	if operation.Path == "" || !routes.authorize(response, request, requestID, session, namespaceID,
		operation, subjectTeamTargets(namespaceID, teamID), nil, false) {
		return
	}
	actor := routes.actor(request, session, requestID)
	var result subjectmanagement.MutationResult
	var err error
	switch request.Method {
	case http.MethodPut:
		key, valid := requireIdempotencyKey(response, request, requestID)
		if !valid {
			return
		}
		var body managementapi.MembershipPutRequest
		if !decodeSubjectBody(response, request, requestID, &body) {
			return
		}
		result, err = routes.service.PutMembership(request.Context(), subjectmanagement.PutMembershipRequest{
			NamespaceID: namespaceID, TeamID: teamID, UserID: userID,
			Role: accesscontrol.TeamRole(body.Role), IdempotencyKey: string(key), Actor: actor,
		})
	case http.MethodPatch:
		revision, valid := requireSubjectRevision(response, request, requestID, "membership")
		if !valid {
			return
		}
		var body managementapi.MembershipPatchRequest
		if !decodeSubjectBody(response, request, requestID, &body) {
			return
		}
		var role *accesscontrol.TeamRole
		var status *accesscontrol.MembershipStatus
		if body.Role != nil {
			value := accesscontrol.TeamRole(*body.Role)
			role = &value
		}
		if body.Status != nil {
			value := accesscontrol.MembershipStatus(*body.Status)
			status = &value
		}
		result, err = routes.service.UpdateMembership(request.Context(), subjectmanagement.UpdateMembershipRequest{
			NamespaceID: namespaceID, TeamID: teamID, UserID: userID,
			ExpectedRevision: revision, Role: role, Status: status, Actor: actor,
		})
	case http.MethodDelete:
		if !subjectDeleteRequest(response, request, requestID) {
			return
		}
		revision, valid := requireSubjectRevision(response, request, requestID, "membership")
		if !valid {
			return
		}
		result, err = routes.service.DeleteMembership(request.Context(), subjectmanagement.DeleteMembershipRequest{
			NamespaceID: namespaceID, TeamID: teamID, UserID: userID,
			ExpectedRevision: revision, Actor: actor,
		})
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	if err != nil {
		writeSubjectError(response, err, requestID)
		return
	}
	routes.writeMutation(response, result, requestID)
}

func (routes *SubjectRoutes) authenticate(response http.ResponseWriter, request *http.Request, requestID string) (string, managementauth.AuthenticatedSession, bool) {
	namespaceID, err := routes.namespaces.ResolveNamespace(request.Context(), request)
	if err != nil || !canonicalUUID(namespaceID) {
		writeProviderError(response, http.StatusBadRequest, "invalid_namespace", "A valid namespace is required.", requestID)
		return "", managementauth.AuthenticatedSession{}, false
	}
	token, ok := bearerToken(request)
	if !ok {
		writeProviderError(response, http.StatusUnauthorized, "unauthenticated", "Authentication is required.", requestID)
		return "", managementauth.AuthenticatedSession{}, false
	}
	session, err := routes.sessions.Authenticate(request.Context(), token, namespaceID, routes.now().UTC())
	if err != nil {
		status, code, message := http.StatusServiceUnavailable, "authentication_unavailable", "Authentication state is unavailable."
		if errors.Is(err, managementauth.ErrAuthenticationDenied) {
			status, code, message = http.StatusUnauthorized, "unauthenticated", "Authentication is required."
		}
		writeProviderError(response, status, code, message, requestID)
		return "", managementauth.AuthenticatedSession{}, false
	}
	if session.NamespaceID != namespaceID || !canonicalUUID(session.Session.PrincipalID) {
		writeProviderError(response, http.StatusServiceUnavailable, "authentication_unavailable", "Authentication state is unavailable.", requestID)
		return "", managementauth.AuthenticatedSession{}, false
	}
	return namespaceID, session, true
}

func (routes *SubjectRoutes) authorize(response http.ResponseWriter, request *http.Request, requestID string,
	session managementauth.AuthenticatedSession, namespaceID string, operation managementapi.OperationContract,
	targets map[string][]accesscontrol.ScopedTarget, conditions map[string]bool, nondisclosing bool,
) bool {
	_, err := routes.authorization.Authorize(request.Context(), AuthorizationRequest{
		Operation: operation, Session: session, NamespaceID: namespaceID, Targets: targets, Conditions: conditions,
	})
	if err == nil {
		return true
	}
	if errors.Is(err, managementauthorization.ErrDenied) {
		if nondisclosing {
			writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		} else {
			writeProviderError(response, http.StatusForbidden, "forbidden", "Permission denied.", requestID)
		}
	} else {
		writeProviderError(response, http.StatusServiceUnavailable, "authorization_unavailable", "Authorization state is unavailable.", requestID)
	}
	return false
}

func (routes *SubjectRoutes) actor(request *http.Request, session managementauth.AuthenticatedSession, requestID string) subjectmanagement.Actor {
	return subjectmanagement.Actor{
		PrincipalID: session.Session.PrincipalID,
		ActorChain:  []string{session.Session.PrincipalID}, RequestID: requestID, SourceIP: directRequestIP(request),
	}
}

func (routes *SubjectRoutes) operation(method managementapi.HTTPMethod, path string) managementapi.OperationContract {
	return routes.operations[subjectOperationKey(method, path)]
}

func (routes *SubjectRoutes) writeMutation(response http.ResponseWriter, result subjectmanagement.MutationResult, requestID string) {
	response.Header().Set(managementapi.HeaderETag, subjectETag(result.Kind, result.Revision))
	if result.Idempotent {
		setIdempotencyReplayHeader(response, result.Replayed)
	}
	if result.HTTPStatus == http.StatusNoContent {
		response.WriteHeader(http.StatusNoContent)
		return
	}
	var replayed *bool
	if result.Idempotent {
		value := result.Replayed
		replayed = &value
	}
	writeProviderJSON(response, result.HTTPStatus, managementapi.NewResourceMutationReceipt(
		result.Kind, result.ID, result.Revision, replayed), requestID)
}

func subjectUserTargets(namespaceID, userID string) map[string][]accesscontrol.ScopedTarget {
	return map[string][]accesscontrol.ScopedTarget{"user": {subjectUserTarget(namespaceID, userID)}}
}

func subjectTeamTargets(namespaceID, teamID string) map[string][]accesscontrol.ScopedTarget {
	return map[string][]accesscontrol.ScopedTarget{"team": {subjectTeamTarget(namespaceID, teamID)}}
}

func subjectUserNamespaceTargets(namespaceID string) map[string][]accesscontrol.ScopedTarget {
	return map[string][]accesscontrol.ScopedTarget{"user": {{Scope: accesscontrol.NamespaceScope(accesscontrol.NamespaceID(namespaceID))}}}
}

func subjectTeamNamespaceTargets(namespaceID string) map[string][]accesscontrol.ScopedTarget {
	return map[string][]accesscontrol.ScopedTarget{"team": {{Scope: accesscontrol.NamespaceScope(accesscontrol.NamespaceID(namespaceID))}}}
}

func subjectUserTarget(namespaceID, userID string) accesscontrol.ScopedTarget {
	return accesscontrol.ScopedTarget{Scope: accesscontrol.UserScope(accesscontrol.NamespaceID(namespaceID), accesscontrol.UserID(userID))}
}

func subjectTeamTarget(namespaceID, teamID string) accesscontrol.ScopedTarget {
	return accesscontrol.ScopedTarget{Scope: accesscontrol.TeamScope(accesscontrol.NamespaceID(namespaceID), accesscontrol.TeamID(teamID))}
}

func subjectResourceTarget(namespaceID string, kind accesscontrol.ScopeResourceType, id string) accesscontrol.ScopedTarget {
	return accesscontrol.ScopedTarget{Scope: accesscontrol.ResourceScope(accesscontrol.NamespaceID(namespaceID), kind, accesscontrol.ResourceID(id))}
}
