package managementserver

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/invitationmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
)

const (
	invitationsPath            = managementapi.BasePath + "/invitations"
	onboardingPath             = managementapi.BasePath + "/onboarding"
	maximumInvitationBodyBytes = 128 << 10
)

var invitationETagPattern = regexp.MustCompile(`^"invitation:([1-9][0-9]*)"$`)

type InvitationRoutes struct {
	service       InvitationManagementService
	namespaces    NamespaceResolver
	sessions      SessionAuthenticator
	authorization Authorizer
	now           func() time.Time
	operations    map[string]managementapi.OperationContract
}

func NewInvitationRoutes(options InvitationRoutesOptions) (*InvitationRoutes, error) {
	if options.Service == nil || options.Namespaces == nil || options.Sessions == nil || options.Authorization == nil {
		return nil, errors.New("invitation routes require service, namespace, session, and authorization dependencies")
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	routes := &InvitationRoutes{
		service: options.Service, namespaces: options.Namespaces, sessions: options.Sessions,
		authorization: options.Authorization, now: now,
		operations: make(map[string]managementapi.OperationContract),
	}
	for _, contract := range invitationHTTPContracts() {
		operation, found := managementapi.LookupOperation(contract.method, contract.path)
		if !found {
			return nil, fmt.Errorf("invitation operation contract %s %s is unavailable", contract.method, contract.path)
		}
		routes.operations[invitationOperationKey(contract.method, contract.path)] = operation
	}
	return routes, nil
}

func (routes *InvitationRoutes) Register(mux *http.ServeMux) {
	if routes == nil || mux == nil {
		panic("Invitation routes and mux are required")
	}
	for _, pattern := range []string{
		"GET " + invitationsPath, "POST " + invitationsPath,
		"GET " + invitationsPath + "/", "DELETE " + invitationsPath + "/",
		"POST " + invitationsPath + "/", "POST " + onboardingPath,
	} {
		mux.Handle(pattern, routes)
	}
}

func (routes *InvitationRoutes) Ready(ctx context.Context) error {
	if routes == nil || routes.service == nil {
		return invitationmanagement.ErrUnavailable
	}
	return routes.service.Ready(ctx)
}

func (routes *InvitationRoutes) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	requestID := managementRequestID(request)
	setProviderResponseHeaders(response, requestID)
	if routes == nil || request == nil || request.URL == nil || request.URL.EscapedPath() != request.URL.Path {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	switch {
	case request.URL.Path == invitationsPath && request.Method == http.MethodGet:
		routes.list(response, request, requestID)
	case request.URL.Path == invitationsPath && request.Method == http.MethodPost:
		routes.create(response, request, requestID)
	case strings.HasPrefix(request.URL.Path, invitationsPath+"/"):
		routes.resource(response, request, requestID)
	case request.URL.Path == onboardingPath && request.Method == http.MethodPost:
		routes.onboard(response, request, requestID)
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
}

func (routes *InvitationRoutes) list(response http.ResponseWriter, request *http.Request, requestID string) {
	query, err := strictProviderQuery(request.URL.RawQuery, map[string]bool{"cursor": true, "pageSize": true, "status": true})
	if err != nil || !map[string]bool{"": true, "pending": true, "accepted": true, "expired": true, "revoked": true}[query.Get("status")] {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Invitation list query is invalid.", requestID)
		return
	}
	pageSize, err := parseOptionalPageSize(query.Get("pageSize"))
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "pageSize must be between 1 and 200.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok || !routes.authorize(response, request, requestID, session, namespaceID,
		routes.operation(managementapi.MethodGET, invitationsPath), invitationNamespaceTargets(namespaceID), nil, false) {
		return
	}
	page, err := routes.service.List(request.Context(), invitationmanagement.ListRequest{
		NamespaceID: namespaceID, Status: invitationmanagement.Status(query.Get("status")),
		Cursor: query.Get("cursor"), PageSize: pageSize,
	})
	if err != nil {
		writeInvitationError(response, err, requestID)
		return
	}
	writeProviderJSON(response, http.StatusOK, invitationPageDTO(page), requestID)
}

func (routes *InvitationRoutes) create(response http.ResponseWriter, request *http.Request, requestID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Invitation create does not accept query parameters.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	var body managementapi.InvitationCreateRequest
	if !decodeInvitationBody(response, request, requestID, &body) {
		return
	}
	team := invitationTeam(body.Team)
	targets := invitationNamespaceTargets(namespaceID)
	if team != nil {
		targets["team"] = []accesscontrol.ScopedTarget{subjectTeamTarget(namespaceID, team.TeamID)}
	}
	conditions := map[string]bool{"team_role_requested": team != nil}
	if !routes.authorize(response, request, requestID, session, namespaceID,
		routes.operation(managementapi.MethodPOST, invitationsPath), targets, conditions, false) {
		return
	}
	key, ok := requireIdempotencyKey(response, request, requestID)
	if !ok {
		return
	}
	result, err := routes.service.Create(request.Context(), invitationmanagement.CreateRequest{
		NamespaceID: namespaceID, Expected: invitationExpected(body.ExpectedIdentity),
		DisplayName: body.DisplayName, RoleGrants: invitationGrants(body.RoleGrants), Team: team,
		ExpiresAt: body.ExpiresAt, IdempotencyKey: string(key),
		Actor: routes.actor(request, session, requestID, "Create invitation."),
	})
	if err != nil {
		writeInvitationError(response, err, requestID)
		return
	}
	writeInvitationSecret(response, http.StatusCreated, result, requestID)
}

func (routes *InvitationRoutes) resource(response http.ResponseWriter, request *http.Request, requestID string) {
	invitationID, rotate, ok := invitationPathValue(request.URL.Path)
	if !ok || (rotate && request.Method != http.MethodPost) || (!rotate && request.Method != http.MethodGet && request.Method != http.MethodDelete) {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	if rotate {
		routes.rotate(response, request, requestID, invitationID)
		return
	}
	if request.Method == http.MethodGet {
		routes.get(response, request, requestID, invitationID)
		return
	}
	routes.revoke(response, request, requestID, invitationID)
}

func (routes *InvitationRoutes) get(response http.ResponseWriter, request *http.Request, requestID, invitationID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Invitation detail does not accept query parameters.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	invitation, err := routes.service.Get(request.Context(), namespaceID, invitationID)
	if err != nil {
		writeInvitationError(response, err, requestID)
		return
	}
	if !routes.authorize(response, request, requestID, session, namespaceID,
		routes.operation(managementapi.MethodGET, invitationsPath+"/{invitationId}"),
		invitationResourceTargets(namespaceID, invitation), nil, true) {
		return
	}
	response.Header().Set(managementapi.HeaderETag, invitationETag(invitation.Revision))
	writeProviderJSON(response, http.StatusOK, managementapi.InvitationDetail{Data: invitationDTO(invitation)}, requestID)
}

func (routes *InvitationRoutes) rotate(response http.ResponseWriter, request *http.Request, requestID, invitationID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Token rotation does not accept query parameters.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	var body managementapi.InvitationRotateTokenRequest
	if !decodeInvitationBody(response, request, requestID, &body) {
		return
	}
	invitation, err := routes.service.Get(request.Context(), namespaceID, invitationID)
	if err != nil {
		writeInvitationError(response, err, requestID)
		return
	}
	targets, conditions := invitationMutationTargets(namespaceID, invitation)
	if !routes.authorize(response, request, requestID, session, namespaceID,
		routes.operation(managementapi.MethodPOST, invitationsPath+"/{invitationId}:rotate-token"),
		targets, conditions, true) {
		return
	}
	revision, ok := requireInvitationRevision(response, request, requestID)
	if !ok {
		return
	}
	key, ok := requireIdempotencyKey(response, request, requestID)
	if !ok {
		return
	}
	result, err := routes.service.Rotate(request.Context(), invitationmanagement.RotateRequest{
		NamespaceID: namespaceID, InvitationID: invitationID, ExpectedRevision: revision,
		ExpiresAt: body.ExpiresAt, IdempotencyKey: string(key),
		Actor: routes.actor(request, session, requestID, "Rotate invitation token."),
	})
	if err != nil {
		writeInvitationError(response, err, requestID)
		return
	}
	writeInvitationSecret(response, http.StatusOK, result, requestID)
}

func (routes *InvitationRoutes) revoke(response http.ResponseWriter, request *http.Request, requestID, invitationID string) {
	if request.URL.RawQuery != "" || !noRequestBody(response, request, requestID) {
		if request.URL.RawQuery != "" {
			writeProviderError(response, http.StatusBadRequest, "invalid_request", "Invitation revoke accepts no query or request body.", requestID)
		}
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	invitation, err := routes.service.Get(request.Context(), namespaceID, invitationID)
	if err != nil {
		writeInvitationError(response, err, requestID)
		return
	}
	targets, conditions := invitationMutationTargets(namespaceID, invitation)
	if !routes.authorize(response, request, requestID, session, namespaceID,
		routes.operation(managementapi.MethodDELETE, invitationsPath+"/{invitationId}"), targets, conditions, true) {
		return
	}
	revision, ok := requireInvitationRevision(response, request, requestID)
	if !ok {
		return
	}
	result, err := routes.service.Revoke(request.Context(), invitationmanagement.RevokeRequest{
		NamespaceID: namespaceID, InvitationID: invitationID, ExpectedRevision: revision,
		Actor: routes.actor(request, session, requestID, "Revoke invitation."),
	})
	if err != nil {
		writeInvitationError(response, err, requestID)
		return
	}
	response.Header().Set(managementapi.HeaderETag, invitationETag(result.Invitation.Revision))
	response.WriteHeader(http.StatusNoContent)
}

func (routes *InvitationRoutes) onboard(response http.ResponseWriter, request *http.Request, requestID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Onboarding does not accept query parameters.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	var body managementapi.OnboardingCreateRequest
	if !decodeInvitationBody(response, request, requestID, &body) {
		return
	}
	team := invitationTeam(body.Team)
	grants := invitationGrants(body.RoleGrants)
	snapshot, err := routes.service.PrepareOnboarding(request.Context(), namespaceID,
		session.Session.PrincipalID, grants, team)
	if err != nil {
		writeInvitationError(response, err, requestID)
		return
	}
	targets := onboardingTargets(namespaceID, snapshot)
	conditions := map[string]bool{
		"role_binding_requested":    len(grants) != 0,
		"team_membership_requested": team != nil,
		"first_key_requested":       body.CreateFirstKey,
		"access_binding_requested":  team == nil,
		"rate_binding_requested":    team == nil,
	}
	if !routes.authorize(response, request, requestID, session, namespaceID,
		routes.operation(managementapi.MethodPOST, onboardingPath), targets, conditions, false) {
		return
	}
	key, ok := requireIdempotencyKey(response, request, requestID)
	if !ok {
		return
	}
	result, err := routes.service.Onboard(request.Context(), invitationmanagement.PrivilegedOnboardingRequest{
		NamespaceID: namespaceID, PrincipalID: body.PrincipalID, Email: body.Email,
		DisplayName: body.DisplayName, RoleGrants: grants, Team: team,
		CreateFirstKey: body.CreateFirstKey, IdempotencyKey: string(key),
		Actor:            routes.actor(request, session, requestID, "Create onboarding identity."),
		PreparedSnapshot: &snapshot,
	})
	if err != nil {
		writeInvitationError(response, err, requestID)
		return
	}
	writeOnboardingSecret(response, result, requestID)
}

func (routes *InvitationRoutes) authenticate(response http.ResponseWriter, request *http.Request,
	requestID string,
) (string, managementauth.AuthenticatedSession, bool) {
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

func (routes *InvitationRoutes) authorize(response http.ResponseWriter, request *http.Request, requestID string,
	session managementauth.AuthenticatedSession, namespaceID string, operation managementapi.OperationContract,
	targets map[string][]accesscontrol.ScopedTarget, conditions map[string]bool, nondisclosing bool,
) bool {
	_, err := routes.authorization.Authorize(request.Context(), AuthorizationRequest{
		Operation: operation, Session: session, NamespaceID: namespaceID,
		Targets: targets, Conditions: conditions,
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

func (routes *InvitationRoutes) actor(request *http.Request, session managementauth.AuthenticatedSession,
	requestID, reason string,
) invitationmanagement.Actor {
	return invitationmanagement.Actor{
		PrincipalID: session.Session.PrincipalID,
		ActorChain:  []string{session.Session.PrincipalID}, RequestID: requestID,
		SourceIP: directRequestIP(request), Reason: reason,
	}
}

func (routes *InvitationRoutes) operation(method managementapi.HTTPMethod, path string) managementapi.OperationContract {
	return routes.operations[invitationOperationKey(method, path)]
}

func decodeInvitationBody(response http.ResponseWriter, request *http.Request, requestID string, target any) bool {
	if request.ContentLength > maximumInvitationBodyBytes {
		writeProviderError(response, http.StatusRequestEntityTooLarge, "invalid_request", "Request body is too large.", requestID)
		return false
	}
	mediaType, parameters, err := mime.ParseMediaType(request.Header.Get("Content-Type"))
	if err != nil || mediaType != managementapi.JSONMediaType ||
		(len(parameters) != 0 && (len(parameters) != 1 || !strings.EqualFold(parameters["charset"], "utf-8"))) {
		writeProviderError(response, http.StatusUnsupportedMediaType, "unsupported_media_type", "Use the Management API media type.", requestID)
		return false
	}
	request.Body = http.MaxBytesReader(response, request.Body, maximumInvitationBodyBytes)
	decoder := json.NewDecoder(request.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		var maximum *http.MaxBytesError
		if errors.As(err, &maximum) {
			writeProviderError(response, http.StatusRequestEntityTooLarge, "invalid_request", "Request body is too large.", requestID)
		} else {
			writeProviderError(response, http.StatusBadRequest, "invalid_request", "Request body is invalid.", requestID)
		}
		return false
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Request body is invalid.", requestID)
		return false
	}
	return true
}

func requireInvitationRevision(response http.ResponseWriter, request *http.Request, requestID string) (uint64, bool) {
	values := request.Header.Values(managementapi.HeaderIfMatch)
	if len(values) != 1 {
		writeProviderError(response, http.StatusPreconditionRequired, "precondition_required", "If-Match is required.", requestID)
		return 0, false
	}
	match := invitationETagPattern.FindStringSubmatch(values[0])
	if len(match) != 2 {
		writeProviderError(response, http.StatusBadRequest, "invalid_precondition", "If-Match is invalid.", requestID)
		return 0, false
	}
	revision, err := strconv.ParseUint(match[1], 10, 64)
	if err != nil || revision == 0 {
		writeProviderError(response, http.StatusBadRequest, "invalid_precondition", "If-Match is invalid.", requestID)
		return 0, false
	}
	return revision, true
}

func invitationETag(revision uint64) string {
	return `"invitation:` + strconv.FormatUint(revision, 10) + `"`
}

func invitationPathValue(path string) (string, bool, bool) {
	value := strings.TrimPrefix(path, invitationsPath+"/")
	rotate := strings.HasSuffix(value, ":rotate-token")
	if rotate {
		value = strings.TrimSuffix(value, ":rotate-token")
	}
	if !canonicalUUID(value) {
		return "", false, false
	}
	return value, rotate, true
}

func invitationExpected(value managementapi.InvitationExpectedIdentity) invitationmanagement.ExpectedIdentity {
	return invitationmanagement.ExpectedIdentity{Issuer: value.Issuer, Subject: value.Subject, Email: value.Email}
}

func invitationTeam(value *managementapi.InvitationTeamAssignment) *invitationmanagement.TeamAssignment {
	if value == nil {
		return nil
	}
	return &invitationmanagement.TeamAssignment{TeamID: value.TeamID, Role: accesscontrol.TeamRole(value.Role)}
}

func invitationGrants(values []managementapi.InvitationRoleGrantRequest) []invitationmanagement.RequestedRoleGrant {
	result := make([]invitationmanagement.RequestedRoleGrant, len(values))
	for index, value := range values {
		result[index] = invitationmanagement.RequestedRoleGrant{
			RoleID:    value.RoleID,
			ScopeKind: value.ScopeKind, DelegationCeiling: append([]string(nil), value.DelegationCeiling...),
		}
	}
	return result
}

func invitationNamespaceTargets(namespaceID string) map[string][]accesscontrol.ScopedTarget {
	return map[string][]accesscontrol.ScopedTarget{"target": {{
		Scope: accesscontrol.NamespaceScope(accesscontrol.NamespaceID(namespaceID)),
	}}}
}

func invitationResourceTargets(namespaceID string, invitation invitationmanagement.Invitation) map[string][]accesscontrol.ScopedTarget {
	return map[string][]accesscontrol.ScopedTarget{"target": {{Scope: accesscontrol.ResourceScope(
		accesscontrol.NamespaceID(namespaceID), accesscontrol.ScopeResourceInvitation,
		accesscontrol.ResourceID(invitation.ID),
	)}}}
}

func invitationMutationTargets(namespaceID string, invitation invitationmanagement.Invitation) (map[string][]accesscontrol.ScopedTarget, map[string]bool) {
	targets := invitationResourceTargets(namespaceID, invitation)
	teamRequested := invitation.Snapshot.Team != nil
	if teamRequested {
		targets["team"] = []accesscontrol.ScopedTarget{subjectTeamTarget(namespaceID, invitation.Snapshot.Team.TeamID)}
	}
	return targets, map[string]bool{"team_role_requested": teamRequested}
}

func onboardingTargets(namespaceID string, snapshot invitationmanagement.OnboardingSnapshot) map[string][]accesscontrol.ScopedTarget {
	userNamespace := accesscontrol.ScopedTarget{Scope: accesscontrol.NamespaceScope(accesscontrol.NamespaceID(namespaceID))}
	targets := map[string][]accesscontrol.ScopedTarget{
		"target": {userNamespace}, "user": {userNamespace}, "owner": {userNamespace},
	}
	if snapshot.Team != nil {
		targets["team"] = []accesscontrol.ScopedTarget{subjectTeamTarget(namespaceID, snapshot.Team.TeamID)}
	} else {
		targets["access_policy"] = []accesscontrol.ScopedTarget{{Scope: accesscontrol.ResourceScope(
			accesscontrol.NamespaceID(namespaceID), accesscontrol.ScopeResourceAccessPolicy,
			accesscontrol.ResourceID(snapshot.AccessPolicyID))}}
		targets["rate_policy"] = []accesscontrol.ScopedTarget{{Scope: accesscontrol.ResourceScope(
			accesscontrol.NamespaceID(namespaceID), accesscontrol.ScopeResourceRateLimitPolicy,
			accesscontrol.ResourceID(snapshot.RateLimitPolicyID))}}
	}
	return targets
}

func writeInvitationSecret(response http.ResponseWriter, status int, result invitationmanagement.SecretResult, requestID string) {
	setProviderResponseHeaders(response, requestID)
	response.Header().Set("Content-Type", managementapi.JSONMediaType)
	response.Header().Set(managementapi.HeaderETag, invitationETag(result.Invitation.Revision))
	if status == http.StatusCreated {
		response.Header().Set("Location", invitationsPath+"/"+result.Invitation.ID)
	}
	setIdempotencyReplayHeader(response, result.Replayed)
	response.WriteHeader(status)
	_, _ = response.Write(append(append([]byte(nil), result.CanonicalJSON...), '\n'))
}

func writeOnboardingSecret(response http.ResponseWriter, result invitationmanagement.PrivilegedOnboardingResult, requestID string) {
	setProviderResponseHeaders(response, requestID)
	response.Header().Set("Content-Type", managementapi.JSONMediaType)
	if result.Result.UserID != "" {
		response.Header().Set("Location", usersPath+"/"+result.Result.UserID)
	}
	setIdempotencyReplayHeader(response, result.Replayed)
	response.WriteHeader(http.StatusCreated)
	_, _ = response.Write(append(append([]byte(nil), result.CanonicalJSON...), '\n'))
}

func writeInvitationError(response http.ResponseWriter, err error, requestID string) {
	switch {
	case errors.Is(err, invitationmanagement.ErrInvalidRequest), errors.Is(err, invitationmanagement.ErrInvalidToken):
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Invitation request is invalid.", requestID)
	case errors.Is(err, invitationmanagement.ErrNotFound):
		writeProviderError(response, http.StatusNotFound, "not_found", "Invitation not found.", requestID)
	case errors.Is(err, invitationmanagement.ErrRevisionConflict):
		writeProviderError(response, http.StatusPreconditionFailed, "revision_conflict", "The invitation changed. Refresh and retry.", requestID)
	case errors.Is(err, invitationmanagement.ErrExpired):
		writeProviderError(response, http.StatusGone, "invitation_expired", "The invitation expired.", requestID)
	case errors.Is(err, invitationmanagement.ErrSecretExpired):
		writeProviderError(response, http.StatusGone, "secret_result_expired", "The one-time secret delivery window expired.", requestID)
	case errors.Is(err, invitationmanagement.ErrDelegationDenied), errors.Is(err, invitationmanagement.ErrIdentityMismatch):
		writeProviderError(response, http.StatusForbidden, "forbidden", "Permission denied.", requestID)
	case errors.Is(err, invitationmanagement.ErrDefaultsChanged):
		writeProviderError(response, http.StatusConflict, "onboarding_defaults_changed", "Onboarding defaults changed. Refresh and retry.", requestID)
	case errors.Is(err, invitationmanagement.ErrAlreadyAccepted):
		writeProviderError(response, http.StatusConflict, "identity_already_onboarded", "This identity is already onboarded.", requestID)
	case errors.Is(err, invitationmanagement.ErrConflict):
		writeProviderError(response, http.StatusConflict, "conflict", "Invitation state conflicts with this request.", requestID)
	default:
		writeProviderError(response, http.StatusServiceUnavailable, "invitation_service_unavailable", "Invitation service is unavailable.", requestID)
	}
}

type invitationHTTPContract struct {
	method managementapi.HTTPMethod
	path   string
}

func invitationHTTPContracts() []invitationHTTPContract {
	return []invitationHTTPContract{
		{managementapi.MethodGET, invitationsPath},
		{managementapi.MethodPOST, invitationsPath},
		{managementapi.MethodGET, invitationsPath + "/{invitationId}"},
		{managementapi.MethodDELETE, invitationsPath + "/{invitationId}"},
		{managementapi.MethodPOST, invitationsPath + "/{invitationId}:rotate-token"},
		{managementapi.MethodPOST, onboardingPath},
	}
}

func invitationOperationKey(method managementapi.HTTPMethod, path string) string {
	return string(method) + " " + path
}
