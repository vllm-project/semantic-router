package managementserver

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/delegationmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
)

const (
	selfInferenceKeysPath     = managementapi.BasePath + "/self/inference-keys"
	selfInferenceSessionsPath = managementapi.BasePath + "/self/inference-sessions"
)

type DelegationRoutes struct {
	service       DelegationManagementService
	namespaces    NamespaceResolver
	sessions      SessionAuthenticator
	authorization Authorizer
	now           func() time.Time
	operations    map[string]managementapi.OperationContract
}

func NewDelegationRoutes(options DelegationRoutesOptions) (*DelegationRoutes, error) {
	if options.Service == nil || options.Namespaces == nil || options.Sessions == nil || options.Authorization == nil {
		return nil, errors.New("delegation Management routes require service, namespace, session, and authorization dependencies")
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	routes := &DelegationRoutes{
		service: options.Service, namespaces: options.Namespaces,
		sessions: options.Sessions, authorization: options.Authorization, now: now,
		operations: make(map[string]managementapi.OperationContract),
	}
	for _, item := range delegationHTTPContracts() {
		operation, found := managementapi.LookupOperation(item.method, item.path)
		if !found {
			return nil, fmt.Errorf("delegation operation contract %s %s is unavailable", item.method, item.path)
		}
		routes.operations[string(item.method)+" "+item.path] = operation
	}
	return routes, nil
}

func (routes *DelegationRoutes) Register(mux *http.ServeMux) {
	if routes == nil || mux == nil {
		panic("delegation Management routes and mux are required")
	}
	mux.Handle("GET "+selfInferenceKeysPath, routes)
	mux.Handle("GET "+selfInferenceSessionsPath, routes)
	mux.Handle("POST "+selfInferenceSessionsPath, routes)
	mux.Handle("DELETE "+selfInferenceSessionsPath+"/{sessionId}", routes)
	mux.Handle("GET "+apiKeysPath+"/{keyId}/inference-sessions", routes)
	mux.Handle("DELETE "+apiKeysPath+"/{keyId}/inference-sessions/{sessionId}", routes)
	mux.Handle("POST "+apiKeysPath+"/{keyId}/inference-sessions:revoke-all", routes)
}

func (routes *DelegationRoutes) Ready(ctx context.Context) error { return routes.service.Ready(ctx) }

func (routes *DelegationRoutes) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	requestID := managementRequestID(request)
	setProviderResponseHeaders(response, requestID)
	if routes == nil || request == nil || request.URL == nil || request.URL.EscapedPath() != request.URL.Path {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	switch {
	case request.URL.Path == selfInferenceKeysPath && request.Method == http.MethodGet:
		routes.listEligible(response, request, requestID)
	case request.URL.Path == selfInferenceSessionsPath && request.Method == http.MethodGet:
		routes.listSelfSessions(response, request, requestID)
	case request.URL.Path == selfInferenceSessionsPath && request.Method == http.MethodPost:
		routes.create(response, request, requestID)
	case request.Method == http.MethodDelete && request.PathValue("keyId") == "":
		routes.revokeSelf(response, request, requestID, request.PathValue("sessionId"))
	case request.Method == http.MethodGet:
		routes.listKeySessions(response, request, requestID, request.PathValue("keyId"))
	case request.Method == http.MethodDelete:
		routes.revokeKeySession(response, request, requestID, request.PathValue("keyId"), request.PathValue("sessionId"))
	case request.Method == http.MethodPost:
		routes.revokeAll(response, request, requestID, request.PathValue("keyId"))
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
}

func (routes *DelegationRoutes) listEligible(response http.ResponseWriter, request *http.Request, requestID string) {
	pageSize, cursor, ok := delegationPageRequest(response, request, requestID)
	if !ok {
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	self, err := routes.service.ResolveSelf(request.Context(), namespaceID, session.Session.PrincipalID, session.Session.ID)
	if err != nil {
		writeDelegationError(response, err, requestID)
		return
	}
	operation := routes.operation(managementapi.MethodGET, selfInferenceKeysPath)
	if !routes.authorize(response, request, requestID, session, namespaceID, operation,
		map[string][]accesscontrol.ScopedTarget{"user": {{Scope: accesscontrol.UserScope(accesscontrol.NamespaceID(namespaceID), accesscontrol.UserID(self.UserID))}}}, nil, true) {
		return
	}
	page, err := routes.service.ListEligibleKeys(request.Context(), delegationmanagement.ListRequest{
		NamespaceID: namespaceID, PrincipalID: session.Session.PrincipalID,
		ManagementSessionID: session.Session.ID, Cursor: cursor, PageSize: pageSize,
	})
	if err != nil {
		writeDelegationError(response, err, requestID)
		return
	}
	data := make([]managementapi.EligibleInferenceKey, len(page.Items))
	for index, key := range page.Items {
		data[index] = managementapi.EligibleInferenceKey{
			KeyID: key.KeyID, Name: key.Name,
			Owner:         managementapi.APIKeyOwner{Type: string(key.OwnerKind), ID: key.OwnerID},
			ContextTeamID: key.ContextTeamID, ExpiresAt: key.ExpiresAt,
		}
	}
	writeProviderJSON(response, http.StatusOK, managementapi.EligibleInferenceKeyPage{
		Data: data,
		Page: managementapi.PageInfo{NextCursor: page.NextCursor, HasMore: page.HasMore, PageSize: page.PageSize},
	}, requestID)
}

func (routes *DelegationRoutes) listSelfSessions(response http.ResponseWriter, request *http.Request, requestID string) {
	pageSize, cursor, ok := delegationPageRequest(response, request, requestID)
	if !ok {
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	self, err := routes.service.ResolveSelf(request.Context(), namespaceID, session.Session.PrincipalID, session.Session.ID)
	if err != nil {
		writeDelegationError(response, err, requestID)
		return
	}
	if !routes.authorize(response, request, requestID, session, namespaceID,
		routes.operation(managementapi.MethodGET, selfInferenceSessionsPath),
		map[string][]accesscontrol.ScopedTarget{"user": {{Scope: accesscontrol.UserScope(accesscontrol.NamespaceID(namespaceID), accesscontrol.UserID(self.UserID))}}}, nil, true) {
		return
	}
	page, err := routes.service.ListSessions(request.Context(), delegationmanagement.ListRequest{
		NamespaceID: namespaceID, PrincipalID: session.Session.PrincipalID, Cursor: cursor, PageSize: pageSize,
	})
	if err != nil {
		writeDelegationError(response, err, requestID)
		return
	}
	routes.writeSessionPage(response, page, requestID)
}

func (routes *DelegationRoutes) create(response http.ResponseWriter, request *http.Request, requestID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, 400, "invalid_request", "Delegation create does not accept query parameters.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	idempotencyKey, ok := requireIdempotencyKey(response, request, requestID)
	if !ok {
		return
	}
	var body managementapi.DelegatedInferenceSessionCreateRequest
	if !decodeAPIKeyBody(response, request, requestID, &body) {
		return
	}
	if !canonicalUUID(body.KeyID) {
		writeProviderError(response, 400, "invalid_request", "keyId is invalid.", requestID)
		return
	}
	key, err := routes.service.GetKey(request.Context(), namespaceID, body.KeyID)
	if err != nil {
		writeDelegationError(response, err, requestID)
		return
	}
	if !routes.authorize(response, request, requestID, session, namespaceID,
		routes.operation(managementapi.MethodPOST, selfInferenceSessionsPath), apiKeyTargets(key), nil, true) {
		return
	}
	result, err := routes.service.Create(request.Context(), delegationmanagement.CreateRequest{
		NamespaceID: namespaceID, KeyID: body.KeyID, IdempotencyKey: string(idempotencyKey), Actor: routes.actor(request, session, requestID),
	})
	if err != nil {
		writeDelegationError(response, err, requestID)
		return
	}
	response.Header().Set("Location", selfInferenceSessionsPath+"/"+result.Session.ID)
	setIdempotencyReplayHeader(response, result.Replayed)
	response.Header().Set("Content-Type", managementapi.JSONMediaType)
	response.WriteHeader(http.StatusCreated)
	_, _ = response.Write(append(append([]byte(nil), result.CanonicalJSON...), '\n'))
}

func (routes *DelegationRoutes) revokeSelf(response http.ResponseWriter, request *http.Request, requestID, sessionID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, 400, "invalid_request", "Delegation revoke does not accept query parameters.", requestID)
		return
	}
	if !noRequestBody(response, request, requestID) {
		return
	}
	if !canonicalUUID(sessionID) {
		writeProviderError(response, http.StatusNotFound, "not_found", "Delegated inference session not found.", requestID)
		return
	}
	namespaceID, authenticated, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	delegated, err := routes.service.GetSession(request.Context(), namespaceID, sessionID)
	if err != nil || delegated.PrincipalID != authenticated.Session.PrincipalID {
		writeProviderError(response, http.StatusNotFound, "not_found", "Delegated inference session not found.", requestID)
		return
	}
	key, err := routes.service.GetKey(request.Context(), namespaceID, delegated.APIKeyID)
	if err != nil {
		writeDelegationError(response, err, requestID)
		return
	}
	if !routes.authorize(response, request, requestID, authenticated, namespaceID,
		routes.operation(managementapi.MethodDELETE, selfInferenceSessionsPath+"/{sessionId}"), apiKeyTargets(key), nil, true) {
		return
	}
	_, err = routes.service.Revoke(request.Context(), delegationmanagement.RevokeRequest{
		NamespaceID: namespaceID,
		SessionID:   sessionID, PrincipalID: authenticated.Session.PrincipalID, Actor: routes.actor(request, authenticated, requestID),
	})
	if err != nil {
		writeDelegationError(response, err, requestID)
		return
	}
	response.WriteHeader(http.StatusNoContent)
}

func (routes *DelegationRoutes) listKeySessions(response http.ResponseWriter, request *http.Request, requestID, keyID string) {
	if !canonicalUUID(keyID) {
		writeProviderError(response, http.StatusNotFound, "not_found", "API key not found.", requestID)
		return
	}
	pageSize, cursor, ok := delegationPageRequest(response, request, requestID)
	if !ok {
		return
	}
	namespaceID, session, key, ok := routes.authorizedKey(response, request, requestID, keyID,
		routes.operation(managementapi.MethodGET, apiKeysPath+"/{keyId}/inference-sessions"))
	if !ok {
		return
	}
	_ = key
	page, err := routes.service.ListSessions(request.Context(), delegationmanagement.ListRequest{
		NamespaceID: namespaceID, APIKeyID: keyID, Cursor: cursor, PageSize: pageSize,
	})
	if err != nil {
		writeDelegationError(response, err, requestID)
		return
	}
	_ = session
	routes.writeSessionPage(response, page, requestID)
}

func (routes *DelegationRoutes) revokeKeySession(response http.ResponseWriter, request *http.Request, requestID, keyID, sessionID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, 400, "invalid_request", "Delegation revoke does not accept query parameters.", requestID)
		return
	}
	if !noRequestBody(response, request, requestID) {
		return
	}
	if !canonicalUUID(keyID) || !canonicalUUID(sessionID) {
		writeProviderError(response, http.StatusNotFound, "not_found", "Delegated inference session not found.", requestID)
		return
	}
	namespaceID, session, _, ok := routes.authorizedKey(response, request, requestID, keyID,
		routes.operation(managementapi.MethodDELETE, apiKeysPath+"/{keyId}/inference-sessions/{sessionId}"))
	if !ok {
		return
	}
	_, err := routes.service.Revoke(request.Context(), delegationmanagement.RevokeRequest{
		NamespaceID: namespaceID,
		SessionID:   sessionID, APIKeyID: keyID, Actor: routes.actor(request, session, requestID),
	})
	if err != nil {
		writeDelegationError(response, err, requestID)
		return
	}
	response.WriteHeader(http.StatusNoContent)
}

func (routes *DelegationRoutes) revokeAll(response http.ResponseWriter, request *http.Request, requestID, keyID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, 400, "invalid_request", "Delegation revoke-all does not accept query parameters.", requestID)
		return
	}
	if !noRequestBody(response, request, requestID) {
		return
	}
	if !canonicalUUID(keyID) {
		writeProviderError(response, http.StatusNotFound, "not_found", "API key not found.", requestID)
		return
	}
	namespaceID, session, _, ok := routes.authorizedKey(response, request, requestID, keyID,
		routes.operation(managementapi.MethodPOST, apiKeysPath+"/{keyId}/inference-sessions:revoke-all"))
	if !ok {
		return
	}
	idempotencyKey, ok := requireIdempotencyKey(response, request, requestID)
	if !ok {
		return
	}
	result, err := routes.service.RevokeAll(request.Context(), delegationmanagement.RevokeAllRequest{
		NamespaceID: namespaceID, KeyID: keyID, IdempotencyKey: string(idempotencyKey), Actor: routes.actor(request, session, requestID),
	})
	if err != nil {
		writeDelegationError(response, err, requestID)
		return
	}
	setIdempotencyReplayHeader(response, result.Replayed)
	response.WriteHeader(http.StatusNoContent)
}

func (routes *DelegationRoutes) authorizedKey(response http.ResponseWriter, request *http.Request, requestID, keyID string,
	operation managementapi.OperationContract,
) (string, managementauth.AuthenticatedSession, accesscontrol.APIKey, bool) {
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return "", managementauth.AuthenticatedSession{}, accesscontrol.APIKey{}, false
	}
	key, err := routes.service.GetKey(request.Context(), namespaceID, keyID)
	if err != nil {
		writeDelegationError(response, err, requestID)
		return "", managementauth.AuthenticatedSession{}, accesscontrol.APIKey{}, false
	}
	if !routes.authorize(response, request, requestID, session, namespaceID, operation, apiKeyTargets(key), nil, true) {
		return "", managementauth.AuthenticatedSession{}, accesscontrol.APIKey{}, false
	}
	return namespaceID, session, key, true
}

func (routes *DelegationRoutes) authenticate(response http.ResponseWriter, request *http.Request, requestID string) (string, managementauth.AuthenticatedSession, bool) {
	namespaceID, err := routes.namespaces.ResolveNamespace(request.Context(), request)
	if err != nil || !canonicalUUID(namespaceID) {
		writeProviderError(response, 400, "invalid_namespace", "A valid namespace is required.", requestID)
		return "", managementauth.AuthenticatedSession{}, false
	}
	token, ok := bearerToken(request)
	if !ok {
		writeProviderError(response, 401, "unauthenticated", "Authentication is required.", requestID)
		return "", managementauth.AuthenticatedSession{}, false
	}
	session, err := routes.sessions.Authenticate(request.Context(), token, namespaceID, routes.now().UTC())
	if err != nil {
		if errors.Is(err, managementauth.ErrAuthenticationDenied) {
			writeProviderError(response, 401, "unauthenticated", "Authentication is required.", requestID)
		} else {
			writeProviderError(response, 503, "authentication_unavailable", "Authentication state is unavailable.", requestID)
		}
		return "", managementauth.AuthenticatedSession{}, false
	}
	return namespaceID, session, true
}

func (routes *DelegationRoutes) authorize(response http.ResponseWriter, request *http.Request, requestID string,
	session managementauth.AuthenticatedSession, namespaceID string, operation managementapi.OperationContract,
	targets map[string][]accesscontrol.ScopedTarget, conditions map[string]bool, nondisclosing bool,
) bool {
	_, err := routes.authorization.Authorize(request.Context(), AuthorizationRequest{
		Operation: operation,
		Session:   session, NamespaceID: namespaceID, Targets: targets, Conditions: conditions,
	})
	if err == nil {
		return true
	}
	if errors.Is(err, managementauthorization.ErrDenied) {
		if nondisclosing {
			writeProviderError(response, 404, "not_found", "Resource not found.", requestID)
		} else {
			writeProviderError(response, 403, "forbidden", "Permission denied.", requestID)
		}
	} else {
		writeProviderError(response, 503, "authorization_unavailable", "Authorization state is unavailable.", requestID)
	}
	return false
}

func (routes *DelegationRoutes) actor(request *http.Request, session managementauth.AuthenticatedSession, requestID string) delegationmanagement.Actor {
	return delegationmanagement.Actor{
		PrincipalID: session.Session.PrincipalID, ManagementSessionID: session.Session.ID,
		ActorChain: []string{session.Session.PrincipalID}, RequestID: requestID, SourceIP: directRequestIP(request),
	}
}

func (routes *DelegationRoutes) operation(method managementapi.HTTPMethod, path string) managementapi.OperationContract {
	return routes.operations[string(method)+" "+path]
}

func (routes *DelegationRoutes) writeSessionPage(response http.ResponseWriter,
	page delegationmanagement.ResultPage[delegationmanagement.Session], requestID string,
) {
	data := make([]managementapi.DelegatedInferenceSession, len(page.Items))
	for index, session := range page.Items {
		data[index] = delegatedSessionDTO(session)
	}
	writeProviderJSON(response, 200, managementapi.DelegatedInferenceSessionPage{
		Data: data,
		Page: managementapi.PageInfo{NextCursor: page.NextCursor, HasMore: page.HasMore, PageSize: page.PageSize},
	}, requestID)
}

func delegatedSessionDTO(value delegationmanagement.Session) managementapi.DelegatedInferenceSession {
	return managementapi.DelegatedInferenceSession{
		SessionID: value.ID, PublicID: value.PublicID,
		KeyID: value.APIKeyID, UserID: value.UserID, TeamID: value.TeamID, Audience: value.Audience,
		Status: string(value.Status), NotBefore: value.NotBefore, ExpiresAt: value.ExpiresAt, CreatedAt: value.CreatedAt,
	}
}

func delegationPageRequest(response http.ResponseWriter, request *http.Request, requestID string) (int, string, bool) {
	query, err := strictProviderQuery(request.URL.RawQuery, map[string]bool{"cursor": true, "pageSize": true})
	if err != nil {
		writeProviderError(response, 400, "invalid_request", "Delegation query is invalid.", requestID)
		return 0, "", false
	}
	pageSize, err := parseOptionalPageSize(query.Get("pageSize"))
	if err != nil {
		writeProviderError(response, 400, "invalid_request", "pageSize must be between 1 and 200.", requestID)
		return 0, "", false
	}
	return pageSize, query.Get("cursor"), true
}

func writeDelegationError(response http.ResponseWriter, err error, requestID string) {
	switch {
	case errors.Is(err, delegationmanagement.ErrInvalidRequest):
		writeProviderError(response, 400, "invalid_request", "Delegation request is invalid.", requestID)
	case errors.Is(err, delegationmanagement.ErrNotFound), errors.Is(err, delegationmanagement.ErrNotEligible):
		writeProviderError(response, 404, "not_found", "Delegation resource not found.", requestID)
	case errors.Is(err, managementcommand.ErrConflict):
		writeProviderError(response, 409, "idempotency_conflict", "Idempotency-Key was already used for a different request.", requestID)
	case errors.Is(err, delegationmanagement.ErrSessionLimit):
		writeProviderError(response, 429, "delegation_session_limit", "The delegated session limit was reached.", requestID)
	case errors.Is(err, delegationmanagement.ErrSecretResultExpired):
		writeProviderError(response, 410, "secret_result_expired", "The one-time secret delivery window expired.", requestID)
	case errors.Is(err, delegationmanagement.ErrCredentialInactive):
		writeProviderError(response, 409, "delegated_credential_inactive", "The delegated credential is inactive.", requestID)
	default:
		writeProviderError(response, 503, "delegation_service_unavailable", "Delegation service is unavailable.", requestID)
	}
}

type delegationHTTPContract struct {
	method managementapi.HTTPMethod
	path   string
}

func delegationHTTPContracts() []delegationHTTPContract {
	return []delegationHTTPContract{
		{managementapi.MethodGET, selfInferenceKeysPath},
		{managementapi.MethodGET, selfInferenceSessionsPath},
		{managementapi.MethodPOST, selfInferenceSessionsPath},
		{managementapi.MethodDELETE, selfInferenceSessionsPath + "/{sessionId}"},
		{managementapi.MethodGET, apiKeysPath + "/{keyId}/inference-sessions"},
		{managementapi.MethodDELETE, apiKeysPath + "/{keyId}/inference-sessions/{sessionId}"},
		{managementapi.MethodPOST, apiKeysPath + "/{keyId}/inference-sessions:revoke-all"},
	}
}

var _ RouteRegistrar = (*DelegationRoutes)(nil)
