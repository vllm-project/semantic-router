package managementserver

import (
	"encoding/json"
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity"
)

func (routes *IdentityLifecycleRoutes) me(response http.ResponseWriter, request *http.Request) {
	requestID := managementRequestID(request)
	if !validLifecycleRequest(response, request, requestID, false) || !noRequestBody(response, request, requestID) {
		return
	}
	session, ok := routes.authenticatedAndAuthorized(response, request, requestID, managementapi.MethodGET, mePath)
	if !ok {
		return
	}
	view, err := routes.service.Me(request.Context(), session)
	if err != nil {
		writeIdentityError(response, err, requestID)
		return
	}
	writeProviderJSON(response, http.StatusOK, selfViewDTO(view), requestID)
}

func (routes *IdentityLifecycleRoutes) listSelfManagementSessions(response http.ResponseWriter, request *http.Request) {
	requestID := managementRequestID(request)
	if !validLifecycleRequest(response, request, requestID, true) || !noRequestBody(response, request, requestID) {
		return
	}
	pageRequest, ok := identityPageRequest(response, request, requestID)
	if !ok {
		return
	}
	session, ok := routes.authenticatedAndAuthorized(response, request, requestID, managementapi.MethodGET, selfManagementSessionPath)
	if !ok {
		return
	}
	page, err := routes.service.ListManagementSessions(request.Context(), session.Session.PrincipalID, pageRequest)
	if err != nil {
		writeIdentityError(response, err, requestID)
		return
	}
	writeManagementSessionPage(response, page, pageRequest.Limit, requestID)
}

func (routes *IdentityLifecycleRoutes) revokeSelfManagementSession(response http.ResponseWriter, request *http.Request) {
	requestID := managementRequestID(request)
	if !validLifecycleRequest(response, request, requestID, false) || !noRequestBody(response, request, requestID) {
		return
	}
	sessionID := request.PathValue("sessionId")
	if !canonicalUUID(sessionID) {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	session, ok := routes.authenticatedAndAuthorized(response, request, requestID, managementapi.MethodDELETE, selfManagementSessionPath+"/{sessionId}")
	if !ok {
		return
	}
	mutation, err := routes.service.RevokeSelfManagementSession(
		request.Context(), session.Session.PrincipalID, sessionID,
		identityActor(request, session, requestID, "Revoke own Management session"),
	)
	if err != nil {
		writeIdentityError(response, err, requestID)
		return
	}
	_ = mutation
	response.WriteHeader(http.StatusNoContent)
}

func (routes *IdentityLifecycleRoutes) revokeManagementSession(response http.ResponseWriter, request *http.Request) {
	requestID := managementRequestID(request)
	if !validLifecycleRequest(response, request, requestID, false) {
		return
	}
	sessionID, pathOK := lifecycleActionID(request.URL.Path, managementSessionPath, ":revoke")
	if !pathOK {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	session, ok := routes.authenticatedAndAuthorized(response, request, requestID, managementapi.MethodPOST, managementSessionPath+"/{sessionId}:revoke")
	if !ok {
		return
	}
	key, ok := requireIdempotencyKey(response, request, requestID)
	if !ok {
		return
	}
	var body managementapi.ManagementSessionRevokeRequest
	if !decodeIdentityBody(response, request, requestID, &body) {
		return
	}
	canonical, err := json.Marshal(body)
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Request is invalid.", requestID)
		return
	}
	now := routes.now().UTC()
	command, err := routes.commands.Bind(
		managementcommand.ClusterCommandScope(), session.Session.PrincipalID, request.URL.Path,
		string(key), canonical, now, now.Add(identityCommandTTL),
	)
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Request is invalid.", requestID)
		return
	}
	mutation, result, err := routes.service.RevokeManagementSession(request.Context(), managementidentity.SessionRevocationCommand{
		SessionID: sessionID, Command: command,
		Actor: identityActor(request, session, requestID, body.Reason),
	})
	if err != nil {
		writeIdentityError(response, err, requestID)
		return
	}
	setIdempotencyReplayHeader(response, result.Replayed)
	writeProviderJSON(response, http.StatusOK, managementSessionRevocationDTO(mutation), requestID)
}

func (routes *IdentityLifecycleRoutes) listPrincipalManagementSessions(response http.ResponseWriter, request *http.Request) {
	requestID := managementRequestID(request)
	if !validLifecycleRequest(response, request, requestID, true) || !noRequestBody(response, request, requestID) {
		return
	}
	principalID := request.PathValue("principalId")
	if !canonicalUUID(principalID) {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	pageRequest, ok := identityPageRequest(response, request, requestID)
	if !ok {
		return
	}
	_, ok = routes.authenticatedAndAuthorized(response, request, requestID, managementapi.MethodGET, principalPath+"/{principalId}/management-sessions")
	if !ok {
		return
	}
	page, err := routes.service.ListManagementSessions(request.Context(), principalID, pageRequest)
	if err != nil {
		writeIdentityError(response, err, requestID)
		return
	}
	writeManagementSessionPage(response, page, pageRequest.Limit, requestID)
}

func (routes *IdentityLifecycleRoutes) revokePrincipalManagementSessions(response http.ResponseWriter, request *http.Request) {
	requestID := managementRequestID(request)
	if !validLifecycleRequest(response, request, requestID, false) {
		return
	}
	principalID := request.PathValue("principalId")
	if !canonicalUUID(principalID) {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	session, ok := routes.authenticatedAndAuthorized(response, request, requestID, managementapi.MethodPOST, principalPath+"/{principalId}/management-sessions:revoke-all")
	if !ok {
		return
	}
	key, ok := requireIdempotencyKey(response, request, requestID)
	if !ok {
		return
	}
	var body managementapi.ManagementSessionRevokeRequest
	if !decodeIdentityBody(response, request, requestID, &body) {
		return
	}
	canonical, err := json.Marshal(body)
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Request is invalid.", requestID)
		return
	}
	now := routes.now().UTC()
	command, err := routes.commands.Bind(
		managementcommand.ClusterCommandScope(), session.Session.PrincipalID, request.URL.Path,
		string(key), canonical, now, now.Add(identityCommandTTL),
	)
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Request is invalid.", requestID)
		return
	}
	result, err := routes.service.RevokePrincipalManagementSessions(request.Context(), managementidentity.PrincipalSessionRevocationCommand{
		PrincipalID: principalID, Command: command,
		Actor: identityActor(request, session, requestID, body.Reason),
	})
	if err != nil {
		writeIdentityError(response, err, requestID)
		return
	}
	setIdempotencyReplayHeader(response, result.Result.Replayed)
	writeProviderJSON(response, http.StatusOK, managementapi.PrincipalManagementSessionsRevocation{
		PrincipalID: principalID, RevokedCount: result.RevokedCount, AlreadyRevoked: result.AlreadyRevoked,
	}, requestID)
}

func writeManagementSessionPage(
	response http.ResponseWriter,
	page managementidentity.ManagementSessionPage,
	pageSize int,
	requestID string,
) {
	data := make([]managementapi.ManagementSession, len(page.Items))
	for index := range page.Items {
		data[index] = managementSessionDTO(page.Items[index])
	}
	writeProviderJSON(response, http.StatusOK, managementapi.ManagementSessionPage{
		Data: data, Page: identityPageInfo(page.NextCursor, pageSize),
	}, requestID)
}

func managementSessionDTO(value managementidentity.ManagementSession) managementapi.ManagementSession {
	return managementapi.ManagementSession{
		SessionID: value.ID, PrincipalID: value.PrincipalID,
		AuthSourceKind: string(value.AuthSourceKind), EvidenceKind: string(value.EvidenceKind),
		AuthenticatedAt: value.AuthenticatedAt, ExpiresAt: value.ExpiresAt,
		Status: string(value.Status), RevokedAt: value.RevokedAt, CreatedAt: value.CreatedAt,
	}
}

func managementSessionRevocationDTO(value managementauth.SessionMutation) managementapi.ManagementSessionRevocation {
	return managementapi.ManagementSessionRevocation{
		SessionID: value.SessionID, Status: string(managementauth.SessionRevoked),
		RevokedAt: value.ChangedAt, Changed: value.Changed,
	}
}
