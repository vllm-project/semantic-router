package managementserver

import (
	"context"
	"errors"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/runtimediagnostics"
)

const runtimeDiagnosticsPath = managementapi.BasePath + "/runtime-diagnostics"

// RuntimeDiagnosticsRoutes exposes a sanitized cluster-health read. An exact
// namespace selector adds publication and quota queue state without granting
// namespace authority or disclosing policy and credential material.
type RuntimeDiagnosticsRoutes struct {
	service       RuntimeDiagnosticsService
	sessions      SessionAuthenticator
	authorization Authorizer
	operation     managementapi.OperationContract
	now           func() time.Time
}

func NewRuntimeDiagnosticsRoutes(options RuntimeDiagnosticsRoutesOptions) (*RuntimeDiagnosticsRoutes, error) {
	if options.Service == nil || options.Sessions == nil || options.Authorization == nil {
		return nil, errors.New("runtime diagnostics routes require service, session, and authorization dependencies")
	}
	operation, found := managementapi.LookupOperation(managementapi.MethodGET, runtimeDiagnosticsPath)
	if !found {
		return nil, errors.New("runtime diagnostics operation contract is unavailable")
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	return &RuntimeDiagnosticsRoutes{
		service: options.Service, sessions: options.Sessions,
		authorization: options.Authorization, operation: operation, now: now,
	}, nil
}

func (routes *RuntimeDiagnosticsRoutes) Register(mux *http.ServeMux) {
	if routes == nil || mux == nil {
		panic("runtime diagnostics routes and mux are required")
	}
	mux.Handle("GET "+runtimeDiagnosticsPath, routes)
}

func (routes *RuntimeDiagnosticsRoutes) Ready(context.Context) error {
	if routes == nil || routes.service == nil || routes.sessions == nil || routes.authorization == nil {
		return errors.New("runtime diagnostics routes are unavailable")
	}
	return nil
}

func (routes *RuntimeDiagnosticsRoutes) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	requestID := managementRequestID(request)
	setProviderResponseHeaders(response, requestID)
	if routes == nil || request == nil || request.URL == nil || request.Method != http.MethodGet ||
		request.URL.Path != runtimeDiagnosticsPath || request.URL.EscapedPath() != request.URL.Path {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	values, err := strictProviderQuery(request.URL.RawQuery, map[string]bool{"namespaceId": true})
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Runtime diagnostics query is invalid.", requestID)
		return
	}
	namespaceID := values.Get("namespaceId")
	if namespaceID != "" && !canonicalUUID(namespaceID) {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "namespaceId must be a canonical UUID.", requestID)
		return
	}
	session, ok := routes.authenticate(response, request, requestID)
	if !ok || !routes.authorize(response, request, requestID, session) {
		return
	}
	snapshot, err := routes.service.Read(request.Context(), namespaceID)
	switch {
	case errors.Is(err, runtimediagnostics.ErrNotFound):
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	case err != nil:
		writeProviderError(response, http.StatusServiceUnavailable, "diagnostics_unavailable", "Runtime diagnostics are unavailable.", requestID)
	default:
		writeProviderJSON(response, http.StatusOK, snapshot, requestID)
	}
}

func (routes *RuntimeDiagnosticsRoutes) authenticate(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
) (managementauth.AuthenticatedSession, bool) {
	token, ok := bearerToken(request)
	if !ok {
		writeProviderError(response, http.StatusUnauthorized, "unauthenticated", "Authentication is required.", requestID)
		return managementauth.AuthenticatedSession{}, false
	}
	session, err := routes.sessions.Authenticate(request.Context(), token, "", routes.now().UTC())
	if err != nil {
		status, code, message := http.StatusServiceUnavailable, "authentication_unavailable", "Authentication state is unavailable."
		if errors.Is(err, managementauth.ErrAuthenticationDenied) {
			status, code, message = http.StatusUnauthorized, "unauthenticated", "Authentication is required."
		}
		writeProviderError(response, status, code, message, requestID)
		return managementauth.AuthenticatedSession{}, false
	}
	if session.NamespaceID != "" || !canonicalUUID(session.Session.PrincipalID) {
		writeProviderError(response, http.StatusServiceUnavailable, "authentication_unavailable", "Authentication state is unavailable.", requestID)
		return managementauth.AuthenticatedSession{}, false
	}
	return session, true
}

func (routes *RuntimeDiagnosticsRoutes) authorize(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
	session managementauth.AuthenticatedSession,
) bool {
	_, err := routes.authorization.Authorize(request.Context(), AuthorizationRequest{
		Operation: routes.operation,
		Session:   session,
	})
	if err == nil {
		return true
	}
	if errors.Is(err, managementauthorization.ErrDenied) {
		writeProviderError(response, http.StatusForbidden, "forbidden", "Permission denied.", requestID)
	} else {
		writeProviderError(response, http.StatusServiceUnavailable, "authorization_unavailable", "Authorization state is unavailable.", requestID)
	}
	return false
}
