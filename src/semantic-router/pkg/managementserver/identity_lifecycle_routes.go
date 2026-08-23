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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity"
)

const (
	mePath                    = managementapi.BasePath + "/me"
	selfManagementSessionPath = managementapi.BasePath + "/self/management-sessions"
	managementSessionPath     = managementapi.BasePath + "/management-sessions"
	trustedIssuerPath         = managementapi.BasePath + "/trusted-identity-issuers"
	backchannelLogoutPath     = managementapi.BasePath + "/auth/backchannel-logout"
)

type IdentityLifecycleService interface {
	Ready(context.Context) error
	Me(context.Context, managementauth.AuthenticatedSession) (managementidentity.SelfView, error)
	ListManagementSessions(context.Context, string, managementidentity.ListRequest) (managementidentity.ManagementSessionPage, error)
	RevokeSelfManagementSession(context.Context, string, string, managementidentity.MutationActor) (managementauth.SessionMutation, error)
	RevokeManagementSession(context.Context, managementidentity.SessionRevocationCommand) (managementauth.SessionMutation, managementidentity.MutationResult, error)
	RevokePrincipalManagementSessions(context.Context, managementidentity.PrincipalSessionRevocationCommand) (managementidentity.PrincipalSessionRevocation, error)
	GetTrustedIdentityIssuer(context.Context, string) (managementidentity.TrustedIdentityIssuer, error)
	ListTrustedIdentityIssuers(context.Context, managementidentity.ListRequest) (managementidentity.TrustedIdentityIssuerPage, error)
	CreateTrustedIdentityIssuer(context.Context, managementidentity.CreateTrustedIdentityIssuer) (managementidentity.IssuerMutation, error)
	UpdateTrustedIdentityIssuer(context.Context, managementidentity.UpdateTrustedIdentityIssuer) (managementidentity.IssuerMutation, error)
	DeleteTrustedIdentityIssuer(context.Context, string, uint64, managementidentity.MutationActor) (managementidentity.IssuerMutation, error)
	RefreshTrustedIdentityIssuer(context.Context, managementidentity.RefreshTrustedIdentityIssuer) (managementidentity.IssuerMutation, error)
	BackchannelLogout(context.Context, string, string, string, time.Time) (managementidentity.BackchannelLogoutResult, error)
}

type IdentityLifecycleRoutesOptions struct {
	Service                IdentityLifecycleService
	Sessions               SessionAuthenticator
	Authorization          Authorizer
	Commands               *managementcommand.Codec
	AllowPlaintextForTests bool
	Now                    func() time.Time
}

type IdentityLifecycleRoutes struct {
	service        IdentityLifecycleService
	sessions       SessionAuthenticator
	authorization  Authorizer
	commands       *managementcommand.Codec
	allowPlaintext bool
	now            func() time.Time
	operations     map[string]managementapi.OperationContract
}

func NewIdentityLifecycleRoutes(options IdentityLifecycleRoutesOptions) (*IdentityLifecycleRoutes, error) {
	if options.Service == nil || options.Sessions == nil || options.Authorization == nil || options.Commands == nil {
		return nil, errors.New("management identity lifecycle routes require service, session, authorization, and command dependencies")
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	routes := &IdentityLifecycleRoutes{
		service: options.Service, sessions: options.Sessions, authorization: options.Authorization,
		commands: options.Commands, allowPlaintext: options.AllowPlaintextForTests,
		now: now, operations: make(map[string]managementapi.OperationContract),
	}
	for _, contract := range identityLifecycleHTTPContracts() {
		operation, found := managementapi.LookupOperation(contract.method, contract.path)
		if !found {
			return nil, fmt.Errorf("management identity lifecycle operation %s %s is unavailable", contract.method, contract.path)
		}
		routes.operations[string(contract.method)+" "+contract.path] = operation
	}
	return routes, nil
}

func (routes *IdentityLifecycleRoutes) Register(mux *http.ServeMux) {
	if routes == nil || mux == nil {
		panic("Management identity lifecycle routes and mux are required")
	}
	mux.HandleFunc("GET "+mePath, routes.me)
	mux.HandleFunc("GET "+selfManagementSessionPath, routes.listSelfManagementSessions)
	mux.HandleFunc("DELETE "+selfManagementSessionPath+"/{sessionId}", routes.revokeSelfManagementSession)
	mux.HandleFunc("POST "+managementSessionPath+"/", routes.revokeManagementSession)
	mux.HandleFunc("GET "+principalPath+"/{principalId}/management-sessions", routes.listPrincipalManagementSessions)
	mux.HandleFunc("POST "+principalPath+"/{principalId}/management-sessions:revoke-all", routes.revokePrincipalManagementSessions)
	mux.HandleFunc("GET "+trustedIssuerPath, routes.trustedIssuers)
	mux.HandleFunc("POST "+trustedIssuerPath, routes.trustedIssuers)
	mux.HandleFunc("GET "+trustedIssuerPath+"/{issuerId}", routes.trustedIssuer)
	mux.HandleFunc("PATCH "+trustedIssuerPath+"/{issuerId}", routes.trustedIssuer)
	mux.HandleFunc("DELETE "+trustedIssuerPath+"/{issuerId}", routes.trustedIssuer)
	mux.HandleFunc("POST "+trustedIssuerPath+"/", routes.refreshTrustedIssuerKeys)
	mux.HandleFunc("POST "+backchannelLogoutPath, routes.backchannelLogout)
}

func (routes *IdentityLifecycleRoutes) Ready(ctx context.Context) error {
	if routes == nil || routes.service == nil {
		return errors.New("management identity lifecycle routes are unavailable")
	}
	return routes.service.Ready(ctx)
}

func (routes *IdentityLifecycleRoutes) authenticate(
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
		writeAuthenticationError(response, err, requestID)
		return managementauth.AuthenticatedSession{}, false
	}
	if session.NamespaceID != "" {
		writeProviderError(response, http.StatusServiceUnavailable, "authentication_unavailable", "Authentication state is unavailable.", requestID)
		return managementauth.AuthenticatedSession{}, false
	}
	return session, true
}

func (routes *IdentityLifecycleRoutes) authorize(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
	session managementauth.AuthenticatedSession,
	operation managementapi.OperationContract,
	targets map[string][]accesscontrol.ScopedTarget,
) bool {
	_, err := routes.authorization.Authorize(request.Context(), AuthorizationRequest{
		Operation: operation, Session: session, Targets: targets,
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

func (routes *IdentityLifecycleRoutes) authenticatedAndAuthorized(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
	method managementapi.HTTPMethod,
	path string,
) (managementauth.AuthenticatedSession, bool) {
	session, ok := routes.authenticate(response, request, requestID)
	if !ok || !routes.authorize(response, request, requestID, session, routes.operation(method, path), nil) {
		return managementauth.AuthenticatedSession{}, false
	}
	return session, true
}

func (routes *IdentityLifecycleRoutes) operation(method managementapi.HTTPMethod, path string) managementapi.OperationContract {
	return routes.operations[string(method)+" "+path]
}

type identityLifecycleHTTPContract struct {
	method managementapi.HTTPMethod
	path   string
}

func identityLifecycleHTTPContracts() []identityLifecycleHTTPContract {
	return []identityLifecycleHTTPContract{
		{managementapi.MethodGET, mePath},
		{managementapi.MethodGET, selfManagementSessionPath},
		{managementapi.MethodDELETE, selfManagementSessionPath + "/{sessionId}"},
		{managementapi.MethodPOST, managementSessionPath + "/{sessionId}:revoke"},
		{managementapi.MethodGET, principalPath + "/{principalId}/management-sessions"},
		{managementapi.MethodPOST, principalPath + "/{principalId}/management-sessions:revoke-all"},
		{managementapi.MethodGET, trustedIssuerPath},
		{managementapi.MethodPOST, trustedIssuerPath},
		{managementapi.MethodGET, trustedIssuerPath + "/{issuerId}"},
		{managementapi.MethodPATCH, trustedIssuerPath + "/{issuerId}"},
		{managementapi.MethodDELETE, trustedIssuerPath + "/{issuerId}"},
		{managementapi.MethodPOST, trustedIssuerPath + "/{issuerId}:refresh-keys"},
		{managementapi.MethodPOST, backchannelLogoutPath},
	}
}

func validLifecycleRequest(response http.ResponseWriter, request *http.Request, requestID string, allowQuery bool) bool {
	setProviderResponseHeaders(response, requestID)
	if request == nil || request.URL == nil || request.URL.EscapedPath() != request.URL.Path {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return false
	}
	if !allowQuery && request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Query parameters are not accepted.", requestID)
		return false
	}
	return true
}

func lifecycleActionID(path, base, suffix string) (string, bool) {
	value := strings.TrimPrefix(path, base+"/")
	if value == path || !strings.HasSuffix(value, suffix) {
		return "", false
	}
	value = strings.TrimSuffix(value, suffix)
	return value, !strings.Contains(value, "/") && canonicalUUID(value)
}

var _ RouteRegistrar = (*IdentityLifecycleRoutes)(nil)
