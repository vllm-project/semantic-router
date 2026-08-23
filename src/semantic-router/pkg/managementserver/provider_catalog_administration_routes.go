package managementserver

import (
	"errors"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
)

const (
	providerCatalogBootstrapPath = managementapi.BasePath + "/provider-catalog:bootstrap"
	providerCatalogActivatePath  = managementapi.BasePath + "/provider-catalog:activate"
)

type ProviderCatalogAdministrationRoutes struct {
	administration     ProviderCatalogAdministration
	sessions           SessionAuthenticator
	authorization      Authorizer
	now                func() time.Time
	bootstrapOperation managementapi.OperationContract
	activateOperation  managementapi.OperationContract
}

func NewProviderCatalogAdministrationRoutes(
	options ProviderCatalogAdministrationRoutesOptions,
) (*ProviderCatalogAdministrationRoutes, error) {
	if options.Administration == nil || options.Sessions == nil || options.Authorization == nil {
		return nil, errors.New("provider Catalog administration routes require lifecycle, session, and authorization dependencies")
	}
	bootstrap, bootstrapFound := managementapi.LookupOperation(managementapi.MethodPOST, providerCatalogBootstrapPath)
	activate, activateFound := managementapi.LookupOperation(managementapi.MethodPOST, providerCatalogActivatePath)
	if !bootstrapFound || !activateFound {
		return nil, errors.New("provider Catalog administration operation contracts are unavailable")
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	return &ProviderCatalogAdministrationRoutes{
		administration: options.Administration, sessions: options.Sessions,
		authorization: options.Authorization, now: now,
		bootstrapOperation: bootstrap, activateOperation: activate,
	}, nil
}

func (routes *ProviderCatalogAdministrationRoutes) Register(mux *http.ServeMux) {
	if routes == nil || mux == nil {
		panic("Provider Catalog administration routes and mux are required")
	}
	mux.Handle("POST "+providerCatalogBootstrapPath, routes)
	mux.Handle("POST "+providerCatalogActivatePath, routes)
}

func (routes *ProviderCatalogAdministrationRoutes) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	requestID := managementRequestID(request)
	setProviderResponseHeaders(response, requestID)
	if routes == nil || request == nil || request.URL == nil || request.URL.EscapedPath() != request.URL.Path ||
		request.Method != http.MethodPost || request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	var operation managementapi.OperationContract
	switch request.URL.Path {
	case providerCatalogBootstrapPath:
		operation = routes.bootstrapOperation
	case providerCatalogActivatePath:
		operation = routes.activateOperation
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	session, ok := routes.authenticate(response, request, requestID)
	if !ok || !routes.authorize(response, request, requestID, session, operation) {
		return
	}
	if request.URL.Path == providerCatalogBootstrapPath {
		routes.bootstrap(response, request, requestID)
		return
	}
	routes.activate(response, request, requestID)
}

func (routes *ProviderCatalogAdministrationRoutes) bootstrap(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
) {
	var body managementapi.ProviderCatalogBootstrapRequest
	if err := decodeStrictProviderJSON(response, request, &body); err != nil {
		writeProviderCatalogAdministrationDecodeError(response, err, requestID)
		return
	}
	generation, err := body.Generation()
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "expectedGeneration is invalid.", requestID)
		return
	}
	state, err := routes.administration.BootstrapRegistry(request.Context(), generation)
	if err != nil {
		writeProviderCatalogAdministrationError(response, err, requestID)
		return
	}
	writeProviderJSON(response, http.StatusOK, providerCatalogPublicationDTO(state), requestID)
}

func (routes *ProviderCatalogAdministrationRoutes) activate(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
) {
	var body managementapi.ProviderCatalogActivateRequest
	if err := decodeStrictProviderJSON(response, request, &body); err != nil {
		writeProviderCatalogAdministrationDecodeError(response, err, requestID)
		return
	}
	generation, err := body.Generation()
	if err != nil || !validAuthorityDigest(body.Revision) {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "revision or expectedGeneration is invalid.", requestID)
		return
	}
	state, err := routes.administration.Activate(request.Context(), body.Revision, generation)
	if err != nil {
		writeProviderCatalogAdministrationError(response, err, requestID)
		return
	}
	writeProviderJSON(response, http.StatusOK, providerCatalogPublicationDTO(state), requestID)
}

func (routes *ProviderCatalogAdministrationRoutes) authenticate(
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

func (routes *ProviderCatalogAdministrationRoutes) authorize(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
	session managementauth.AuthenticatedSession,
	operation managementapi.OperationContract,
) bool {
	_, err := routes.authorization.Authorize(request.Context(), AuthorizationRequest{
		Operation: operation, Session: session,
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

func writeProviderCatalogAdministrationDecodeError(response http.ResponseWriter, err error, requestID string) {
	if errors.Is(err, errProviderBodyTooLarge) {
		writeProviderError(response, http.StatusRequestEntityTooLarge, "invalid_request", "Request body is too large.", requestID)
		return
	}
	writeProviderError(response, http.StatusBadRequest, "invalid_request", "Request body is invalid.", requestID)
}

func writeProviderCatalogAdministrationError(response http.ResponseWriter, err error, requestID string) {
	switch {
	case errors.Is(err, providercatalog.ErrPublicationConflict):
		writeProviderError(response, http.StatusConflict, "catalog_conflict", "Provider catalog state changed. Refresh and retry.", requestID)
	case errors.Is(err, providercatalog.ErrActivationBlocked):
		var blocked *providercatalog.ActivationBlockedError
		details := []managementapi.ErrorDetail(nil)
		if errors.As(err, &blocked) {
			details = providerCatalogBlockerDetails(blocked.Blockers)
		}
		writeProviderJSON(response, http.StatusConflict, managementapi.ErrorResponse{Error: managementapi.APIError{
			Code: "activation_blocked", Message: "Required rollout groups are not ready.",
			RequestID: requestID, Details: details,
		}}, requestID)
	default:
		writeProviderError(response, http.StatusServiceUnavailable, "catalog_unavailable", "Provider catalog is unavailable.", requestID)
	}
}

func providerCatalogBlockerDetails(blockers providercatalog.ActivationBlockers) []managementapi.ErrorDetail {
	details := make([]managementapi.ErrorDetail, 0,
		len(blockers.Missing)+len(blockers.Expired)+len(blockers.Divergent)+len(blockers.Incompatible))
	appendGroups := func(groups []string, reason string) {
		for _, group := range groups {
			details = append(details, managementapi.ErrorDetail{Field: "rolloutGroup", Reason: reason + ": " + group})
		}
	}
	missing := make([]string, len(blockers.Missing))
	for index, group := range blockers.Missing {
		missing[index] = group.Key()
	}
	expired := make([]string, len(blockers.Expired))
	for index, group := range blockers.Expired {
		expired[index] = group.Key()
	}
	divergent := make([]string, len(blockers.Divergent))
	for index, group := range blockers.Divergent {
		divergent[index] = group.Key()
	}
	incompatible := make([]string, len(blockers.Incompatible))
	for index, blocker := range blockers.Incompatible {
		incompatible[index] = blocker.RolloutGroup.Key()
	}
	appendGroups(missing, "missing compatible lease")
	appendGroups(expired, "lease expired")
	appendGroups(divergent, "capability mismatch")
	appendGroups(incompatible, "replica incompatible")
	return details
}
