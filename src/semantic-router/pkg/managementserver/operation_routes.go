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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policybulk"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policymanagement"
)

const operationsPath = managementapi.BasePath + "/operations"

type operationRouteContracts struct {
	list, detail, cancel         managementapi.OperationContract
	accessRead, rateRead         managementapi.OperationContract
	accessMutation, rateMutation managementapi.OperationContract
}

type OperationRoutes struct {
	service       OperationService
	details       *operationDetailRegistry
	namespaces    NamespaceResolver
	sessions      SessionAuthenticator
	authorization Authorizer
	scopes        ResultScopeResolver
	now           func() time.Time
	contracts     operationRouteContracts
}

func NewOperationRoutes(options OperationRoutesOptions) (*OperationRoutes, error) {
	scopes := configuredResultScopes(options.Scopes, options.Authorization)
	if options.Service == nil || options.Namespaces == nil || options.Sessions == nil || options.Authorization == nil || scopes == nil {
		return nil, errors.New("management Operation routes require service, namespace, session, and authorization dependencies")
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	contracts := operationRouteContracts{}
	lookups := []struct {
		method managementapi.HTTPMethod
		path   string
		target *managementapi.OperationContract
	}{
		{managementapi.MethodGET, operationsPath, &contracts.list},
		{managementapi.MethodGET, operationsPath + "/{operationId}", &contracts.detail},
		{managementapi.MethodPOST, operationsPath + "/{operationId}:cancel", &contracts.cancel},
		{managementapi.MethodGET, accessBindingsPath, &contracts.accessRead},
		{managementapi.MethodGET, rateBindingsPath, &contracts.rateRead},
		{managementapi.MethodPOST, accessBindingBulkPath, &contracts.accessMutation},
		{managementapi.MethodPOST, rateBindingBulkPath, &contracts.rateMutation},
	}
	for _, lookup := range lookups {
		contract, found := managementapi.LookupOperation(lookup.method, lookup.path)
		if !found {
			return nil, fmt.Errorf("management Operation contract %s %s is unavailable", lookup.method, lookup.path)
		}
		*lookup.target = contract
	}
	readers := make([]OperationDetailReader, 0, 1+len(options.DetailReaders))
	readers = append(readers, &policyBulkOperationDetailReader{
		service:       options.Service,
		authorization: options.Authorization, contracts: contracts,
	})
	readers = append(readers, options.DetailReaders...)
	details, err := newOperationDetailRegistry(readers)
	if err != nil {
		return nil, err
	}
	return &OperationRoutes{
		service: options.Service, details: details, namespaces: options.Namespaces,
		sessions: options.Sessions, authorization: options.Authorization, scopes: scopes, now: now, contracts: contracts,
	}, nil
}

func (routes *OperationRoutes) Register(mux *http.ServeMux) {
	if routes == nil || mux == nil {
		panic("Management Operation routes and mux are required")
	}
	mux.Handle("GET "+operationsPath, routes)
	mux.Handle("GET "+operationsPath+"/", routes)
	mux.Handle("POST "+operationsPath+"/", routes)
}

func (routes *OperationRoutes) Ready(ctx context.Context) error {
	if routes == nil || routes.service == nil {
		return policybulk.ErrUnavailable
	}
	return routes.service.Ready(ctx)
}

func (routes *OperationRoutes) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	requestID := managementRequestID(request)
	setProviderResponseHeaders(response, requestID)
	if routes == nil || request == nil || request.URL == nil || request.URL.EscapedPath() != request.URL.Path {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	if request.Method == http.MethodGet && request.URL.Path == operationsPath {
		routes.list(response, request, requestID)
		return
	}
	operationID, cancel, valid := parseOperationPath(request.URL.Path)
	if !valid || (cancel && request.Method != http.MethodPost) || (!cancel && request.Method != http.MethodGet) {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	if cancel {
		routes.cancel(response, request, requestID, operationID)
		return
	}
	routes.detail(response, request, requestID, operationID)
}

func (routes *OperationRoutes) list(response http.ResponseWriter, request *http.Request, requestID string) {
	if operationRequestHasBody(request) {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Operation list does not accept a request body.", requestID)
		return
	}
	query, err := strictProviderQuery(request.URL.RawQuery, map[string]bool{
		"cursor": true, "pageSize": true, "kind": true, "state": true, "originPrincipalId": true,
	})
	pageSize, pageErr := parseOptionalPageSize(query.Get("pageSize"))
	if err != nil || pageErr != nil ||
		(query.Get("originPrincipalId") != "" && !canonicalUUID(query.Get("originPrincipalId"))) ||
		!validOperationKindFilter(query.Get("kind")) || !validOperationStateFilter(query.Get("state")) {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Operation list query is invalid.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	operationScope, err := resolveListResultScope(request.Context(), routes.scopes, session, namespaceID,
		accesscontrol.PermissionOperationRead)
	if err != nil {
		writeResultScopeError(response, err, requestID)
		return
	}
	accessScope, err := resolveListResultScope(request.Context(), routes.scopes, session, namespaceID,
		accesscontrol.PermissionAccessPolicyRead)
	if err != nil {
		writeResultScopeError(response, err, requestID)
		return
	}
	rateScope, err := resolveListResultScope(request.Context(), routes.scopes, session, namespaceID,
		accesscontrol.PermissionRatePolicyRead)
	if err != nil {
		writeResultScopeError(response, err, requestID)
		return
	}
	page, err := routes.service.List(request.Context(), policybulk.ListRequest{
		NamespaceID: namespaceID, OriginPrincipalID: query.Get("originPrincipalId"),
		Kind: query.Get("kind"), State: policybulk.OperationState(query.Get("state")),
		Cursor: query.Get("cursor"), PageSize: pageSize,
		Visibility: policybulk.OperationVisibility{
			PrincipalID: session.Session.PrincipalID,
			Operation:   operationScope, Access: accessScope, Rate: rateScope,
		},
	})
	if err != nil {
		writeOperationError(response, err, requestID)
		return
	}
	writeProviderJSON(response, http.StatusOK, newPolicyBulkOperationPage(page), requestID)
}

func (routes *OperationRoutes) detail(response http.ResponseWriter, request *http.Request, requestID, operationID string) {
	if request.URL.RawQuery != "" || operationRequestHasBody(request) {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Operation detail does not accept query parameters or a request body.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	detail, err := routes.details.Read(request.Context(), OperationDetailReadRequest{
		NamespaceID: namespaceID, OperationID: operationID, Session: session,
	})
	if err != nil {
		writeOperationDetailError(response, err, requestID)
		return
	}
	response.Header().Set(managementapi.HeaderETag, operationETag(detail.Version))
	writeProviderJSON(response, http.StatusOK, detail.Operation, requestID)
}

func (routes *OperationRoutes) cancel(response http.ResponseWriter, request *http.Request, requestID, operationID string) {
	if request.URL.RawQuery != "" || operationRequestHasBody(request) {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Operation cancel does not accept query parameters or a request body.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	operation, cancelErr := routes.service.Get(request.Context(), namespaceID, operationID)
	if cancelErr != nil {
		writeOperationError(response, cancelErr, requestID)
		return
	}
	if err := routes.authorizeStoredOperation(request.Context(), session, operation, operationAuthorizationCancel); err != nil {
		if operationDenied(err) {
			writeProviderError(response, http.StatusNotFound, "not_found", "Operation not found.", requestID)
		} else {
			writeProviderError(response, http.StatusServiceUnavailable, "authorization_unavailable", "Authorization state is unavailable.", requestID)
		}
		return
	}
	revision, ok := requireOperationRevision(response, request, requestID)
	if !ok {
		return
	}
	idempotencyKey, ok := requireIdempotencyKey(response, request, requestID)
	if !ok {
		return
	}
	result, cancelErr := routes.service.Cancel(request.Context(), policybulk.CancelRequest{
		NamespaceID: namespaceID, OperationID: operationID, ExpectedVersion: revision,
		IdempotencyKey: string(idempotencyKey), Actor: policymanagement.Actor{
			PrincipalID: session.Session.PrincipalID, ActorChain: []string{session.Session.PrincipalID},
			RequestID: requestID, SourceIP: directRequestIP(request),
		},
	})
	if cancelErr != nil {
		writeOperationError(response, cancelErr, requestID)
		return
	}
	response.Header().Set(managementapi.HeaderETag, operationETag(result.Operation.Version))
	setIdempotencyReplayHeader(response, result.Replayed)
	writeProviderJSON(response, http.StatusOK, newPolicyBulkOperation(result.Operation), requestID)
}

func (routes *OperationRoutes) authenticate(response http.ResponseWriter, request *http.Request, requestID string) (string, managementauth.AuthenticatedSession, bool) {
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

func parseOperationPath(path string) (string, bool, bool) {
	value := strings.TrimPrefix(path, operationsPath+"/")
	if value == path || value == "" || strings.Contains(value, "/") {
		return "", false, false
	}
	cancel := strings.HasSuffix(value, ":cancel")
	if cancel {
		value = strings.TrimSuffix(value, ":cancel")
	}
	return value, cancel, canonicalUUID(value)
}

func validOperationKindFilter(value string) bool {
	return value == "" || value == policybulk.AccessBindingOperationKind || value == policybulk.RateBindingOperationKind
}

func validOperationStateFilter(value string) bool {
	return value == "" || policybulk.OperationState(value).Valid()
}

var _ RouteRegistrar = (*OperationRoutes)(nil)
