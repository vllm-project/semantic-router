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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policymanagement"
)

const (
	accessPoliciesPath    = managementapi.BasePath + "/access-policies"
	accessBindingsPath    = managementapi.BasePath + "/access-policy-bindings"
	ratePoliciesPath      = managementapi.BasePath + "/rate-limit-policies"
	rateBindingsPath      = managementapi.BasePath + "/rate-limit-bindings"
	accessBindingBulkPath = accessBindingsPath + ":bulk-apply"
	rateBindingBulkPath   = rateBindingsPath + ":bulk-apply"
)

type PolicyRoutes struct {
	service       PolicyManagementService
	bulk          PolicyBulkService
	namespaces    NamespaceResolver
	sessions      SessionAuthenticator
	authorization Authorizer
	scopes        ResultScopeResolver
	now           func() time.Time
	operations    map[string]managementapi.OperationContract
}

func NewPolicyRoutes(options PolicyRoutesOptions) (*PolicyRoutes, error) {
	scopes := configuredResultScopes(options.Scopes, options.Authorization)
	if options.Service == nil || options.Bulk == nil || options.Namespaces == nil ||
		options.Sessions == nil || options.Authorization == nil || scopes == nil {
		return nil, errors.New("policy Management routes require service, bulk, namespace, session, and authorization dependencies")
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	routes := &PolicyRoutes{
		service: options.Service, bulk: options.Bulk,
		namespaces: options.Namespaces, sessions: options.Sessions,
		authorization: options.Authorization, scopes: scopes, now: now,
		operations: make(map[string]managementapi.OperationContract),
	}
	for _, contract := range policyHTTPContracts() {
		operation, found := managementapi.LookupOperation(contract.method, contract.path)
		if !found {
			return nil, fmt.Errorf("policy Management operation contract %s %s is unavailable", contract.method, contract.path)
		}
		routes.operations[policyOperationKey(contract.method, contract.path)] = operation
	}
	return routes, nil
}

func (routes *PolicyRoutes) Register(mux *http.ServeMux) {
	if routes == nil || mux == nil {
		panic("Policy Management routes and mux are required")
	}
	for _, path := range []string{accessPoliciesPath, accessBindingsPath, ratePoliciesPath, rateBindingsPath} {
		mux.Handle("GET "+path, routes)
		mux.Handle("POST "+path, routes)
		mux.Handle("GET "+path+"/", routes)
		mux.Handle("PATCH "+path+"/", routes)
		mux.Handle("DELETE "+path+"/", routes)
	}
	mux.Handle("POST "+accessBindingBulkPath, routes)
	mux.Handle("POST "+rateBindingBulkPath, routes)
}

func (routes *PolicyRoutes) Ready(ctx context.Context) error {
	if routes == nil || routes.service == nil || routes.bulk == nil {
		return policymanagement.ErrUnavailable
	}
	if err := routes.service.Ready(ctx); err != nil {
		return err
	}
	return routes.bulk.Ready(ctx)
}

func (routes *PolicyRoutes) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	requestID := managementRequestID(request)
	setProviderResponseHeaders(response, requestID)
	if routes == nil || request == nil || request.URL == nil || request.URL.EscapedPath() != request.URL.Path {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	switch request.URL.Path {
	case accessPoliciesPath:
		routes.accessPolicyCollection(response, request, requestID)
		return
	case ratePoliciesPath:
		routes.ratePolicyCollection(response, request, requestID)
		return
	case accessBindingsPath:
		routes.accessBindingCollection(response, request, requestID)
		return
	case rateBindingsPath:
		routes.rateBindingCollection(response, request, requestID)
		return
	case accessBindingBulkPath:
		if request.Method == http.MethodPost {
			routes.bulkAccessBindings(response, request, requestID)
			return
		}
	case rateBindingBulkPath:
		if request.Method == http.MethodPost {
			routes.bulkRateBindings(response, request, requestID)
			return
		}
	}
	for _, candidate := range []struct {
		base string
		kind policyResourceKind
	}{
		{accessPoliciesPath, policyResourceAccessPolicy},
		{ratePoliciesPath, policyResourceRatePolicy},
		{accessBindingsPath, policyResourceAccessBinding},
		{rateBindingsPath, policyResourceRateBinding},
	} {
		if id, ok := policyPathID(request.URL.Path, candidate.base); ok {
			routes.policyResource(response, request, requestID, candidate.kind, id)
			return
		}
	}
	writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
}

func (routes *PolicyRoutes) accessPolicyCollection(response http.ResponseWriter, request *http.Request, requestID string) {
	switch request.Method {
	case http.MethodGet:
		routes.listAccessPolicies(response, request, requestID)
	case http.MethodPost:
		routes.createAccessPolicy(response, request, requestID)
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
}

func (routes *PolicyRoutes) ratePolicyCollection(response http.ResponseWriter, request *http.Request, requestID string) {
	switch request.Method {
	case http.MethodGet:
		routes.listRatePolicies(response, request, requestID)
	case http.MethodPost:
		routes.createRatePolicy(response, request, requestID)
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
}

func (routes *PolicyRoutes) accessBindingCollection(response http.ResponseWriter, request *http.Request, requestID string) {
	switch request.Method {
	case http.MethodGet:
		routes.listAccessBindings(response, request, requestID)
	case http.MethodPost:
		routes.createAccessBinding(response, request, requestID)
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
}

func (routes *PolicyRoutes) rateBindingCollection(response http.ResponseWriter, request *http.Request, requestID string) {
	switch request.Method {
	case http.MethodGet:
		routes.listRateBindings(response, request, requestID)
	case http.MethodPost:
		routes.createRateBinding(response, request, requestID)
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
}

type policyResourceKind uint8

const (
	policyResourceAccessPolicy policyResourceKind = iota + 1
	policyResourceRatePolicy
	policyResourceAccessBinding
	policyResourceRateBinding
)

func (routes *PolicyRoutes) policyResource(response http.ResponseWriter, request *http.Request, requestID string, kind policyResourceKind, id string) {
	switch kind {
	case policyResourceAccessPolicy:
		routes.accessPolicyResource(response, request, requestID, id)
	case policyResourceRatePolicy:
		routes.ratePolicyResource(response, request, requestID, id)
	case policyResourceAccessBinding:
		routes.accessBindingResource(response, request, requestID, id)
	case policyResourceRateBinding:
		routes.rateBindingResource(response, request, requestID, id)
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
}

func policyPathID(path, base string) (string, bool) {
	value := strings.TrimPrefix(path, base+"/")
	return value, value != path && canonicalUUID(value) && !strings.Contains(value, "/")
}

func (routes *PolicyRoutes) authenticate(response http.ResponseWriter, request *http.Request, requestID string) (string, managementauth.AuthenticatedSession, bool) {
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

func (routes *PolicyRoutes) authorize(response http.ResponseWriter, request *http.Request, requestID string,
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

func (routes *PolicyRoutes) actor(request *http.Request, session managementauth.AuthenticatedSession, requestID string) policymanagement.Actor {
	return policymanagement.Actor{
		PrincipalID: session.Session.PrincipalID,
		ActorChain:  []string{session.Session.PrincipalID}, RequestID: requestID, SourceIP: directRequestIP(request),
	}
}

func (routes *PolicyRoutes) operation(method managementapi.HTTPMethod, path string) managementapi.OperationContract {
	return routes.operations[policyOperationKey(method, path)]
}

type policyHTTPContract struct {
	method managementapi.HTTPMethod
	path   string
}

func policyHTTPContracts() []policyHTTPContract {
	contracts := make([]policyHTTPContract, 0, 22)
	for _, resource := range []struct {
		base, parameter string
	}{
		{accessPoliciesPath, "policyId"},
		{ratePoliciesPath, "policyId"},
		{accessBindingsPath, "bindingId"},
		{rateBindingsPath, "bindingId"},
	} {
		contracts = append(contracts,
			policyHTTPContract{managementapi.MethodGET, resource.base},
			policyHTTPContract{managementapi.MethodPOST, resource.base},
			policyHTTPContract{managementapi.MethodGET, resource.base + "/{" + resource.parameter + "}"},
			policyHTTPContract{managementapi.MethodPATCH, resource.base + "/{" + resource.parameter + "}"},
			policyHTTPContract{managementapi.MethodDELETE, resource.base + "/{" + resource.parameter + "}"})
	}
	return append(contracts,
		policyHTTPContract{managementapi.MethodPOST, accessBindingBulkPath},
		policyHTTPContract{managementapi.MethodPOST, rateBindingBulkPath})
}

func policyOperationKey(method managementapi.HTTPMethod, path string) string {
	return string(method) + " " + path
}

var _ RouteRegistrar = (*PolicyRoutes)(nil)
