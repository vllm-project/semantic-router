package managementserver

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"regexp"
	"strconv"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/namespacemanagement"
)

const namespacesPath = managementapi.BasePath + "/namespaces"

var namespaceETagPattern = regexp.MustCompile(`^"(namespace|self-service-policy|management-security-policy|routing-claim-schema):([1-9][0-9]*)"$`)

type NamespaceRoutes struct {
	service       NamespaceManagementService
	sessions      SessionAuthenticator
	authorization Authorizer
	scopes        NamespaceResultScopeResolver
	now           func() time.Time
	operations    map[string]managementapi.OperationContract
}

func NewNamespaceRoutes(options NamespaceRoutesOptions) (*NamespaceRoutes, error) {
	if options.Service == nil || options.Sessions == nil || options.Authorization == nil || options.Scopes == nil {
		return nil, errors.New("namespace Management routes require service, session, authorization, and result-scope dependencies")
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	routes := &NamespaceRoutes{
		service: options.Service, sessions: options.Sessions, authorization: options.Authorization,
		scopes: options.Scopes, now: now, operations: make(map[string]managementapi.OperationContract),
	}
	for _, contract := range namespaceHTTPContracts() {
		operation, found := managementapi.LookupOperation(contract.method, contract.path)
		if !found {
			return nil, fmt.Errorf("namespace Management operation %s %s is unavailable", contract.method, contract.path)
		}
		routes.operations[string(contract.method)+" "+contract.path] = operation
	}
	return routes, nil
}

func (routes *NamespaceRoutes) Register(mux *http.ServeMux) {
	if routes == nil || mux == nil {
		panic("Namespace Management routes and mux are required")
	}
	mux.HandleFunc("GET "+namespacesPath, routes.list)
	mux.HandleFunc("POST "+namespacesPath, routes.create)
	mux.HandleFunc("GET "+namespacesPath+"/{namespaceId}", routes.namespace)
	mux.HandleFunc("PATCH "+namespacesPath+"/{namespaceId}", routes.namespace)
	mux.HandleFunc("DELETE "+namespacesPath+"/{namespaceId}", routes.namespace)
	mux.HandleFunc("GET "+namespacesPath+"/{namespaceId}/self-service-policy", routes.selfService)
	mux.HandleFunc("PATCH "+namespacesPath+"/{namespaceId}/self-service-policy", routes.selfService)
	mux.HandleFunc("GET "+namespacesPath+"/{namespaceId}/management-security-policy", routes.security)
	mux.HandleFunc("PATCH "+namespacesPath+"/{namespaceId}/management-security-policy", routes.security)
	mux.HandleFunc("GET "+namespacesPath+"/{namespaceId}/routing-claim-schema", routes.routingClaims)
	mux.HandleFunc("PATCH "+namespacesPath+"/{namespaceId}/routing-claim-schema", routes.routingClaims)
}

func (routes *NamespaceRoutes) Ready(ctx context.Context) error {
	if routes == nil || routes.service == nil {
		return namespacemanagement.ErrUnavailable
	}
	return routes.service.Ready(ctx)
}

func (routes *NamespaceRoutes) list(response http.ResponseWriter, request *http.Request) {
	requestID := namespaceRequest(response, request)
	query, pageSize, ok := subjectListQuery(response, request, requestID, map[string]bool{"": true, "active": true, "disabled": true})
	if !ok {
		return
	}
	session, ok := routes.authenticate(response, request, requestID, "")
	if !ok {
		return
	}
	scope, err := routes.scopes.ResolveNamespaceResultScope(request.Context(), session.Session.PrincipalID)
	if err != nil {
		writeNamespaceAuthorizationError(response, err, requestID, false)
		return
	}
	page, err := routes.service.ListNamespaces(request.Context(), namespacemanagement.ListRequest{
		Scope:  scope,
		Status: query.Get("status"), Cursor: query.Get("cursor"), PageSize: pageSize,
	})
	if err != nil {
		writeNamespaceError(response, err, requestID)
		return
	}
	writeProviderJSON(response, http.StatusOK, namespacePageDTO(page), requestID)
}

func (routes *NamespaceRoutes) create(response http.ResponseWriter, request *http.Request) {
	requestID := namespaceRequest(response, request)
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Namespace create does not accept query parameters.", requestID)
		return
	}
	session, ok := routes.authenticate(response, request, requestID, "")
	if !ok || !routes.authorize(response, request, requestID, session, "", routes.operation(managementapi.MethodPOST, namespacesPath), nil, nil, false) {
		return
	}
	key, ok := requireIdempotencyKey(response, request, requestID)
	if !ok {
		return
	}
	var body managementapi.NamespaceCreateRequest
	if !decodeSubjectBody(response, request, requestID, &body) {
		return
	}
	result, err := routes.service.CreateNamespace(request.Context(), namespacemanagement.CreateNamespaceRequest{
		Name: body.Name, BillingCurrency: body.BillingCurrency, IdempotencyKey: string(key), Actor: namespaceActor(request, session, requestID, body.Reason),
	})
	if err != nil {
		writeNamespaceError(response, err, requestID)
		return
	}
	response.Header().Set("Location", namespacesPath+"/"+result.ID)
	writeNamespaceMutation(response, result, requestID)
}

func (routes *NamespaceRoutes) namespace(response http.ResponseWriter, request *http.Request) {
	requestID := namespaceRequest(response, request)
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Namespace resource does not accept query parameters.", requestID)
		return
	}
	namespaceID := request.PathValue("namespaceId")
	if !canonicalUUID(namespaceID) {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	if request.Method == http.MethodDelete {
		routes.delete(response, request, requestID, namespaceID)
		return
	}
	session, ok := routes.authenticate(response, request, requestID, namespaceID)
	if !ok {
		return
	}
	method := managementapi.HTTPMethod(request.Method)
	if !routes.authorize(response, request, requestID, session, namespaceID,
		routes.operation(method, namespacesPath+"/{namespaceId}"), namespaceTargets(namespaceID, "target"), nil, true) {
		return
	}
	switch request.Method {
	case http.MethodGet:
		value, err := routes.service.GetNamespace(request.Context(), namespaceID)
		if err != nil {
			writeNamespaceError(response, err, requestID)
			return
		}
		response.Header().Set(managementapi.HeaderETag, namespaceETag("namespace", value.Revision))
		writeProviderJSON(response, http.StatusOK, managementapi.NamespaceDetail{Data: namespaceDTO(value)}, requestID)
	case http.MethodPatch:
		revision, ok := requireNamespaceRevision(response, request, requestID, "namespace")
		if !ok {
			return
		}
		var body managementapi.NamespacePatchRequest
		if !decodeSubjectBody(response, request, requestID, &body) {
			return
		}
		result, err := routes.service.PatchNamespace(request.Context(), namespacemanagement.PatchNamespaceRequest{
			NamespaceID: namespaceID, ExpectedRevision: revision, Status: accesscontrol.NamespaceStatus(body.Status), Actor: namespaceActor(request, session, requestID, body.Reason),
		})
		if err != nil {
			writeNamespaceError(response, err, requestID)
			return
		}
		writeNamespaceMutation(response, result, requestID)
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
}

func (routes *NamespaceRoutes) delete(response http.ResponseWriter, request *http.Request, requestID, namespaceID string) {
	if !subjectDeleteRequest(response, request, requestID) {
		return
	}
	session, ok := routes.authenticate(response, request, requestID, "")
	if !ok {
		return
	}
	if !routes.authorize(response, request, requestID, session, "", routes.operation(managementapi.MethodDELETE, namespacesPath+"/{namespaceId}"), nil, nil, true) {
		return
	}
	revision, ok := requireNamespaceRevision(response, request, requestID, "namespace")
	if !ok {
		return
	}
	result, err := routes.service.DeleteNamespace(request.Context(), namespacemanagement.DeleteNamespaceRequest{
		NamespaceID:      namespaceID,
		ExpectedRevision: revision,
		Actor:            namespaceActor(request, session, requestID, "Delete Namespace"),
	})
	if err != nil {
		writeNamespaceError(response, err, requestID)
		return
	}
	writeNamespaceMutation(response, result, requestID)
}

func (routes *NamespaceRoutes) selfService(response http.ResponseWriter, request *http.Request) {
	requestID := namespaceRequest(response, request)
	namespaceID := request.PathValue("namespaceId")
	if request.URL.RawQuery != "" || !canonicalUUID(namespaceID) {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	session, ok := routes.authenticate(response, request, requestID, namespaceID)
	if !ok {
		return
	}
	path := namespacesPath + "/{namespaceId}/self-service-policy"
	if request.Method == http.MethodGet {
		if !routes.authorize(response, request, requestID, session, namespaceID, routes.operation(managementapi.MethodGET, path), nil, nil, true) {
			return
		}
		value, err := routes.service.GetSelfServicePolicy(request.Context(), namespaceID)
		if err != nil {
			writeNamespaceError(response, err, requestID)
			return
		}
		response.Header().Set(managementapi.HeaderETag, namespaceETag("self-service-policy", value.Revision))
		writeProviderJSON(response, http.StatusOK, managementapi.SelfServicePolicyDetail{Data: selfServicePolicyDTO(value)}, requestID)
		return
	}
	current, err := routes.service.GetSelfServicePolicy(request.Context(), namespaceID)
	if err != nil {
		writeNamespaceError(response, err, requestID)
		return
	}
	revision, ok := requireNamespaceRevision(response, request, requestID, "self-service-policy")
	if !ok {
		return
	}
	var body managementapi.SelfServicePolicyPatchRequest
	if !decodeSubjectBody(response, request, requestID, &body) {
		return
	}
	targets, conditions := selfServiceAuthorization(namespaceID, current, body)
	if !routes.authorize(response, request, requestID, session, namespaceID, routes.operation(managementapi.MethodPATCH, path), targets, conditions, true) {
		return
	}
	var capabilities *[]accesscontrol.TeamAdminCapability
	if body.TeamAdminCapabilities != nil {
		value := capabilitiesFromDTO(*body.TeamAdminCapabilities)
		capabilities = &value
	}
	result, err := routes.service.PatchSelfServicePolicy(request.Context(), namespacemanagement.PatchSelfServicePolicyRequest{
		NamespaceID: namespaceID, ExpectedRevision: revision, MaxKeysPerUser: body.MaxKeysPerUser,
		MaxDelegatedSessions: body.MaxDelegatedSessions, DelegatedSessionTTLSeconds: body.DelegatedSessionTTLSeconds,
		AllowTeamKeyDelegation: body.AllowTeamKeyDelegation, AutomaticFirstKey: body.AutomaticFirstKey,
		TeamAdminCapabilities: capabilities, DefaultAccessPolicyID: body.DefaultAccessPolicyID,
		DefaultRateLimitPolicyID: body.DefaultRateLimitPolicyID, Actor: namespaceActor(request, session, requestID, body.Reason),
	})
	if err != nil {
		writeNamespaceError(response, err, requestID)
		return
	}
	writeNamespaceMutation(response, result, requestID)
}

func (routes *NamespaceRoutes) security(response http.ResponseWriter, request *http.Request) {
	requestID := namespaceRequest(response, request)
	namespaceID := request.PathValue("namespaceId")
	if request.URL.RawQuery != "" || !canonicalUUID(namespaceID) {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	session, ok := routes.authenticate(response, request, requestID, namespaceID)
	if !ok {
		return
	}
	path := namespacesPath + "/{namespaceId}/management-security-policy"
	method := managementapi.HTTPMethod(request.Method)
	if !routes.authorize(response, request, requestID, session, namespaceID, routes.operation(method, path), nil, nil, true) {
		return
	}
	if request.Method == http.MethodGet {
		value, err := routes.service.GetManagementSecurityPolicy(request.Context(), namespaceID)
		if err != nil {
			writeNamespaceError(response, err, requestID)
			return
		}
		response.Header().Set(managementapi.HeaderETag, namespaceETag("management-security-policy", value.Revision))
		writeProviderJSON(response, http.StatusOK, managementapi.NamespaceManagementSecurityPolicyDetail{Data: securityPolicyDTO(value)}, requestID)
		return
	}
	revision, ok := requireNamespaceRevision(response, request, requestID, "management-security-policy")
	if !ok {
		return
	}
	var body managementapi.NamespaceManagementSecurityPolicyPatchRequest
	if !decodeSubjectBody(response, request, requestID, &body) {
		return
	}
	result, err := routes.service.PatchManagementSecurityPolicy(request.Context(), namespacemanagement.PatchManagementSecurityPolicyRequest{
		NamespaceID: namespaceID, ExpectedRevision: revision, ActionRequirements: requirementsFromDTO(body.ActionRequirements),
		Session: session.Session, Actor: namespaceActor(request, session, requestID, body.Reason),
	})
	if err != nil {
		writeNamespaceError(response, err, requestID)
		return
	}
	writeNamespaceMutation(response, result, requestID)
}

func (routes *NamespaceRoutes) routingClaims(response http.ResponseWriter, request *http.Request) {
	requestID := namespaceRequest(response, request)
	namespaceID := request.PathValue("namespaceId")
	if request.URL.RawQuery != "" || !canonicalUUID(namespaceID) {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	session, ok := routes.authenticate(response, request, requestID, namespaceID)
	if !ok {
		return
	}
	path := namespacesPath + "/{namespaceId}/routing-claim-schema"
	method := managementapi.HTTPMethod(request.Method)
	if !routes.authorize(response, request, requestID, session, namespaceID, routes.operation(method, path), nil, nil, true) {
		return
	}
	if request.Method == http.MethodGet {
		value, err := routes.service.GetRoutingClaimSchema(request.Context(), namespaceID)
		if err != nil {
			writeNamespaceError(response, err, requestID)
			return
		}
		response.Header().Set(managementapi.HeaderETag, namespaceETag("routing-claim-schema", value.Revision))
		writeProviderJSON(response, http.StatusOK, managementapi.RoutingClaimSchemaDetail{Data: routingClaimSchemaDTO(value)}, requestID)
		return
	}
	revision, ok := requireNamespaceRevision(response, request, requestID, "routing-claim-schema")
	if !ok {
		return
	}
	var body managementapi.RoutingClaimSchemaPatchRequest
	if !decodeSubjectBody(response, request, requestID, &body) {
		return
	}
	result, err := routes.service.PatchRoutingClaimSchema(request.Context(), namespacemanagement.PatchRoutingClaimSchemaRequest{
		NamespaceID: namespaceID, ExpectedRevision: revision, Definitions: routingClaimDefinitionsFromDTO(body.Definitions),
		Actor: namespaceActor(request, session, requestID, body.Reason),
	})
	if err != nil {
		writeNamespaceError(response, err, requestID)
		return
	}
	writeNamespaceMutation(response, result, requestID)
}

func (routes *NamespaceRoutes) authenticate(response http.ResponseWriter, request *http.Request, requestID, namespaceID string) (managementauth.AuthenticatedSession, bool) {
	token, ok := bearerToken(request)
	if !ok {
		writeProviderError(response, http.StatusUnauthorized, "unauthenticated", "Authentication is required.", requestID)
		return managementauth.AuthenticatedSession{}, false
	}
	session, err := routes.sessions.Authenticate(request.Context(), token, namespaceID, routes.now().UTC())
	if err != nil {
		writeAuthenticationError(response, err, requestID)
		return managementauth.AuthenticatedSession{}, false
	}
	if session.NamespaceID != namespaceID || !canonicalUUID(session.Session.PrincipalID) {
		writeProviderError(response, http.StatusServiceUnavailable, "authentication_unavailable", "Authentication state is unavailable.", requestID)
		return managementauth.AuthenticatedSession{}, false
	}
	return session, true
}

func (routes *NamespaceRoutes) authorize(response http.ResponseWriter, request *http.Request, requestID string, session managementauth.AuthenticatedSession, namespaceID string, operation managementapi.OperationContract, targets map[string][]accesscontrol.ScopedTarget, conditions map[string]bool, nondisclosing bool) bool {
	_, err := routes.authorization.Authorize(request.Context(), AuthorizationRequest{Operation: operation, Session: session, NamespaceID: namespaceID, Targets: targets, Conditions: conditions})
	if err == nil {
		return true
	}
	writeNamespaceAuthorizationError(response, err, requestID, nondisclosing)
	return false
}

func (routes *NamespaceRoutes) operation(method managementapi.HTTPMethod, path string) managementapi.OperationContract {
	return routes.operations[string(method)+" "+path]
}

func namespaceRequest(response http.ResponseWriter, request *http.Request) string {
	requestID := managementRequestID(request)
	setProviderResponseHeaders(response, requestID)
	return requestID
}

func namespaceActor(request *http.Request, session managementauth.AuthenticatedSession, requestID, reason string) namespacemanagement.Actor {
	return namespacemanagement.Actor{PrincipalID: session.Session.PrincipalID, ActorChain: []string{session.Session.PrincipalID}, RequestID: requestID, SourceIP: directRequestIP(request), Reason: reason}
}

func namespaceTargets(namespaceID string, operands ...string) map[string][]accesscontrol.ScopedTarget {
	result := make(map[string][]accesscontrol.ScopedTarget, len(operands))
	target := accesscontrol.ScopedTarget{Scope: accesscontrol.NamespaceScope(accesscontrol.NamespaceID(namespaceID))}
	for _, operand := range operands {
		result[operand] = []accesscontrol.ScopedTarget{target}
	}
	return result
}

func selfServiceAuthorization(namespaceID string, current namespacemanagement.SelfServicePolicy, body managementapi.SelfServicePolicyPatchRequest) (map[string][]accesscontrol.ScopedTarget, map[string]bool) {
	targets := map[string][]accesscontrol.ScopedTarget{}
	conditions := map[string]bool{}
	add := func(condition, operand, id string, kind accesscontrol.ScopeResourceType) {
		if id == "" {
			return
		}
		conditions[condition] = true
		targets[operand] = []accesscontrol.ScopedTarget{{Scope: accesscontrol.ResourceScope(accesscontrol.NamespaceID(namespaceID), kind, accesscontrol.ResourceID(id))}}
	}
	add("current_access_policy_default_present", "current_access_policy_default", current.DefaultAccessPolicyID, accesscontrol.ScopeResourceAccessPolicy)
	add("current_rate_policy_default_present", "current_rate_policy_default", current.DefaultRateLimitPolicyID, accesscontrol.ScopeResourceRateLimitPolicy)
	accessTarget, rateTarget := current.DefaultAccessPolicyID, current.DefaultRateLimitPolicyID
	if body.DefaultAccessPolicyID != nil {
		accessTarget = *body.DefaultAccessPolicyID
	}
	if body.DefaultRateLimitPolicyID != nil {
		rateTarget = *body.DefaultRateLimitPolicyID
	}
	add("target_access_policy_default_present", "target_access_policy_default", accessTarget, accesscontrol.ScopeResourceAccessPolicy)
	add("target_rate_policy_default_present", "target_rate_policy_default", rateTarget, accesscontrol.ScopeResourceRateLimitPolicy)
	return targets, conditions
}

func requireNamespaceRevision(response http.ResponseWriter, request *http.Request, requestID, kind string) (uint64, bool) {
	values := request.Header.Values(managementapi.HeaderIfMatch)
	if len(values) != 1 {
		writeProviderError(response, http.StatusPreconditionRequired, "precondition_required", "If-Match is required.", requestID)
		return 0, false
	}
	match := namespaceETagPattern.FindStringSubmatch(values[0])
	if len(match) != 3 || match[1] != kind {
		writeProviderError(response, http.StatusBadRequest, "invalid_precondition", "If-Match is invalid.", requestID)
		return 0, false
	}
	revision, err := strconv.ParseUint(match[2], 10, 64)
	if err != nil || revision == 0 {
		writeProviderError(response, http.StatusBadRequest, "invalid_precondition", "If-Match is invalid.", requestID)
		return 0, false
	}
	return revision, true
}

func namespaceETag(kind string, revision uint64) string {
	return `"` + kind + `:` + strconv.FormatUint(revision, 10) + `"`
}

func writeNamespaceMutation(response http.ResponseWriter, result namespacemanagement.MutationResult, requestID string) {
	response.Header().Set(managementapi.HeaderETag, namespaceETag(namespaceETagKind(result.Kind), result.Revision))
	setIdempotencyReplayHeader(response, result.Replayed)
	if result.HTTPStatus == http.StatusNoContent {
		response.WriteHeader(http.StatusNoContent)
		return
	}
	var replayed *bool
	if result.Kind == "namespace" && result.HTTPStatus == http.StatusCreated {
		value := result.Replayed
		replayed = &value
	}
	writeProviderJSON(response, result.HTTPStatus, managementapi.NewResourceMutationReceipt(result.Kind, result.ID, result.Revision, replayed), requestID)
}

func namespaceETagKind(kind string) string {
	switch kind {
	case "self_service_policy":
		return "self-service-policy"
	case "management_security_policy":
		return "management-security-policy"
	case "routing_claim_schema":
		return "routing-claim-schema"
	default:
		return "namespace"
	}
}

func writeNamespaceAuthorizationError(response http.ResponseWriter, err error, requestID string, nondisclosing bool) {
	if errors.Is(err, managementauthorization.ErrDenied) {
		if nondisclosing {
			writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		} else {
			writeProviderError(response, http.StatusForbidden, "forbidden", "Permission denied.", requestID)
		}
		return
	}
	writeProviderError(response, http.StatusServiceUnavailable, "authorization_unavailable", "Authorization state is unavailable.", requestID)
}

func writeNamespaceError(response http.ResponseWriter, err error, requestID string) {
	switch {
	case errors.Is(err, namespacemanagement.ErrInvalidRequest):
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Namespace request is invalid.", requestID)
	case errors.Is(err, namespacemanagement.ErrNotFound):
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	case errors.Is(err, namespacemanagement.ErrAlreadyExists):
		writeProviderError(response, http.StatusConflict, "already_exists", "Namespace already exists.", requestID)
	case errors.Is(err, namespacemanagement.ErrIdempotencyConflict):
		writeProviderError(response, http.StatusConflict, "idempotency_conflict", "Idempotency-Key was already used for a different request.", requestID)
	case errors.Is(err, namespacemanagement.ErrRevisionConflict):
		writeProviderError(response, http.StatusPreconditionFailed, "revision_conflict", "The resource changed. Refresh and retry.", requestID)
	case errors.Is(err, namespacemanagement.ErrDependency):
		writeProviderError(response, http.StatusConflict, "active_dependencies", "Disable active Namespace resources before deletion.", requestID)
	case errors.Is(err, namespacemanagement.ErrAssurance):
		writeProviderError(response, http.StatusForbidden, "authentication_requirement_not_met", "Current authentication assurance cannot loosen this policy.", requestID)
	default:
		writeProviderError(response, http.StatusServiceUnavailable, "namespace_management_unavailable", "Namespace Management is unavailable.", requestID)
	}
}

type namespaceHTTPContract struct {
	method managementapi.HTTPMethod
	path   string
}

func namespaceHTTPContracts() []namespaceHTTPContract {
	return []namespaceHTTPContract{
		{managementapi.MethodGET, namespacesPath},
		{managementapi.MethodPOST, namespacesPath},
		{managementapi.MethodGET, namespacesPath + "/{namespaceId}"},
		{managementapi.MethodPATCH, namespacesPath + "/{namespaceId}"},
		{managementapi.MethodDELETE, namespacesPath + "/{namespaceId}"},
		{managementapi.MethodGET, namespacesPath + "/{namespaceId}/self-service-policy"},
		{managementapi.MethodPATCH, namespacesPath + "/{namespaceId}/self-service-policy"},
		{managementapi.MethodGET, namespacesPath + "/{namespaceId}/management-security-policy"},
		{managementapi.MethodPATCH, namespacesPath + "/{namespaceId}/management-security-policy"},
		{managementapi.MethodGET, namespacesPath + "/{namespaceId}/routing-claim-schema"},
		{managementapi.MethodPATCH, namespacesPath + "/{namespaceId}/routing-claim-schema"},
	}
}

var _ RouteRegistrar = (*NamespaceRoutes)(nil)
