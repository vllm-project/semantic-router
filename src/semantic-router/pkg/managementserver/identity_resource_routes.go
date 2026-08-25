package managementserver

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity"
)

const (
	principalPath      = managementapi.BasePath + "/management-principals"
	rolePath           = managementapi.BasePath + "/management-roles"
	bindingPath        = managementapi.BasePath + "/role-bindings"
	policyPath         = managementapi.BasePath + "/management-session-policy"
	identityCommandTTL = 24 * time.Hour
)

type IdentityResourceLifecycle interface {
	Ready(context.Context) error
}

type PrincipalResourceService interface {
	GetPrincipal(context.Context, string) (managementidentity.Principal, error)
	ListPrincipals(context.Context, managementidentity.ListRequest) (managementidentity.PrincipalPage, error)
	CreatePrincipal(context.Context, managementidentity.CreatePrincipal) (managementidentity.MutationResult, error)
	UpdatePrincipal(context.Context, managementidentity.UpdatePrincipal) (managementidentity.MutationResult, error)
	DeletePrincipal(context.Context, string, uint64, managementidentity.MutationActor) (managementidentity.MutationResult, error)
}

type RoleResourceService interface {
	GetRole(context.Context, string) (managementidentity.Role, error)
	ListRoles(context.Context, string, managementidentity.ListRequest) (managementidentity.RolePage, error)
	CreateRole(context.Context, managementidentity.CreateRole) (managementidentity.MutationResult, error)
	UpdateRole(context.Context, managementidentity.UpdateRole) (managementidentity.MutationResult, error)
	DeleteRole(context.Context, string, uint64, managementidentity.MutationActor) (managementidentity.MutationResult, error)
}

type RoleBindingResourceService interface {
	GetRoleBinding(context.Context, string) (managementidentity.RoleBinding, error)
	ListRoleBindings(context.Context, string, managementidentity.ListRequest) (managementidentity.RoleBindingPage, error)
	CreateRoleBinding(context.Context, managementidentity.CreateRoleBinding) (managementidentity.MutationResult, error)
	UpdateRoleBinding(context.Context, managementidentity.UpdateRoleBinding) (managementidentity.MutationResult, error)
	DeleteRoleBinding(context.Context, string, uint64, managementidentity.MutationActor) (managementidentity.MutationResult, error)
}

type PrincipalDirectoryService interface {
	GetPrincipalUserLink(context.Context, string, string) (managementidentity.PrincipalUserLink, error)
	GetPrincipalDirectoryEntry(context.Context, string, string) (managementidentity.PrincipalDirectoryEntry, error)
	ListPrincipalDirectory(context.Context, managementidentity.PrincipalDirectoryRequest) (managementidentity.PrincipalDirectoryPage, error)
	ListPrincipalUserLinks(context.Context, managementidentity.PrincipalUserLinkListRequest) (managementidentity.PrincipalUserLinkPage, error)
	ListPrincipalLinks(context.Context, string, managementidentity.ListRequest) (managementidentity.PrincipalUserLinkPage, error)
}

type PrincipalLinkMutationService interface {
	PutPrincipalUserLink(context.Context, managementidentity.LinkMutation) (managementidentity.MutationResult, error)
	DeletePrincipalUserLink(context.Context, managementidentity.LinkMutation) (managementidentity.MutationResult, error)
}

type ManagementSessionPolicyService interface {
	LoadSessionPolicy(context.Context) (managementauth.SessionPolicy, error)
	UpdateSessionPolicy(context.Context, managementauth.SessionPolicy, uint64, managementidentity.MutationActor) (managementidentity.MutationResult, error)
}

type IdentityCoreResourceService interface {
	PrincipalResourceService
	RoleResourceService
	RoleBindingResourceService
}

type IdentityDirectoryResourceService interface {
	PrincipalDirectoryService
	PrincipalLinkMutationService
	ManagementSessionPolicyService
}

type IdentityResourceService interface {
	IdentityResourceLifecycle
	IdentityCoreResourceService
	IdentityDirectoryResourceService
}

type IdentityResourceRoutesOptions struct {
	Service       IdentityResourceService
	Sessions      SessionAuthenticator
	Authorization Authorizer
	Commands      *managementcommand.Codec
	Now           func() time.Time
}

type IdentityResourceRoutes struct {
	service       IdentityResourceService
	sessions      SessionAuthenticator
	authorization Authorizer
	commands      *managementcommand.Codec
	now           func() time.Time
	operations    map[string]managementapi.OperationContract
}

func NewIdentityResourceRoutes(options IdentityResourceRoutesOptions) (*IdentityResourceRoutes, error) {
	if options.Service == nil || options.Sessions == nil || options.Authorization == nil || options.Commands == nil {
		return nil, errors.New("management identity routes require service, session, authorization, and command dependencies")
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	routes := &IdentityResourceRoutes{service: options.Service, sessions: options.Sessions, authorization: options.Authorization, commands: options.Commands, now: now, operations: map[string]managementapi.OperationContract{}}
	for _, operation := range managementapi.Operations() {
		if strings.HasPrefix(operation.Path, principalPath) || strings.HasPrefix(operation.Path, rolePath) ||
			strings.HasPrefix(operation.Path, bindingPath) || operation.Path == policyPath ||
			isIdentityDirectoryOperation(operation.Path) {
			routes.operations[string(operation.Method)+" "+operation.Path] = operation
		}
	}
	return routes, nil
}

func (routes *IdentityResourceRoutes) Register(mux *http.ServeMux) {
	if routes == nil || mux == nil {
		panic("Management identity routes and mux are required")
	}
	for _, path := range []string{principalPath, rolePath, bindingPath} {
		mux.Handle("GET "+path, routes)
		mux.Handle("POST "+path, routes)
		mux.Handle("GET "+path+"/", routes)
		mux.Handle("PATCH "+path+"/", routes)
		mux.Handle("DELETE "+path+"/", routes)
	}
	registerIdentityDirectoryRoutes(mux, routes)
	mux.Handle("GET "+policyPath, routes)
	mux.Handle("PATCH "+policyPath, routes)
}

func (routes *IdentityResourceRoutes) Ready(ctx context.Context) error {
	return routes.service.Ready(ctx)
}

func (routes *IdentityResourceRoutes) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	requestID := managementRequestID(request)
	setProviderResponseHeaders(response, requestID)
	if routes == nil || request == nil || request.URL == nil || request.URL.EscapedPath() != request.URL.Path {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	if request.Method != http.MethodGet && request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Query parameters are not accepted.", requestID)
		return
	}
	if routes.serveIdentityDirectory(response, request, requestID) {
		return
	}
	switch {
	case request.URL.Path == principalPath:
		routes.principals(response, request, requestID)
	case strings.HasPrefix(request.URL.Path, principalPath+"/"):
		routes.principal(response, request, requestID)
	case request.URL.Path == rolePath:
		routes.roles(response, request, requestID)
	case strings.HasPrefix(request.URL.Path, rolePath+"/"):
		routes.role(response, request, requestID)
	case request.URL.Path == bindingPath:
		routes.bindings(response, request, requestID)
	case strings.HasPrefix(request.URL.Path, bindingPath+"/"):
		routes.binding(response, request, requestID)
	case request.URL.Path == policyPath:
		routes.policy(response, request, requestID)
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
}

func (routes *IdentityResourceRoutes) principals(response http.ResponseWriter, request *http.Request, requestID string) {
	session, ok := routes.authenticate(response, request, requestID, "")
	if !ok || !routes.authorize(response, request, requestID, session, "", routes.operation(request.Method, principalPath), nil) {
		return
	}
	switch request.Method {
	case http.MethodGet:
		pageRequest, ok := identityPageRequest(response, request, requestID)
		if !ok {
			return
		}
		page, err := routes.service.ListPrincipals(request.Context(), pageRequest)
		if err != nil {
			writeIdentityError(response, err, requestID)
			return
		}
		data := make([]managementapi.ManagementPrincipal, len(page.Items))
		for index := range page.Items {
			data[index] = principalDTO(page.Items[index])
		}
		writeProviderJSON(response, http.StatusOK, managementapi.Page[managementapi.ManagementPrincipal]{Data: data, Page: identityPageInfo(page.NextCursor, pageRequest.Limit)}, requestID)
	case http.MethodPost:
		key, ok := requireIdempotencyKey(response, request, requestID)
		if !ok {
			return
		}
		var body managementapi.ManagementPrincipalCreateRequest
		if !decodeIdentityBody(response, request, requestID, &body) {
			return
		}
		canonical, _ := json.Marshal(body)
		now := routes.now().UTC()
		command, err := routes.commands.Bind(managementcommand.ClusterCommandScope(), session.Session.PrincipalID, principalPath, string(key), canonical, now, now.Add(identityCommandTTL))
		if err != nil {
			writeProviderError(response, http.StatusBadRequest, "invalid_request", "Request is invalid.", requestID)
			return
		}
		id := uuid.NewString()
		result, err := routes.service.CreatePrincipal(request.Context(), managementidentity.CreatePrincipal{
			ID: id, Issuer: body.Issuer, Subject: body.Subject, DisplayName: body.DisplayName,
			VerifiedEmail: body.VerifiedEmail, Attributes: body.Attributes, Command: command, Actor: identityActor(request, session, requestID, "Create Management principal"),
		})
		if err != nil {
			writeIdentityError(response, err, requestID)
			return
		}
		writeIdentityMutation(response, result, principalETag, principalPath, requestID)
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
}

func (routes *IdentityResourceRoutes) principal(response http.ResponseWriter, request *http.Request, requestID string) {
	id, ok := identityPathID(request.URL.Path, principalPath)
	if !ok || request.URL.RawQuery != "" {
		writeProviderError(response, 404, "not_found", "Resource not found.", requestID)
		return
	}
	session, ok := routes.authenticate(response, request, requestID, "")
	if !ok || !routes.authorize(response, request, requestID, session, "", routes.operation(request.Method, principalPath+"/{principalId}"), nil) {
		return
	}
	switch request.Method {
	case http.MethodGet:
		value, err := routes.service.GetPrincipal(request.Context(), id)
		if err != nil {
			writeIdentityError(response, err, requestID)
			return
		}
		response.Header().Set(managementapi.HeaderETag, principalETag(uint64(value.Revision)))
		writeProviderJSON(response, 200, map[string]any{"data": principalDTO(value)}, requestID)
	case http.MethodPatch:
		revision, ok := identityRevision(response, request, requestID, "mp")
		if !ok {
			return
		}
		var body managementapi.ManagementPrincipalPatchRequest
		if !decodeIdentityBody(response, request, requestID, &body) {
			return
		}
		var status *accesscontrol.PrincipalStatus
		if body.Status != nil {
			value := accesscontrol.PrincipalStatus(*body.Status)
			status = &value
		}
		result, err := routes.service.UpdatePrincipal(request.Context(), managementidentity.UpdatePrincipal{ID: id, ExpectedRevision: revision, DisplayName: body.DisplayName, VerifiedEmail: body.VerifiedEmail, Status: status, Actor: identityActor(request, session, requestID, body.Reason)})
		if err != nil {
			writeIdentityError(response, err, requestID)
			return
		}
		writeIdentityMutation(response, result, principalETag, principalPath, requestID)
	case http.MethodDelete:
		revision, ok := identityRevision(response, request, requestID, "mp")
		if !ok {
			return
		}
		result, err := routes.service.DeletePrincipal(request.Context(), id, revision, identityActor(request, session, requestID, "Delete Management principal"))
		if err != nil {
			writeIdentityError(response, err, requestID)
			return
		}
		response.Header().Set(managementapi.HeaderETag, principalETag(result.Revision))
		response.WriteHeader(http.StatusNoContent)
	default:
		writeProviderError(response, 404, "not_found", "Resource not found.", requestID)
	}
}

func (routes *IdentityResourceRoutes) roles(response http.ResponseWriter, request *http.Request, requestID string) {
	switch request.Method {
	case http.MethodGet:
		namespaceID := request.URL.Query().Get("namespaceId")
		if namespaceID != "" && !canonicalUUID(namespaceID) {
			writeProviderError(response, 400, "invalid_request", "namespaceId is invalid.", requestID)
			return
		}
		session, ok := routes.authenticate(response, request, requestID, namespaceID)
		if !ok || !routes.authorize(response, request, requestID, session, namespaceID, routes.operation(request.Method, rolePath), identityScopeTarget(namespaceID)) {
			return
		}
		pageRequest, ok := identityPageRequest(response, request, requestID, "namespaceId")
		if !ok {
			return
		}
		page, err := routes.service.ListRoles(request.Context(), namespaceID, pageRequest)
		if err != nil {
			writeIdentityError(response, err, requestID)
			return
		}
		data := make([]managementapi.ManagementRole, len(page.Items))
		for index := range page.Items {
			data[index] = roleDTO(page.Items[index])
		}
		writeProviderJSON(response, 200, managementapi.Page[managementapi.ManagementRole]{Data: data, Page: identityPageInfo(page.NextCursor, pageRequest.Limit)}, requestID)
	case http.MethodPost:
		key, ok := requireIdempotencyKey(response, request, requestID)
		if !ok {
			return
		}
		var body managementapi.ManagementRoleCreateRequest
		if !decodeIdentityBody(response, request, requestID, &body) {
			return
		}
		namespaceID := body.NamespaceID
		if !canonicalUUID(namespaceID) {
			writeProviderError(response, 400, "invalid_scope", "Namespace is invalid.", requestID)
			return
		}
		session, ok := routes.authenticate(response, request, requestID, namespaceID)
		if !ok || !routes.authorize(response, request, requestID, session, namespaceID, routes.operation(request.Method, rolePath), identityScopeTarget(namespaceID)) {
			return
		}
		permissions, err := permissions(body.Permissions)
		if err != nil {
			writeProviderError(response, 400, "invalid_request", "Permissions are invalid.", requestID)
			return
		}
		canonical, _ := json.Marshal(body)
		now := routes.now().UTC()
		command, err := routes.commands.Bind(managementcommand.NamespaceCommandScope(namespaceID), session.Session.PrincipalID, rolePath, string(key), canonical, now, now.Add(identityCommandTTL))
		if err != nil {
			writeProviderError(response, 400, "invalid_request", "Request is invalid.", requestID)
			return
		}
		result, err := routes.service.CreateRole(request.Context(), managementidentity.CreateRole{ID: uuid.NewString(), NamespaceID: namespaceID, Name: body.Name, DisplayName: body.DisplayName, Description: body.Description, Permissions: permissions, Command: command, Actor: identityActor(request, session, requestID, "Create Management role")})
		if err != nil {
			writeIdentityError(response, err, requestID)
			return
		}
		writeIdentityMutation(response, result, roleETag, rolePath, requestID)
	}
}

func (routes *IdentityResourceRoutes) role(response http.ResponseWriter, request *http.Request, requestID string) {
	id, ok := identityPathID(request.URL.Path, rolePath)
	if !ok || request.URL.RawQuery != "" {
		writeProviderError(response, 404, "not_found", "Resource not found.", requestID)
		return
	}
	value, err := routes.service.GetRole(request.Context(), id)
	if err != nil {
		writeIdentityError(response, err, requestID)
		return
	}
	namespaceID := string(value.Role.NamespaceID)
	session, ok := routes.authenticate(response, request, requestID, namespaceID)
	if !ok || !routes.authorize(response, request, requestID, session, namespaceID, routes.operation(request.Method, rolePath+"/{roleId}"), identityScopeTarget(namespaceID)) {
		return
	}
	switch request.Method {
	case http.MethodGet:
		response.Header().Set(managementapi.HeaderETag, roleETag(uint64(value.Role.Revision)))
		writeProviderJSON(response, 200, map[string]any{"data": roleDTO(value)}, requestID)
	case http.MethodPatch:
		revision, ok := identityRevision(response, request, requestID, "mr")
		if !ok {
			return
		}
		var body managementapi.ManagementRolePatchRequest
		if !decodeIdentityBody(response, request, requestID, &body) {
			return
		}
		result, err := routes.service.UpdateRole(request.Context(), managementidentity.UpdateRole{ID: id, ExpectedRevision: revision, DisplayName: body.DisplayName, Description: body.Description, Actor: identityActor(request, session, requestID, body.Reason)})
		if err != nil {
			writeIdentityError(response, err, requestID)
			return
		}
		writeIdentityMutation(response, result, roleETag, rolePath, requestID)
	case http.MethodDelete:
		revision, ok := identityRevision(response, request, requestID, "mr")
		if !ok {
			return
		}
		result, err := routes.service.DeleteRole(request.Context(), id, revision, identityActor(request, session, requestID, "Delete Management role"))
		if err != nil {
			writeIdentityError(response, err, requestID)
			return
		}
		response.Header().Set(managementapi.HeaderETag, roleETag(result.Revision))
		response.WriteHeader(204)
	}
}

func (routes *IdentityResourceRoutes) bindings(response http.ResponseWriter, request *http.Request, requestID string) {
	if request.Method == http.MethodGet {
		// Cluster-only unfiltered list; scoped UIs should filter by principal.
		session, ok := routes.authenticate(response, request, requestID, "")
		if !ok || !routes.authorize(response, request, requestID, session, "", routes.operation(request.Method, bindingPath), nil) {
			return
		}
		pageRequest, ok := identityPageRequest(response, request, requestID, "principalId")
		if !ok {
			return
		}
		principalID := request.URL.Query().Get("principalId")
		page, err := routes.service.ListRoleBindings(request.Context(), principalID, pageRequest)
		if err != nil {
			writeIdentityError(response, err, requestID)
			return
		}
		data := make([]managementapi.ManagementRoleBinding, len(page.Items))
		for index := range page.Items {
			data[index] = bindingDTO(page.Items[index])
		}
		writeProviderJSON(response, 200, managementapi.Page[managementapi.ManagementRoleBinding]{Data: data, Page: identityPageInfo(page.NextCursor, pageRequest.Limit)}, requestID)
		return
	}
	key, ok := requireIdempotencyKey(response, request, requestID)
	if !ok {
		return
	}
	var body managementapi.ManagementRoleBindingCreateRequest
	if !decodeIdentityBody(response, request, requestID, &body) {
		return
	}
	scope, err := scopeFromDTO(body.Scope)
	if err != nil {
		writeProviderError(response, 400, "invalid_request", "Scope is invalid.", requestID)
		return
	}
	namespaceID := string(scope.NamespaceID)
	session, ok := routes.authenticate(response, request, requestID, namespaceID)
	if !ok || !routes.authorize(response, request, requestID, session, namespaceID, routes.operation(request.Method, bindingPath), identityScopeTarget(namespaceID)) {
		return
	}
	ceiling, err := permissions(body.DelegationCeiling)
	if err != nil {
		writeProviderError(response, 400, "invalid_request", "Delegation ceiling is invalid.", requestID)
		return
	}
	canonical, _ := json.Marshal(body)
	now := routes.now().UTC()
	commandScope := managementcommand.ClusterCommandScope()
	if namespaceID != "" {
		commandScope = managementcommand.NamespaceCommandScope(namespaceID)
	}
	command, err := routes.commands.Bind(commandScope, session.Session.PrincipalID, bindingPath, string(key), canonical, now, now.Add(identityCommandTTL))
	if err != nil {
		writeProviderError(response, 400, "invalid_request", "Request is invalid.", requestID)
		return
	}
	result, err := routes.service.CreateRoleBinding(request.Context(), managementidentity.CreateRoleBinding{ID: uuid.NewString(), PrincipalID: body.PrincipalID, RoleID: body.RoleID, Scope: scope, DelegationCeiling: ceiling, Command: command, Actor: identityActor(request, session, requestID, "Create Management role binding")})
	if err != nil {
		writeIdentityError(response, err, requestID)
		return
	}
	writeIdentityMutation(response, result, bindingETag, bindingPath, requestID)
}

func (routes *IdentityResourceRoutes) binding(response http.ResponseWriter, request *http.Request, requestID string) {
	id, ok := identityPathID(request.URL.Path, bindingPath)
	if !ok || request.URL.RawQuery != "" {
		writeProviderError(response, 404, "not_found", "Resource not found.", requestID)
		return
	}
	value, err := routes.service.GetRoleBinding(request.Context(), id)
	if err != nil {
		writeIdentityError(response, err, requestID)
		return
	}
	namespaceID := string(value.Binding.Scope.NamespaceID)
	session, ok := routes.authenticate(response, request, requestID, namespaceID)
	if !ok || !routes.authorize(response, request, requestID, session, namespaceID, routes.operation(request.Method, bindingPath+"/{bindingId}"), identityScopeTarget(namespaceID)) {
		return
	}
	switch request.Method {
	case http.MethodGet:
		response.Header().Set(managementapi.HeaderETag, bindingETag(uint64(value.Binding.Revision)))
		writeProviderJSON(response, 200, map[string]any{"data": bindingDTO(value)}, requestID)
	case http.MethodPatch:
		revision, ok := identityRevision(response, request, requestID, "mb")
		if !ok {
			return
		}
		var body managementapi.ManagementRoleBindingPatchRequest
		if !decodeIdentityBody(response, request, requestID, &body) {
			return
		}
		result, err := routes.service.UpdateRoleBinding(request.Context(), managementidentity.UpdateRoleBinding{ID: id, ExpectedRevision: revision, Status: accesscontrol.BindingStatus(body.Status), Actor: identityActor(request, session, requestID, body.Reason)})
		if err != nil {
			writeIdentityError(response, err, requestID)
			return
		}
		writeIdentityMutation(response, result, bindingETag, bindingPath, requestID)
	case http.MethodDelete:
		revision, ok := identityRevision(response, request, requestID, "mb")
		if !ok {
			return
		}
		result, err := routes.service.DeleteRoleBinding(request.Context(), id, revision, identityActor(request, session, requestID, "Delete Management role binding"))
		if err != nil {
			writeIdentityError(response, err, requestID)
			return
		}
		response.Header().Set(managementapi.HeaderETag, bindingETag(result.Revision))
		response.WriteHeader(204)
	}
}

func (routes *IdentityResourceRoutes) policy(response http.ResponseWriter, request *http.Request, requestID string) {
	session, ok := routes.authenticate(response, request, requestID, "")
	if !ok || !routes.authorize(response, request, requestID, session, "", routes.operation(request.Method, policyPath), nil) {
		return
	}
	if request.Method == http.MethodGet {
		policy, err := routes.service.LoadSessionPolicy(request.Context())
		if err != nil {
			writeIdentityError(response, err, requestID)
			return
		}
		response.Header().Set(managementapi.HeaderETag, policyETag(policy.Revision))
		writeProviderJSON(response, 200, map[string]any{"data": policyDTO(policy)}, requestID)
		return
	}
	revision, ok := identityRevision(response, request, requestID, "msp")
	if !ok {
		return
	}
	var body managementapi.ManagementSessionPolicyPatchRequest
	if !decodeIdentityBody(response, request, requestID, &body) {
		return
	}
	policy, err := policyFromDTO(body, revision, routes.now().UTC())
	if err != nil {
		writeProviderError(response, 400, "invalid_request", "Session policy is invalid.", requestID)
		return
	}
	result, err := routes.service.UpdateSessionPolicy(request.Context(), policy, revision, identityActor(request, session, requestID, body.Reason))
	if err != nil {
		writeIdentityError(response, err, requestID)
		return
	}
	writeIdentityMutation(response, result, policyETag, policyPath, requestID)
}

func (routes *IdentityResourceRoutes) authenticate(response http.ResponseWriter, request *http.Request, requestID, namespaceID string) (managementauth.AuthenticatedSession, bool) {
	token, ok := bearerToken(request)
	if !ok {
		writeProviderError(response, 401, "unauthenticated", "Authentication is required.", requestID)
		return managementauth.AuthenticatedSession{}, false
	}
	session, err := routes.sessions.Authenticate(request.Context(), token, namespaceID, routes.now().UTC())
	if err != nil {
		writeAuthenticationError(response, err, requestID)
		return managementauth.AuthenticatedSession{}, false
	}
	if session.NamespaceID != namespaceID {
		writeProviderError(response, 503, "authentication_unavailable", "Authentication state is unavailable.", requestID)
		return managementauth.AuthenticatedSession{}, false
	}
	return session, true
}

func (routes *IdentityResourceRoutes) authorize(response http.ResponseWriter, request *http.Request, requestID string, session managementauth.AuthenticatedSession, namespaceID string, operation managementapi.OperationContract, targets map[string][]accesscontrol.ScopedTarget) bool {
	_, err := routes.authorization.Authorize(request.Context(), AuthorizationRequest{Operation: operation, Session: session, NamespaceID: namespaceID, Targets: targets})
	if err == nil {
		return true
	}
	if errors.Is(err, managementauthorization.ErrDenied) {
		writeProviderError(response, 403, "forbidden", "Permission denied.", requestID)
	} else {
		writeProviderError(response, 503, "authorization_unavailable", "Authorization state is unavailable.", requestID)
	}
	return false
}

func (routes *IdentityResourceRoutes) operation(method, path string) managementapi.OperationContract {
	return routes.operations[method+" "+path]
}

func identityActor(request *http.Request, session managementauth.AuthenticatedSession, requestID, reason string) managementidentity.MutationActor {
	return managementidentity.MutationActor{PrincipalID: session.Session.PrincipalID, RequestID: requestID, SourceIP: directRequestIP(request), Reason: reason}
}

func identityScopeTarget(namespaceID string) map[string][]accesscontrol.ScopedTarget {
	if namespaceID == "" {
		return nil
	}
	target := accesscontrol.ScopedTarget{Scope: accesscontrol.NamespaceScope(accesscontrol.NamespaceID(namespaceID))}
	return map[string][]accesscontrol.ScopedTarget{"target": {target}}
}

func identityPathID(path, base string) (string, bool) {
	value := strings.TrimPrefix(path, base+"/")
	return value, value != path && !strings.Contains(value, "/") && canonicalUUID(value)
}

func identityPageRequest(response http.ResponseWriter, request *http.Request, requestID string, additional ...string) (managementidentity.ListRequest, bool) {
	query := request.URL.Query()
	allowed := map[string]bool{"cursor": true, "pageSize": true}
	for _, name := range additional {
		allowed[name] = true
	}
	for name, values := range query {
		if !allowed[name] || len(values) != 1 {
			writeProviderError(response, 400, "invalid_request", "Query parameters are invalid.", requestID)
			return managementidentity.ListRequest{}, false
		}
	}
	size := 50
	if raw := query.Get("pageSize"); raw != "" {
		parsed, err := strconv.Atoi(raw)
		if err != nil || parsed < 1 || parsed > 200 {
			writeProviderError(response, 400, "invalid_request", "pageSize must be between 1 and 200.", requestID)
			return managementidentity.ListRequest{}, false
		}
		size = parsed
	}
	after := ""
	if cursor := query.Get("cursor"); cursor != "" {
		decoded, err := base64.RawURLEncoding.DecodeString(cursor)
		if err != nil || !canonicalUUID(string(decoded)) {
			writeProviderError(response, 400, "invalid_cursor", "Cursor is invalid.", requestID)
			return managementidentity.ListRequest{}, false
		}
		after = string(decoded)
	}
	return managementidentity.ListRequest{AfterID: after, Limit: size}, true
}

func identityPageInfo(next string, size int) managementapi.PageInfo {
	info := managementapi.PageInfo{HasMore: next != "", PageSize: size}
	if next != "" {
		info.NextCursor = base64.RawURLEncoding.EncodeToString([]byte(next))
	}
	return info
}

var identityETagPattern = regexp.MustCompile(`^"(mp|mr|mb|msp|tii):([1-9][0-9]*)"$`)

func identityRevision(response http.ResponseWriter, request *http.Request, requestID, kind string) (uint64, bool) {
	values := request.Header.Values(managementapi.HeaderIfMatch)
	if len(values) != 1 {
		writeProviderError(response, 428, "precondition_required", "If-Match is required.", requestID)
		return 0, false
	}
	parts := identityETagPattern.FindStringSubmatch(values[0])
	if len(parts) != 3 || parts[1] != kind {
		writeProviderError(response, 400, "invalid_precondition", "If-Match is invalid.", requestID)
		return 0, false
	}
	value, _ := strconv.ParseUint(parts[2], 10, 64)
	return value, value > 0
}
func principalETag(revision uint64) string { return `"mp:` + strconv.FormatUint(revision, 10) + `"` }
func roleETag(revision uint64) string      { return `"mr:` + strconv.FormatUint(revision, 10) + `"` }
func bindingETag(revision uint64) string   { return `"mb:` + strconv.FormatUint(revision, 10) + `"` }
func policyETag(revision uint64) string    { return `"msp:` + strconv.FormatUint(revision, 10) + `"` }
func trustedIssuerETag(revision uint64) string {
	return `"tii:` + strconv.FormatUint(revision, 10) + `"`
}

func writeIdentityMutation(response http.ResponseWriter, result managementidentity.MutationResult, etag func(uint64) string, base, requestID string) {
	response.Header().Set(managementapi.HeaderETag, etag(result.Revision))
	response.Header().Set("Location", base+"/"+result.ID)
	setIdempotencyReplayHeader(response, result.Replayed)
	replayed := result.Replayed
	var metadata *bool
	if result.ResponseStatus == 201 {
		metadata = &replayed
	}
	writeProviderJSON(response, result.ResponseStatus, managementapi.NewResourceMutationReceipt(result.Kind, result.ID, result.Revision, metadata), requestID)
}

func writeIdentityError(response http.ResponseWriter, err error, requestID string) {
	switch {
	case errors.Is(err, managementidentity.ErrNotFound):
		writeProviderError(response, 404, "not_found", "Resource not found.", requestID)
	case errors.Is(err, managementidentity.ErrRevisionConflict):
		writeProviderError(response, 412, "revision_conflict", "Resource changed. Refresh and retry.", requestID)
	case errors.Is(err, managementidentity.ErrAlreadyExists), errors.Is(err, managementidentity.ErrRoleInUse),
		errors.Is(err, managementidentity.ErrPrincipalLinkInUse), errors.Is(err, managementidentity.ErrWorkloadDependency):
		writeProviderError(response, 409, "conflict", "Request conflicts with current state.", requestID)
	case errors.Is(err, managementidentity.ErrDelegationDenied):
		writeProviderError(response, 403, "forbidden", "Delegation is not allowed.", requestID)
	case errors.Is(err, managementidentity.ErrBuiltInImmutable):
		writeProviderError(response, 409, "immutable_resource", "Built-in roles cannot be changed.", requestID)
	case errors.Is(err, managementidentity.ErrInvalidLifecycleRequest):
		writeProviderError(response, 400, "invalid_request", "Identity request is invalid.", requestID)
	case errors.Is(err, managementcommand.ErrConflict):
		writeProviderError(response, 409, "idempotency_conflict", "Idempotency-Key was already used for a different request.", requestID)
	case errors.Is(err, managementauth.ErrSessionNotFound):
		writeProviderError(response, 404, "not_found", "Management session not found.", requestID)
	case errors.Is(err, managementauth.ErrSessionInactive), errors.Is(err, managementauth.ErrSessionConflict):
		writeProviderError(response, 409, "session_conflict", "Management session cannot be changed in its current state.", requestID)
	default:
		writeProviderError(response, 503, "identity_unavailable", "Identity state is unavailable.", requestID)
	}
}

func permissions(values []string) (accesscontrol.PermissionSet, error) {
	converted := make([]accesscontrol.Permission, len(values))
	for index := range values {
		converted[index] = accesscontrol.Permission(values[index])
	}
	return accesscontrol.NewPermissionSet(converted...)
}

func principalDTO(value managementidentity.Principal) managementapi.ManagementPrincipal {
	return managementapi.ManagementPrincipal{PrincipalID: string(value.Identity.ID), Issuer: value.Identity.Issuer, Subject: value.Identity.Subject, DisplayName: value.DisplayName, VerifiedEmail: value.VerifiedEmail, Attributes: value.Identity.Attributes, Status: string(value.Identity.Status), Revision: uint64(value.Revision), CreatedAt: value.Identity.CreatedAt, UpdatedAt: value.Identity.UpdatedAt}
}

func roleDTO(value managementidentity.Role) managementapi.ManagementRole {
	p := value.Role.Permissions.Permissions()
	values := make([]string, len(p))
	for i := range p {
		values[i] = string(p[i])
	}
	return managementapi.ManagementRole{RoleID: string(value.Role.ID), NamespaceID: string(value.Role.NamespaceID), Name: value.Role.Name, DisplayName: value.Role.DisplayName, Description: value.Description, Permissions: values, BuiltIn: value.Role.BuiltIn, Status: string(value.Role.Status), Revision: uint64(value.Role.Revision), CreatedAt: value.CreatedAt, UpdatedAt: value.UpdatedAt}
}

func bindingDTO(value managementidentity.RoleBinding) managementapi.ManagementRoleBinding {
	p := value.Binding.DelegationCeiling.Permissions()
	values := make([]string, len(p))
	for i := range p {
		values[i] = string(p[i])
	}
	s := value.Binding.Scope
	return managementapi.ManagementRoleBinding{BindingID: string(value.Binding.ID), PrincipalID: string(value.Binding.PrincipalID), RoleID: string(value.Binding.RoleID), Scope: managementapi.ManagementScope{Kind: string(s.Kind), NamespaceID: string(s.NamespaceID), TeamID: string(s.TeamID), UserID: string(s.UserID), ResourceType: string(s.ResourceType), ResourceID: string(s.ResourceID)}, DelegationCeiling: values, Status: string(value.Binding.Status), Revision: uint64(value.Binding.Revision), CreatedAt: value.CreatedAt, UpdatedAt: value.UpdatedAt}
}

func scopeFromDTO(value managementapi.ManagementScope) (accesscontrol.Scope, error) {
	switch accesscontrol.ScopeKind(value.Kind) {
	case accesscontrol.ScopeKindCluster:
		return accesscontrol.ClusterScope(), nil
	case accesscontrol.ScopeKindNamespace:
		return accesscontrol.NamespaceScope(accesscontrol.NamespaceID(value.NamespaceID)), nil
	case accesscontrol.ScopeKindTeam:
		return accesscontrol.TeamScope(accesscontrol.NamespaceID(value.NamespaceID), accesscontrol.TeamID(value.TeamID)), nil
	case accesscontrol.ScopeKindUser:
		return accesscontrol.UserScope(accesscontrol.NamespaceID(value.NamespaceID), accesscontrol.UserID(value.UserID)), nil
	case accesscontrol.ScopeKindResource:
		return accesscontrol.ResourceScope(accesscontrol.NamespaceID(value.NamespaceID), accesscontrol.ScopeResourceType(value.ResourceType), accesscontrol.ResourceID(value.ResourceID)), nil
	default:
		return accesscontrol.Scope{}, errors.New("invalid scope")
	}
}

var _ RouteRegistrar = (*IdentityResourceRoutes)(nil)
