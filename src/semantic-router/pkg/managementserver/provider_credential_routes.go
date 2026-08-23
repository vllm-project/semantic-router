package managementserver

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net"
	"net/http"
	"net/netip"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
	credentialmanagement "github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential/management"
)

const (
	providerCredentialPath      = managementapi.BasePath + "/provider-credentials"
	maximumCredentialBodyBytes  = 128 << 10
	maximumCredentialQueryBytes = 16 << 10
)

var providerCredentialETagPattern = regexp.MustCompile(`^"pc:([1-9][0-9]*)"$`)

type ProviderCredentialRoutes struct {
	service       ProviderCredentialService
	namespaces    NamespaceResolver
	sessions      SessionAuthenticator
	authorization Authorizer
	scopes        ResultScopeResolver
	now           func() time.Time
	operations    map[string]managementapi.OperationContract
}

func NewProviderCredentialRoutes(options ProviderCredentialRoutesOptions) (*ProviderCredentialRoutes, error) {
	scopes := configuredResultScopes(options.Scopes, options.Authorization)
	if options.Service == nil || options.Namespaces == nil || options.Sessions == nil || options.Authorization == nil || scopes == nil {
		return nil, fmt.Errorf("ProviderCredential Management routes require service, namespace, session, and authorization dependencies")
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	routes := &ProviderCredentialRoutes{
		service: options.Service, namespaces: options.Namespaces, sessions: options.Sessions,
		authorization: options.Authorization, scopes: scopes, now: now, operations: make(map[string]managementapi.OperationContract),
	}
	for _, operation := range []struct {
		method managementapi.HTTPMethod
		path   string
	}{
		{managementapi.MethodGET, providerCredentialPath},
		{managementapi.MethodPOST, providerCredentialPath},
		{managementapi.MethodGET, providerCredentialPath + "/{credentialId}"},
		{managementapi.MethodPATCH, providerCredentialPath + "/{credentialId}"},
		{managementapi.MethodDELETE, providerCredentialPath + "/{credentialId}"},
		{managementapi.MethodPOST, providerCredentialPath + "/{credentialId}:rotate"},
	} {
		contract, found := managementapi.LookupOperation(operation.method, operation.path)
		if !found {
			return nil, fmt.Errorf("ProviderCredential Management operation contract %s %s is unavailable", operation.method, operation.path)
		}
		routes.operations[string(operation.method)+" "+operation.path] = contract
	}
	return routes, nil
}

func (routes *ProviderCredentialRoutes) Register(mux *http.ServeMux) {
	if routes == nil || mux == nil {
		panic("ProviderCredential Management routes and mux are required")
	}
	mux.Handle("GET "+providerCredentialPath, routes)
	mux.Handle("POST "+providerCredentialPath, routes)
	mux.Handle("GET "+providerCredentialPath+"/", routes)
	mux.Handle("PATCH "+providerCredentialPath+"/", routes)
	mux.Handle("DELETE "+providerCredentialPath+"/", routes)
	mux.Handle("POST "+providerCredentialPath+"/", routes)
}

func (routes *ProviderCredentialRoutes) Ready(ctx context.Context) error {
	if routes == nil || routes.service == nil {
		return errors.New("ProviderCredential Management routes are unavailable")
	}
	return routes.service.Ready(ctx)
}

func (routes *ProviderCredentialRoutes) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	requestID := managementRequestID(request)
	setProviderResponseHeaders(response, requestID)
	if routes == nil || request == nil || request.URL == nil || request.URL.EscapedPath() != request.URL.Path {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	switch {
	case request.Method == http.MethodGet && request.URL.Path == providerCredentialPath:
		routes.list(response, request, requestID)
	case request.Method == http.MethodPost && request.URL.Path == providerCredentialPath:
		routes.create(response, request, requestID)
	case request.Method == http.MethodGet && strings.HasPrefix(request.URL.Path, providerCredentialPath+"/"):
		routes.detail(response, request, requestID)
	case request.Method == http.MethodPatch && strings.HasPrefix(request.URL.Path, providerCredentialPath+"/"):
		routes.patch(response, request, requestID)
	case request.Method == http.MethodDelete && strings.HasPrefix(request.URL.Path, providerCredentialPath+"/"):
		routes.delete(response, request, requestID)
	case request.Method == http.MethodPost && strings.HasPrefix(request.URL.Path, providerCredentialPath+"/"):
		routes.rotate(response, request, requestID)
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
}

func (routes *ProviderCredentialRoutes) list(response http.ResponseWriter, request *http.Request, requestID string) {
	query, err := strictProviderQuery(request.URL.RawQuery, map[string]bool{
		"cursor": true, "pageSize": true, "providerId": true, "status": true,
	})
	if err != nil || len(request.URL.RawQuery) > maximumCredentialQueryBytes {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Provider credential query is invalid.", requestID)
		return
	}
	pageSize, err := parseOptionalPageSize(query.Get("pageSize"))
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "pageSize must be between 1 and 200.", requestID)
		return
	}
	status := providercredential.Status(query.Get("status"))
	if status != "" && status != providercredential.StatusActive && status != providercredential.StatusDisabled && status != providercredential.StatusDeleted {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Provider credential status is invalid.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	operation := routes.operation(managementapi.MethodGET, providerCredentialPath)
	permission, valid := listPermission(operation)
	if !valid {
		writeProviderError(response, http.StatusServiceUnavailable, "authorization_unavailable", "Authorization state is unavailable.", requestID)
		return
	}
	scope, err := resolveListResultScope(request.Context(), routes.scopes, session, namespaceID, permission)
	if err != nil {
		writeResultScopeError(response, err, requestID)
		return
	}
	page, err := routes.service.List(request.Context(), credentialmanagement.ListRequest{
		NamespaceID: namespaceID, ProviderID: query.Get("providerId"), Status: status,
		Cursor: query.Get("cursor"), PageSize: pageSize, Scope: scope,
	})
	if err != nil {
		writeProviderCredentialDomainError(response, err, requestID)
		return
	}
	writeProviderJSON(response, http.StatusOK, newProviderCredentialPage(page), requestID)
}

func (routes *ProviderCredentialRoutes) create(response http.ResponseWriter, request *http.Request, requestID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Provider credential create does not accept query parameters.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	if !routes.authorize(response, request, requestID, session, namespaceID,
		routes.operation(managementapi.MethodPOST, providerCredentialPath), "") {
		return
	}
	key, ok := requireIdempotencyKey(response, request, requestID)
	if !ok {
		return
	}
	var body managementapi.ProviderCredentialCreateRequest
	if !decodeProviderCredentialBody(response, request, requestID, &body) {
		return
	}
	secret := []byte(body.Secret)
	zeroString(&body.Secret)
	defer providercredential.Zero(secret)
	result, err := routes.service.Create(request.Context(), credentialmanagement.CreateRequest{
		NamespaceID: namespaceID, Name: body.Name, ProviderID: body.ProviderID,
		BaseURL: body.BaseURL, Secret: secret, IdempotencyKey: string(key),
		Actor: routes.actor(request, session, requestID),
	})
	if err != nil {
		writeProviderCredentialDomainError(response, err, requestID)
		return
	}
	response.Header().Set(managementapi.HeaderETag, providerCredentialETag(result.Revision))
	response.Header().Set("Location", providerCredentialPath+"/"+result.CredentialID)
	setIdempotencyReplayHeader(response, result.Replayed)
	replayed := result.Replayed
	writeProviderJSON(response, http.StatusCreated, managementapi.NewResourceMutationReceipt(
		"provider_credential", result.CredentialID, result.Revision, &replayed,
	), requestID)
}

func (routes *ProviderCredentialRoutes) detail(response http.ResponseWriter, request *http.Request, requestID string) {
	credentialID, action, ok := providerCredentialPathValue(request.URL.Path)
	if !ok || action != "" || request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok || !routes.authorize(response, request, requestID, session, namespaceID,
		routes.operation(managementapi.MethodGET, providerCredentialPath+"/{credentialId}"), credentialID) {
		return
	}
	credential, err := routes.service.Get(request.Context(), namespaceID, credentialID)
	if err != nil {
		writeProviderCredentialDomainError(response, err, requestID)
		return
	}
	response.Header().Set(managementapi.HeaderETag, providerCredentialETag(credential.Revision))
	writeProviderJSON(response, http.StatusOK, managementapi.ProviderCredentialDetail{
		Data: newProviderCredential(credential),
	}, requestID)
}

func (routes *ProviderCredentialRoutes) patch(response http.ResponseWriter, request *http.Request, requestID string) {
	credentialID, action, ok := providerCredentialPathValue(request.URL.Path)
	if !ok || action != "" || request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok || !routes.authorize(response, request, requestID, session, namespaceID,
		routes.operation(managementapi.MethodPATCH, providerCredentialPath+"/{credentialId}"), credentialID) {
		return
	}
	revision, ok := requireProviderCredentialRevision(response, request, requestID)
	if !ok {
		return
	}
	var body managementapi.ProviderCredentialPatchRequest
	if !decodeProviderCredentialBody(response, request, requestID, &body) {
		return
	}
	actor := routes.actor(request, session, requestID)
	var result credentialmanagement.MutationResult
	var err error
	switch {
	case body.Name != nil && body.Status == nil && body.Secret == nil:
		result, err = routes.service.Rename(request.Context(), credentialmanagement.RenameRequest{
			NamespaceID: namespaceID, CredentialID: credentialID,
			ExpectedRevision: revision, Name: *body.Name, Actor: actor,
		})
	case body.Name == nil && body.Status != nil && *body.Status == "disabled" && body.Secret == nil:
		result, err = routes.service.Disable(request.Context(), credentialmanagement.LifecycleRequest{
			NamespaceID: namespaceID, CredentialID: credentialID,
			ExpectedRevision: revision, Actor: actor,
		})
	case body.Name == nil && body.Status != nil && *body.Status == "active" && body.Secret != nil:
		secret := []byte(*body.Secret)
		zeroString(body.Secret)
		defer providercredential.Zero(secret)
		result, err = routes.service.Reactivate(request.Context(), credentialmanagement.LifecycleRequest{
			NamespaceID: namespaceID, CredentialID: credentialID,
			ExpectedRevision: revision, Secret: secret, Actor: actor,
		})
	default:
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Patch must rename, disable, or reactivate the credential.", requestID)
		return
	}
	if err != nil {
		writeProviderCredentialDomainError(response, err, requestID)
		return
	}
	response.Header().Set(managementapi.HeaderETag, providerCredentialETag(result.Revision))
	writeProviderJSON(response, http.StatusOK, managementapi.NewResourceMutationReceipt(
		"provider_credential", result.CredentialID, result.Revision, nil,
	), requestID)
}

func (routes *ProviderCredentialRoutes) rotate(response http.ResponseWriter, request *http.Request, requestID string) {
	credentialID, action, ok := providerCredentialPathValue(request.URL.Path)
	if !ok || action != "rotate" || request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok || !routes.authorize(response, request, requestID, session, namespaceID,
		routes.operation(managementapi.MethodPOST, providerCredentialPath+"/{credentialId}:rotate"), credentialID) {
		return
	}
	revision, ok := requireProviderCredentialRevision(response, request, requestID)
	if !ok {
		return
	}
	key, ok := requireIdempotencyKey(response, request, requestID)
	if !ok {
		return
	}
	var body managementapi.ProviderCredentialRotateRequest
	if !decodeProviderCredentialBody(response, request, requestID, &body) {
		return
	}
	secret := []byte(body.Secret)
	zeroString(&body.Secret)
	defer providercredential.Zero(secret)
	result, err := routes.service.Rotate(request.Context(), credentialmanagement.RotateRequest{
		NamespaceID: namespaceID, CredentialID: credentialID, ExpectedRevision: revision,
		Secret: secret, IdempotencyKey: string(key), Actor: routes.actor(request, session, requestID),
	})
	if err != nil {
		writeProviderCredentialDomainError(response, err, requestID)
		return
	}
	response.Header().Set(managementapi.HeaderETag, providerCredentialETag(result.Revision))
	setIdempotencyReplayHeader(response, result.Replayed)
	replayed := result.Replayed
	writeProviderJSON(response, http.StatusOK, managementapi.NewResourceMutationReceipt(
		"provider_credential", result.CredentialID, result.Revision, &replayed,
	), requestID)
}

func (routes *ProviderCredentialRoutes) delete(response http.ResponseWriter, request *http.Request, requestID string) {
	credentialID, action, ok := providerCredentialPathValue(request.URL.Path)
	if !ok || action != "" || request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	if request.ContentLength > 0 || len(request.TransferEncoding) != 0 {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Delete does not accept a request body.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok || !routes.authorize(response, request, requestID, session, namespaceID,
		routes.operation(managementapi.MethodDELETE, providerCredentialPath+"/{credentialId}"), credentialID) {
		return
	}
	revision, ok := requireProviderCredentialRevision(response, request, requestID)
	if !ok {
		return
	}
	result, err := routes.service.Delete(request.Context(), credentialmanagement.LifecycleRequest{
		NamespaceID: namespaceID, CredentialID: credentialID,
		ExpectedRevision: revision, Actor: routes.actor(request, session, requestID),
	})
	if err != nil {
		writeProviderCredentialDomainError(response, err, requestID)
		return
	}
	setProviderResponseHeaders(response, requestID)
	response.Header().Set(managementapi.HeaderETag, providerCredentialETag(result.Revision))
	response.WriteHeader(http.StatusNoContent)
}

func (routes *ProviderCredentialRoutes) authenticate(
	response http.ResponseWriter,
	request *http.Request,
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

func (routes *ProviderCredentialRoutes) authorize(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
	session managementauth.AuthenticatedSession,
	namespaceID string,
	operation managementapi.OperationContract,
	credentialID string,
) bool {
	_, err := routes.authorization.Authorize(request.Context(), AuthorizationRequest{
		Operation: operation, Session: session, NamespaceID: namespaceID,
		Targets: providerCredentialAuthorizationTargets(namespaceID, credentialID),
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

func providerCredentialAuthorizationTargets(
	namespaceID string,
	credentialID string,
) map[string][]accesscontrol.ScopedTarget {
	if credentialID == "" {
		return nil
	}
	return map[string][]accesscontrol.ScopedTarget{
		"credential": {{Scope: accesscontrol.ResourceScope(
			accesscontrol.NamespaceID(namespaceID),
			accesscontrol.ScopeResourceProviderCredential,
			accesscontrol.ResourceID(credentialID),
		)}},
	}
}

func (routes *ProviderCredentialRoutes) operation(method managementapi.HTTPMethod, path string) managementapi.OperationContract {
	return routes.operations[string(method)+" "+path]
}

func (routes *ProviderCredentialRoutes) actor(
	request *http.Request,
	session managementauth.AuthenticatedSession,
	requestID string,
) credentialmanagement.Actor {
	return credentialmanagement.Actor{
		PrincipalID: session.Session.PrincipalID, RequestID: requestID,
		SourceIP: directRequestIP(request),
	}
}

func providerCredentialPathValue(path string) (string, string, bool) {
	value := strings.TrimPrefix(path, providerCredentialPath+"/")
	if value == path || value == "" || strings.Contains(value, "/") {
		return "", "", false
	}
	credentialID, action, hasAction := strings.Cut(value, ":")
	if !canonicalUUID(credentialID) || (hasAction && action == "") {
		return "", "", false
	}
	return credentialID, action, true
}

func decodeProviderCredentialBody(response http.ResponseWriter, request *http.Request, requestID string, target any) bool {
	if request.ContentLength > maximumCredentialBodyBytes {
		writeProviderError(response, http.StatusRequestEntityTooLarge, "invalid_request", "Provider credential request body is too large.", requestID)
		return false
	}
	mediaType, parameters, err := mime.ParseMediaType(request.Header.Get("Content-Type"))
	if err != nil || mediaType != managementapi.JSONMediaType ||
		(len(parameters) != 0 && (len(parameters) != 1 || !strings.EqualFold(parameters["charset"], "utf-8"))) {
		writeProviderError(response, http.StatusUnsupportedMediaType, "unsupported_media_type", "Use the Management API media type.", requestID)
		return false
	}
	request.Body = http.MaxBytesReader(response, request.Body, maximumCredentialBodyBytes)
	decoder := json.NewDecoder(request.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		var maximum *http.MaxBytesError
		if errors.As(err, &maximum) {
			writeProviderError(response, http.StatusRequestEntityTooLarge, "invalid_request", "Provider credential request body is too large.", requestID)
		} else {
			writeProviderError(response, http.StatusBadRequest, "invalid_request", "Provider credential request body is invalid.", requestID)
		}
		return false
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Provider credential request body is invalid.", requestID)
		return false
	}
	return true
}

func requireIdempotencyKey(response http.ResponseWriter, request *http.Request, requestID string) (managementapi.IdempotencyKey, bool) {
	values := request.Header.Values(managementapi.HeaderIdempotencyKey)
	if len(values) != 1 {
		writeProviderError(response, http.StatusBadRequest, "invalid_idempotency_key", "A valid Idempotency-Key is required.", requestID)
		return "", false
	}
	key, err := managementapi.ParseIdempotencyKey(values[0])
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_idempotency_key", "A valid Idempotency-Key is required.", requestID)
		return "", false
	}
	return key, true
}

func requireProviderCredentialRevision(response http.ResponseWriter, request *http.Request, requestID string) (uint64, bool) {
	values := request.Header.Values(managementapi.HeaderIfMatch)
	if len(values) != 1 {
		writeProviderError(response, http.StatusPreconditionRequired, "precondition_required", "If-Match is required.", requestID)
		return 0, false
	}
	match := providerCredentialETagPattern.FindStringSubmatch(values[0])
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

func providerCredentialETag(revision uint64) string {
	return `"pc:` + strconv.FormatUint(revision, 10) + `"`
}

func setIdempotencyReplayHeader(response http.ResponseWriter, replayed bool) {
	if replayed {
		response.Header().Set(managementapi.HeaderIdempotencyReplayed, "true")
	}
}

func directRequestIP(request *http.Request) netip.Addr {
	if request == nil {
		return netip.Addr{}
	}
	host, _, err := net.SplitHostPort(request.RemoteAddr)
	if err != nil {
		host = request.RemoteAddr
	}
	address, err := netip.ParseAddr(host)
	if err != nil {
		return netip.Addr{}
	}
	return address.Unmap()
}

func zeroString(value *string) {
	if value != nil {
		*value = ""
	}
}
