package managementserver

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apikeymanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
)

const (
	apiKeysPath            = managementapi.BasePath + "/api-keys"
	maximumAPIKeyBodyBytes = 128 << 10
)

var apiKeyETagPattern = regexp.MustCompile(`^"key:([1-9][0-9]*)"$`)

type APIKeyRoutes struct {
	service       APIKeyManagementService
	namespaces    NamespaceResolver
	sessions      SessionAuthenticator
	authorization Authorizer
	scopes        ResultScopeResolver
	now           func() time.Time
	operations    map[string]managementapi.OperationContract
}

func NewAPIKeyRoutes(options APIKeyRoutesOptions) (*APIKeyRoutes, error) {
	scopes := configuredResultScopes(options.Scopes, options.Authorization)
	if options.Service == nil || options.Namespaces == nil || options.Sessions == nil || options.Authorization == nil || scopes == nil {
		return nil, errors.New("API-key Management routes require service, namespace, session, and authorization dependencies")
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	routes := &APIKeyRoutes{
		service: options.Service, namespaces: options.Namespaces, sessions: options.Sessions,
		authorization: options.Authorization, scopes: scopes, now: now, operations: make(map[string]managementapi.OperationContract),
	}
	for _, contract := range apiKeyHTTPContracts() {
		operation, found := managementapi.LookupOperation(contract.method, contract.path)
		if !found {
			return nil, fmt.Errorf("API-key Management operation contract %s %s is unavailable", contract.method, contract.path)
		}
		routes.operations[apiKeyOperationKey(contract.method, contract.path)] = operation
	}
	return routes, nil
}

func (routes *APIKeyRoutes) Register(mux *http.ServeMux) {
	if routes == nil || mux == nil {
		panic("API-key Management routes and mux are required")
	}
	for _, pattern := range []string{
		"GET " + apiKeysPath, "POST " + apiKeysPath,
		"GET " + apiKeysPath + "/", "POST " + apiKeysPath + "/",
		"PATCH " + apiKeysPath + "/", "DELETE " + apiKeysPath + "/",
	} {
		mux.Handle(pattern, routes)
	}
}

func (routes *APIKeyRoutes) Ready(ctx context.Context) error {
	if routes == nil || routes.service == nil {
		return apikeymanagement.ErrUnavailable
	}
	return routes.service.Ready(ctx)
}

func (routes *APIKeyRoutes) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	requestID := managementRequestID(request)
	setProviderResponseHeaders(response, requestID)
	if routes == nil || request == nil || request.URL == nil || request.URL.EscapedPath() != request.URL.Path {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	if request.URL.Path == apiKeysPath {
		switch request.Method {
		case http.MethodGet:
			routes.list(response, request, requestID)
		case http.MethodPost:
			routes.create(response, request, requestID)
		default:
			writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		}
		return
	}
	path, ok := parseAPIKeyPath(request.URL.Path)
	if !ok {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	switch {
	case !path.credentials && path.action == "" && request.Method == http.MethodGet:
		routes.detail(response, request, requestID, path.keyID)
	case !path.credentials && path.action == "" && request.Method == http.MethodPatch:
		routes.rename(response, request, requestID, path.keyID)
	case !path.credentials && path.action == "" && request.Method == http.MethodDelete:
		routes.delete(response, request, requestID, path.keyID)
	case !path.credentials && request.Method == http.MethodPost:
		routes.action(response, request, requestID, path.keyID, path.action)
	case path.credentials && path.credentialID == "" && path.action == "" && request.Method == http.MethodGet:
		routes.listCredentials(response, request, requestID, path.keyID)
	case path.credentials && path.credentialID == "" && path.action == "rotate" && request.Method == http.MethodPost:
		routes.rotate(response, request, requestID, path.keyID)
	case path.credentials && path.credentialID != "" && path.action == "reveal" && request.Method == http.MethodPost:
		routes.reveal(response, request, requestID, path.keyID, path.credentialID)
	case path.credentials && path.credentialID != "" && path.action == "" && request.Method == http.MethodDelete:
		routes.revokeCredential(response, request, requestID, path.keyID, path.credentialID)
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
}

func (routes *APIKeyRoutes) list(response http.ResponseWriter, request *http.Request, requestID string) {
	query, err := strictProviderQuery(request.URL.RawQuery, map[string]bool{
		"cursor": true, "pageSize": true, "status": true, "ownerType": true, "ownerId": true,
		"search": true,
	})
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "API key query is invalid.", requestID)
		return
	}
	pageSize, err := parseOptionalPageSize(query.Get("pageSize"))
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "pageSize must be between 1 and 200.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	operation := routes.operation(managementapi.MethodGET, apiKeysPath)
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
	page, err := routes.service.List(request.Context(), apikeymanagement.ListKeysRequest{
		NamespaceID: namespaceID, Status: accesscontrol.APIKeyStatus(query.Get("status")),
		OwnerKind: accesscontrol.SubjectKind(query.Get("ownerType")), OwnerID: query.Get("ownerId"),
		Search: query.Get("search"), Cursor: query.Get("cursor"), PageSize: pageSize, Scope: scope,
	})
	if err != nil {
		writeAPIKeyError(response, err, requestID)
		return
	}
	writeProviderJSON(response, http.StatusOK, newAPIKeyPage(page), requestID)
}

func (routes *APIKeyRoutes) create(response http.ResponseWriter, request *http.Request, requestID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "API key create does not accept query parameters.", requestID)
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
	var body managementapi.APIKeyCreateRequest
	if !decodeAPIKeyBody(response, request, requestID, &body) {
		return
	}
	owner := apikeymanagement.Owner{Kind: accesscontrol.SubjectKind(body.Owner.Type), ID: body.Owner.ID}
	target, valid := apiKeyOwnerTarget(namespaceID, owner)
	rateLimitOverride, validOverride := apiKeyRateLimitOverride(body.RateLimitOverride)
	if !validOverride {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Rate limit override must select one existing or inline policy.", requestID)
		return
	}
	targets := map[string][]accesscontrol.ScopedTarget{"owner": {target}}
	if len(body.AccessPolicyIDs) > 0 {
		targets["access_policy"] = make([]accesscontrol.ScopedTarget, 0, len(body.AccessPolicyIDs))
		for _, policyID := range body.AccessPolicyIDs {
			targets["access_policy"] = append(targets["access_policy"], subjectResourceTarget(
				namespaceID, accesscontrol.ScopeResourceAccessPolicy, policyID))
		}
	}
	existingRatePolicy := rateLimitOverride != nil && rateLimitOverride.PolicyID != ""
	if existingRatePolicy {
		targets["rate_policy"] = []accesscontrol.ScopedTarget{subjectResourceTarget(
			namespaceID, accesscontrol.ScopeResourceRateLimitPolicy, rateLimitOverride.PolicyID)}
	}
	inlineRatePolicy := rateLimitOverride != nil && rateLimitOverride.InlinePolicy != nil
	if !valid || !routes.authorize(response, request, requestID, session, namespaceID,
		routes.operation(managementapi.MethodPOST, apiKeysPath),
		targets, map[string]bool{
			"access_policy_binding_requested": len(body.AccessPolicyIDs) > 0,
			"rate_policy_binding_requested":   existingRatePolicy,
			"inline_rate_policy_requested":    inlineRatePolicy,
		}, false) {
		return
	}
	result, err := routes.service.Create(request.Context(), apikeymanagement.CreateRequest{
		NamespaceID: namespaceID, Name: body.Name, Owner: owner, ContextTeamID: body.ContextTeamID,
		ExpiresAt: body.ExpiresAt, Revealable: body.Revealable,
		AccessPolicyIDs: body.AccessPolicyIDs, RateLimitOverride: rateLimitOverride,
		IdempotencyKey: string(idempotencyKey),
		Actor:          routes.actor(request, session, requestID),
	})
	if err != nil {
		writeAPIKeyError(response, err, requestID)
		return
	}
	response.Header().Set("Location", apiKeysPath+"/"+string(result.Key.ID))
	writeAPIKeySecret(response, http.StatusCreated, result, requestID)
}

func (routes *APIKeyRoutes) detail(response http.ResponseWriter, request *http.Request, requestID, keyID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "API key detail does not accept query parameters.", requestID)
		return
	}
	key, _, ok := routes.keyForAuthorizedRequest(response, request, requestID, keyID,
		routes.operation(managementapi.MethodGET, apiKeysPath+"/{keyId}"), true)
	if !ok {
		return
	}
	response.Header().Set(managementapi.HeaderETag, apiKeyETag(uint64(key.Revision)))
	writeProviderJSON(response, http.StatusOK, managementapi.APIKeyDetail{Data: newAPIKey(key)}, requestID)
}

func (routes *APIKeyRoutes) rename(response http.ResponseWriter, request *http.Request, requestID, keyID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "API key update does not accept query parameters.", requestID)
		return
	}
	_, session, ok := routes.keyForAuthorizedRequest(response, request, requestID, keyID,
		routes.operation(managementapi.MethodPATCH, apiKeysPath+"/{keyId}"), false)
	if !ok {
		return
	}
	revision, ok := requireAPIKeyRevision(response, request, requestID)
	if !ok {
		return
	}
	var body managementapi.APIKeyPatchRequest
	if !decodeAPIKeyBody(response, request, requestID, &body) {
		return
	}
	result, err := routes.service.Rename(request.Context(), apikeymanagement.RenameRequest{
		NamespaceID: session.NamespaceID, KeyID: keyID, ExpectedRevision: revision,
		Name: body.Name, Actor: routes.actor(request, session, requestID),
	})
	if err != nil {
		writeAPIKeyError(response, err, requestID)
		return
	}
	routes.writeKeyMutation(response, result, requestID)
}

func (routes *APIKeyRoutes) delete(response http.ResponseWriter, request *http.Request, requestID, keyID string) {
	if request.URL.RawQuery != "" || !noRequestBody(response, request, requestID) {
		return
	}
	_, session, ok := routes.keyForAuthorizedRequest(response, request, requestID, keyID,
		routes.operation(managementapi.MethodDELETE, apiKeysPath+"/{keyId}"), false)
	if !ok {
		return
	}
	revision, ok := requireAPIKeyRevision(response, request, requestID)
	if !ok {
		return
	}
	result, err := routes.service.Delete(request.Context(), apikeymanagement.LifecycleRequest{
		NamespaceID: session.NamespaceID, KeyID: keyID, ExpectedRevision: revision,
		Actor: routes.actor(request, session, requestID),
	})
	if err != nil {
		writeAPIKeyError(response, err, requestID)
		return
	}
	response.Header().Set(managementapi.HeaderETag, apiKeyETag(uint64(result.Key.Revision)))
	response.WriteHeader(http.StatusNoContent)
}

func (routes *APIKeyRoutes) action(response http.ResponseWriter, request *http.Request, requestID, keyID, action string) {
	operationPath := apiKeysPath + "/{keyId}:" + action
	operation, found := managementapi.LookupOperation(managementapi.MethodPOST, operationPath)
	if !found || (action != "enable" && action != "disable" && action != "renew" && action != "reassign") || request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	key, session, ok := routes.keyForRequest(response, request, requestID, keyID)
	if !ok {
		return
	}
	// Every action first proves key.manage on the authoritative key. Reassign
	// performs its additional current/target-owner checks after strict body
	// decoding, but an out-of-scope caller never learns precondition state.
	if !routes.authorize(response, request, requestID, session, session.NamespaceID,
		routes.operation(managementapi.MethodPATCH, apiKeysPath+"/{keyId}"),
		apiKeyTargets(key), nil, true) {
		return
	}
	revision, ok := requireAPIKeyRevision(response, request, requestID)
	if !ok {
		return
	}
	idempotencyKey, ok := requireIdempotencyKey(response, request, requestID)
	if !ok {
		return
	}
	actor := routes.actor(request, session, requestID)
	var result apikeymanagement.MutationResult
	var err error
	switch action {
	case "enable", "disable":
		var body managementapi.APIKeyLifecycleRequest
		if !decodeAPIKeyBody(response, request, requestID, &body) || !routes.authorize(response, request, requestID,
			session, session.NamespaceID, operation, apiKeyTargets(key), nil, false) {
			return
		}
		input := apikeymanagement.LifecycleRequest{
			NamespaceID: session.NamespaceID, KeyID: keyID,
			ExpectedRevision: revision, IdempotencyKey: string(idempotencyKey), Actor: actor,
		}
		if action == "enable" {
			result, err = routes.service.Enable(request.Context(), input)
		} else {
			result, err = routes.service.Disable(request.Context(), input)
		}
	case "renew":
		var body managementapi.APIKeyRenewRequest
		if !decodeAPIKeyBody(response, request, requestID, &body) || !routes.authorize(response, request, requestID,
			session, session.NamespaceID, operation, apiKeyTargets(key), nil, false) {
			return
		}
		result, err = routes.service.Renew(request.Context(), apikeymanagement.RenewRequest{
			NamespaceID: session.NamespaceID, KeyID: keyID, ExpectedRevision: revision,
			ExpiresAt: body.ExpiresAt, IdempotencyKey: string(idempotencyKey), Actor: actor,
		})
	case "reassign":
		var body managementapi.APIKeyReassignRequest
		if !decodeAPIKeyBody(response, request, requestID, &body) {
			return
		}
		targetOwner := apikeymanagement.Owner{Kind: accesscontrol.SubjectKind(body.Owner.Type), ID: body.Owner.ID}
		target, valid := apiKeyOwnerTarget(session.NamespaceID, targetOwner)
		current, currentValid := apiKeyOwnerTarget(session.NamespaceID, apikeymanagement.Owner{Kind: key.Owner.Kind, ID: string(key.Owner.ID)})
		if !valid || !currentValid {
			writeProviderError(response, http.StatusBadRequest, "invalid_request", "API key owner is invalid.", requestID)
			return
		}
		targets := apiKeyTargets(key)
		targets["current_owner"] = []accesscontrol.ScopedTarget{current}
		targets["target_owner"] = []accesscontrol.ScopedTarget{target}
		conditions := map[string]bool{
			"current_user_owner": key.Owner.Kind == accesscontrol.SubjectKindUser,
			"current_team_owner": key.Owner.Kind == accesscontrol.SubjectKindTeam,
			"target_user_owner":  targetOwner.Kind == accesscontrol.SubjectKindUser,
			"target_team_owner":  targetOwner.Kind == accesscontrol.SubjectKindTeam,
		}
		if !routes.authorize(response, request, requestID, session, session.NamespaceID, operation, targets, conditions, false) {
			return
		}
		result, err = routes.service.Reassign(request.Context(), apikeymanagement.ReassignRequest{
			NamespaceID: session.NamespaceID, KeyID: keyID, ExpectedRevision: revision,
			Owner: targetOwner, ContextTeamID: body.ContextTeamID,
			IdempotencyKey: string(idempotencyKey), Actor: actor,
		})
	}
	if err != nil {
		writeAPIKeyError(response, err, requestID)
		return
	}
	routes.writeKeyMutation(response, result, requestID)
}

func (routes *APIKeyRoutes) listCredentials(response http.ResponseWriter, request *http.Request, requestID, keyID string) {
	query, err := strictProviderQuery(request.URL.RawQuery, map[string]bool{
		"cursor": true, "pageSize": true, "status": true,
	})
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Credential query is invalid.", requestID)
		return
	}
	pageSize, err := parseOptionalPageSize(query.Get("pageSize"))
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "pageSize must be between 1 and 200.", requestID)
		return
	}
	_, session, ok := routes.keyForAuthorizedRequest(response, request, requestID, keyID,
		routes.operation(managementapi.MethodGET, apiKeysPath+"/{keyId}/credentials"), true)
	if !ok {
		return
	}
	page, err := routes.service.ListCredentials(request.Context(), apikeymanagement.ListCredentialsRequest{
		NamespaceID: session.NamespaceID, KeyID: keyID,
		Status: accesscontrol.CredentialStatus(query.Get("status")), Cursor: query.Get("cursor"), PageSize: pageSize,
	})
	if err != nil {
		writeAPIKeyError(response, err, requestID)
		return
	}
	writeProviderJSON(response, http.StatusOK, newAPIKeyCredentialPage(page), requestID)
}

func (routes *APIKeyRoutes) rotate(response http.ResponseWriter, request *http.Request, requestID, keyID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Credential rotation does not accept query parameters.", requestID)
		return
	}
	_, session, ok := routes.keyForAuthorizedRequest(response, request, requestID, keyID,
		routes.operation(managementapi.MethodPOST, apiKeysPath+"/{keyId}/credentials:rotate"), false)
	if !ok {
		return
	}
	revision, ok := requireAPIKeyRevision(response, request, requestID)
	if !ok {
		return
	}
	idempotencyKey, ok := requireIdempotencyKey(response, request, requestID)
	if !ok {
		return
	}
	var body managementapi.APIKeyRotateRequest
	if !decodeAPIKeyBody(response, request, requestID, &body) {
		return
	}
	result, err := routes.service.Rotate(request.Context(), apikeymanagement.RotateRequest{
		NamespaceID: session.NamespaceID, KeyID: keyID, ExpectedRevision: revision,
		Overlap: time.Duration(body.OverlapSeconds) * time.Second, Revealable: body.Revealable,
		IdempotencyKey: string(idempotencyKey), Actor: routes.actor(request, session, requestID),
	})
	if err != nil {
		writeAPIKeyError(response, err, requestID)
		return
	}
	writeAPIKeySecret(response, http.StatusOK, result, requestID)
}

func (routes *APIKeyRoutes) reveal(response http.ResponseWriter, request *http.Request, requestID, keyID, credentialID string) {
	if request.URL.RawQuery != "" || !noRequestBody(response, request, requestID) {
		return
	}
	_, session, ok := routes.keyForAuthorizedRequest(response, request, requestID, keyID,
		routes.operation(managementapi.MethodPOST, apiKeysPath+"/{keyId}/credentials/{credentialId}:reveal"), true)
	if !ok {
		return
	}
	secret, err := routes.service.Reveal(request.Context(), apikeymanagement.RevealRequest{
		NamespaceID: session.NamespaceID, KeyID: keyID, CredentialID: credentialID,
		Actor: routes.actor(request, session, requestID),
	})
	if err != nil {
		writeAPIKeyError(response, err, requestID)
		return
	}
	writeProviderJSON(response, http.StatusOK, managementapi.APIKeyRevealResponse{
		KeyID: keyID, CredentialID: credentialID, Secret: secret,
	}, requestID)
}

func (routes *APIKeyRoutes) revokeCredential(response http.ResponseWriter, request *http.Request, requestID, keyID, credentialID string) {
	if request.URL.RawQuery != "" || !noRequestBody(response, request, requestID) {
		return
	}
	_, session, ok := routes.keyForAuthorizedRequest(response, request, requestID, keyID,
		routes.operation(managementapi.MethodDELETE, apiKeysPath+"/{keyId}/credentials/{credentialId}"), false)
	if !ok {
		return
	}
	revision, ok := requireAPIKeyRevision(response, request, requestID)
	if !ok {
		return
	}
	result, err := routes.service.RevokeCredential(request.Context(), apikeymanagement.RevokeCredentialRequest{
		NamespaceID: session.NamespaceID, KeyID: keyID, CredentialID: credentialID,
		ExpectedRevision: revision, Actor: routes.actor(request, session, requestID),
	})
	if err != nil {
		writeAPIKeyError(response, err, requestID)
		return
	}
	response.Header().Set(managementapi.HeaderETag, apiKeyETag(uint64(result.Key.Revision)))
	response.WriteHeader(http.StatusNoContent)
}

func (routes *APIKeyRoutes) authenticate(response http.ResponseWriter, request *http.Request, requestID string) (string, managementauth.AuthenticatedSession, bool) {
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

func (routes *APIKeyRoutes) keyForRequest(response http.ResponseWriter, request *http.Request, requestID, keyID string) (accesscontrol.APIKey, managementauth.AuthenticatedSession, bool) {
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return accesscontrol.APIKey{}, managementauth.AuthenticatedSession{}, false
	}
	key, err := routes.service.Get(request.Context(), namespaceID, keyID)
	if err != nil {
		writeAPIKeyError(response, err, requestID)
		return accesscontrol.APIKey{}, managementauth.AuthenticatedSession{}, false
	}
	return key, session, true
}

func (routes *APIKeyRoutes) keyForAuthorizedRequest(response http.ResponseWriter, request *http.Request, requestID, keyID string,
	operation managementapi.OperationContract, nondisclosing bool,
) (accesscontrol.APIKey, managementauth.AuthenticatedSession, bool) {
	key, session, ok := routes.keyForRequest(response, request, requestID, keyID)
	if !ok || !routes.authorize(response, request, requestID, session, session.NamespaceID,
		operation, apiKeyTargets(key), nil, nondisclosing) {
		return accesscontrol.APIKey{}, managementauth.AuthenticatedSession{}, false
	}
	return key, session, true
}

func (routes *APIKeyRoutes) authorize(response http.ResponseWriter, request *http.Request, requestID string,
	session managementauth.AuthenticatedSession, namespaceID string, operation managementapi.OperationContract,
	targets map[string][]accesscontrol.ScopedTarget, conditions map[string]bool, nondisclosing bool,
) bool {
	_, err := routes.authorization.Authorize(request.Context(), AuthorizationRequest{
		Operation: operation, Session: session, NamespaceID: namespaceID, Targets: targets, Conditions: conditions,
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

func (routes *APIKeyRoutes) actor(request *http.Request, session managementauth.AuthenticatedSession, requestID string) apikeymanagement.Actor {
	return apikeymanagement.Actor{
		PrincipalID: session.Session.PrincipalID, ActorChain: []string{session.Session.PrincipalID},
		RequestID: requestID, SourceIP: directRequestIP(request),
	}
}

func (routes *APIKeyRoutes) operation(method managementapi.HTTPMethod, path string) managementapi.OperationContract {
	return routes.operations[apiKeyOperationKey(method, path)]
}

func (routes *APIKeyRoutes) writeKeyMutation(response http.ResponseWriter, result apikeymanagement.MutationResult, requestID string) {
	response.Header().Set(managementapi.HeaderETag, apiKeyETag(uint64(result.Key.Revision)))
	setIdempotencyReplayHeader(response, result.Replayed)
	writeProviderJSON(response, result.HTTPStatus, managementapi.APIKeyDetail{Data: newAPIKey(result.Key)}, requestID)
}

type apiKeyHTTPContract struct {
	method managementapi.HTTPMethod
	path   string
}

func apiKeyHTTPContracts() []apiKeyHTTPContract {
	return []apiKeyHTTPContract{
		{managementapi.MethodGET, apiKeysPath},
		{managementapi.MethodPOST, apiKeysPath},
		{managementapi.MethodGET, apiKeysPath + "/{keyId}"},
		{managementapi.MethodPATCH, apiKeysPath + "/{keyId}"},
		{managementapi.MethodDELETE, apiKeysPath + "/{keyId}"},
		{managementapi.MethodPOST, apiKeysPath + "/{keyId}:enable"},
		{managementapi.MethodPOST, apiKeysPath + "/{keyId}:disable"},
		{managementapi.MethodPOST, apiKeysPath + "/{keyId}:renew"},
		{managementapi.MethodPOST, apiKeysPath + "/{keyId}:reassign"},
		{managementapi.MethodGET, apiKeysPath + "/{keyId}/credentials"},
		{managementapi.MethodPOST, apiKeysPath + "/{keyId}/credentials:rotate"},
		{managementapi.MethodPOST, apiKeysPath + "/{keyId}/credentials/{credentialId}:reveal"},
		{managementapi.MethodDELETE, apiKeysPath + "/{keyId}/credentials/{credentialId}"},
	}
}

func apiKeyOperationKey(method managementapi.HTTPMethod, path string) string {
	return string(method) + " " + path
}

type parsedAPIKeyPath struct {
	keyID        string
	credentialID string
	action       string
	credentials  bool
}

func parseAPIKeyPath(path string) (parsedAPIKeyPath, bool) {
	value := strings.TrimPrefix(path, apiKeysPath+"/")
	if value == path || value == "" {
		return parsedAPIKeyPath{}, false
	}
	segments := strings.Split(value, "/")
	keyID, action, hasAction := strings.Cut(segments[0], ":")
	if !canonicalUUID(keyID) {
		return parsedAPIKeyPath{}, false
	}
	if len(segments) == 1 {
		if hasAction && action == "" {
			return parsedAPIKeyPath{}, false
		}
		return parsedAPIKeyPath{keyID: keyID, action: action}, true
	}
	if hasAction || len(segments) > 3 {
		return parsedAPIKeyPath{}, false
	}
	credentialSegment, credentialAction, credentialHasAction := strings.Cut(segments[1], ":")
	if credentialSegment != "credentials" {
		return parsedAPIKeyPath{}, false
	}
	if len(segments) == 2 {
		if credentialHasAction && credentialAction == "" {
			return parsedAPIKeyPath{}, false
		}
		return parsedAPIKeyPath{keyID: keyID, action: credentialAction, credentials: true}, true
	}
	if credentialHasAction {
		return parsedAPIKeyPath{}, false
	}
	credentialID, revealAction, revealHasAction := strings.Cut(segments[2], ":")
	if !canonicalUUID(credentialID) || (revealHasAction && revealAction == "") {
		return parsedAPIKeyPath{}, false
	}
	return parsedAPIKeyPath{keyID: keyID, credentialID: credentialID, action: revealAction, credentials: true}, true
}

func apiKeyTargets(key accesscontrol.APIKey) map[string][]accesscontrol.ScopedTarget {
	target := accesscontrol.ScopedTarget{Scope: accesscontrol.ResourceScope(
		key.NamespaceID, accesscontrol.ScopeResourceAPIKey, accesscontrol.ResourceID(key.ID),
	)}
	owner, valid := apiKeyOwnerTarget(string(key.NamespaceID), apikeymanagement.Owner{
		Kind: key.Owner.Kind, ID: string(key.Owner.ID),
	})
	if valid {
		target.Ancestors = []accesscontrol.Scope{owner.Scope}
	}
	return map[string][]accesscontrol.ScopedTarget{"key": {target}}
}

func apiKeyOwnerTarget(namespaceID string, owner apikeymanagement.Owner) (accesscontrol.ScopedTarget, bool) {
	switch owner.Kind {
	case accesscontrol.SubjectKindUser:
		return accesscontrol.ScopedTarget{Scope: accesscontrol.UserScope(
			accesscontrol.NamespaceID(namespaceID), accesscontrol.UserID(owner.ID),
		)}, canonicalUUID(owner.ID)
	case accesscontrol.SubjectKindTeam:
		return accesscontrol.ScopedTarget{Scope: accesscontrol.TeamScope(
			accesscontrol.NamespaceID(namespaceID), accesscontrol.TeamID(owner.ID),
		)}, canonicalUUID(owner.ID)
	default:
		return accesscontrol.ScopedTarget{}, false
	}
}

func apiKeyRateLimitOverride(input *managementapi.APIKeyRateLimitOverride) (*apikeymanagement.RateLimitOverrideInput, bool) {
	if input == nil {
		return nil, true
	}
	hasPolicy := input.PolicyID != ""
	hasInline := input.InlinePolicy != nil
	if hasPolicy == hasInline {
		return nil, false
	}
	result := &apikeymanagement.RateLimitOverrideInput{PolicyID: input.PolicyID}
	if input.InlinePolicy != nil {
		rules, err := policyRules(input.InlinePolicy.Rules)
		if err != nil {
			return nil, false
		}
		result.InlinePolicy = &apikeymanagement.InlineRateLimitPolicyInput{
			Name: input.InlinePolicy.Name, Description: input.InlinePolicy.Description,
			Rules: rules,
		}
	}
	return result, true
}
