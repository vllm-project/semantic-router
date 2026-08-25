package managementserver

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity"
)

const (
	serviceAccountsPath = managementapi.BasePath + "/service-accounts"
	mtlsMappingsPath    = managementapi.BasePath + "/mtls-identity-mappings"
)

type WorkloadIdentityLifecycle interface {
	Ready(context.Context) error
}

type ServiceAccountReadService interface {
	GetServiceAccount(context.Context, string) (managementidentity.ServiceAccount, error)
	ListServiceAccounts(context.Context, managementidentity.ServiceAccountListRequest) (managementidentity.WorkloadPage[managementidentity.ServiceAccount], error)
	ListServiceCredentials(context.Context, managementidentity.ServiceCredentialListRequest) (managementidentity.WorkloadPage[managementidentity.ServiceCredential], error)
}

type ServiceAccountMutationService interface {
	CreateServiceAccount(context.Context, managementidentity.CreateServiceAccountRequest) (managementidentity.ServiceCredentialSecretResult, error)
	PatchServiceAccount(context.Context, managementidentity.PatchServiceAccountRequest) (managementidentity.WorkloadMutationResult, error)
	DeleteServiceAccount(context.Context, managementidentity.DeleteServiceAccountRequest) (managementidentity.WorkloadMutationResult, error)
	RotateServiceCredential(context.Context, managementidentity.RotateServiceCredentialRequest) (managementidentity.ServiceCredentialSecretResult, error)
	RevokeServiceCredential(context.Context, managementidentity.RevokeServiceCredentialRequest) (managementidentity.WorkloadMutationResult, error)
}

type MTLSMappingReadService interface {
	GetMTLSMapping(context.Context, string) (managementidentity.MTLSIdentityMapping, error)
	ListMTLSMappings(context.Context, managementidentity.MTLSMappingListRequest) (managementidentity.WorkloadPage[managementidentity.MTLSIdentityMapping], error)
}

type MTLSMappingMutationService interface {
	CreateMTLSMapping(context.Context, managementidentity.CreateMTLSMappingRequest) (managementidentity.WorkloadMutationResult, error)
	PatchMTLSMapping(context.Context, managementidentity.PatchMTLSMappingRequest) (managementidentity.WorkloadMutationResult, error)
	DeleteMTLSMapping(context.Context, managementidentity.DeleteMTLSMappingRequest) (managementidentity.WorkloadMutationResult, error)
}

type WorkloadIdentityManagement interface {
	WorkloadIdentityLifecycle
	ServiceAccountReadService
	ServiceAccountMutationService
	MTLSMappingReadService
	MTLSMappingMutationService
}

type WorkloadIdentityRoutesOptions struct {
	Service       WorkloadIdentityManagement
	Sessions      SessionAuthenticator
	Authorization Authorizer
	Scopes        ResultScopeResolver
	Now           func() time.Time
}

type WorkloadIdentityRoutes struct {
	service       WorkloadIdentityManagement
	sessions      SessionAuthenticator
	authorization Authorizer
	scopes        ResultScopeResolver
	now           func() time.Time
	operations    map[string]managementapi.OperationContract
}

func NewWorkloadIdentityRoutes(options WorkloadIdentityRoutesOptions) (*WorkloadIdentityRoutes, error) {
	scopes := configuredResultScopes(options.Scopes, options.Authorization)
	if options.Service == nil || options.Sessions == nil || options.Authorization == nil || scopes == nil {
		return nil, errors.New("management workload identity routes require service, session, authorization, and result-scope dependencies")
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	routes := &WorkloadIdentityRoutes{
		service: options.Service, sessions: options.Sessions, authorization: options.Authorization,
		scopes: scopes, now: now, operations: make(map[string]managementapi.OperationContract),
	}
	for _, contract := range workloadIdentityHTTPContracts() {
		operation, found := managementapi.LookupOperation(contract.method, contract.path)
		if !found {
			return nil, fmt.Errorf("management workload identity operation %s %s is unavailable", contract.method, contract.path)
		}
		routes.operations[string(contract.method)+" "+contract.path] = operation
	}
	return routes, nil
}

func (routes *WorkloadIdentityRoutes) Register(mux *http.ServeMux) {
	if routes == nil || mux == nil {
		panic("Management workload identity routes and mux are required")
	}
	mux.HandleFunc("GET "+serviceAccountsPath, routes.listServiceAccounts)
	mux.HandleFunc("POST "+serviceAccountsPath, routes.createServiceAccount)
	mux.HandleFunc("GET "+serviceAccountsPath+"/{serviceAccountId}", routes.getServiceAccount)
	mux.HandleFunc("PATCH "+serviceAccountsPath+"/{serviceAccountId}", routes.patchServiceAccount)
	mux.HandleFunc("DELETE "+serviceAccountsPath+"/{serviceAccountId}", routes.deleteServiceAccount)
	mux.HandleFunc("GET "+serviceAccountsPath+"/{serviceAccountId}/credentials", routes.listServiceCredentials)
	mux.HandleFunc("POST "+serviceAccountsPath+"/{serviceAccountId}/credentials:rotate", routes.rotateServiceCredential)
	mux.HandleFunc("DELETE "+serviceAccountsPath+"/{serviceAccountId}/credentials/{credentialId}", routes.revokeServiceCredential)
	mux.HandleFunc("GET "+mtlsMappingsPath, routes.listMTLSMappings)
	mux.HandleFunc("POST "+mtlsMappingsPath, routes.createMTLSMapping)
	mux.HandleFunc("GET "+mtlsMappingsPath+"/{mappingId}", routes.getMTLSMapping)
	mux.HandleFunc("PATCH "+mtlsMappingsPath+"/{mappingId}", routes.patchMTLSMapping)
	mux.HandleFunc("DELETE "+mtlsMappingsPath+"/{mappingId}", routes.deleteMTLSMapping)
}

func (routes *WorkloadIdentityRoutes) Ready(ctx context.Context) error {
	if routes == nil || routes.service == nil {
		return managementidentity.ErrWorkloadUnavailable
	}
	return routes.service.Ready(ctx)
}

func (routes *WorkloadIdentityRoutes) listServiceAccounts(response http.ResponseWriter, request *http.Request) {
	requestID := workloadRequest(response, request)
	query, ok := workloadListQuery(response, request, requestID, "namespaceId", "status")
	if !ok {
		return
	}
	namespaceID := request.URL.Query().Get("namespaceId")
	if namespaceID != "" && !canonicalUUID(namespaceID) {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "namespaceId is invalid.", requestID)
		return
	}
	session, ok := routes.authenticate(response, request, requestID, namespaceID)
	if !ok {
		return
	}
	operation := routes.operation(managementapi.MethodGET, serviceAccountsPath)
	scope := managementidentity.ServiceAccountResultScope{Cluster: namespaceID == "", All: namespaceID == ""}
	if namespaceID == "" {
		if !routes.authorize(response, request, requestID, session, "", operation, nil, false) {
			return
		}
	} else {
		permission, valid := listPermission(operation)
		if !valid {
			writeProviderError(response, http.StatusServiceUnavailable, "authorization_unavailable", "Authorization state is unavailable.", requestID)
			return
		}
		resultScope, err := resolveListResultScope(request.Context(), routes.scopes, session, namespaceID, permission)
		if err != nil {
			writeResultScopeError(response, err, requestID)
			return
		}
		canonical, err := resultScope.Canonical()
		if err != nil || string(canonical.NamespaceID) != namespaceID {
			writeProviderError(response, http.StatusServiceUnavailable, "authorization_unavailable", "Authorization state is unavailable.", requestID)
			return
		}
		scope.NamespaceID, scope.All = namespaceID, canonical.All
		for _, id := range canonical.IDs(accesscontrol.ScopeResourceServiceAccount) {
			scope.IDs = append(scope.IDs, string(id))
		}
	}
	page, err := routes.service.ListServiceAccounts(request.Context(), managementidentity.ServiceAccountListRequest{
		Scope: scope, Status: managementidentity.ServiceAccountStatus(request.URL.Query().Get("status")),
		Cursor: query.cursor, PageSize: query.pageSize,
	})
	if err != nil {
		writeWorkloadIdentityError(response, err, requestID)
		return
	}
	data := make([]managementapi.ServiceAccount, len(page.Items))
	for index := range page.Items {
		data[index] = serviceAccountDTO(page.Items[index])
	}
	writeProviderJSON(response, http.StatusOK, managementapi.Page[managementapi.ServiceAccount]{
		Data: data, Page: managementapi.PageInfo{NextCursor: page.NextCursor, HasMore: page.HasMore, PageSize: page.PageSize},
	}, requestID)
}

func (routes *WorkloadIdentityRoutes) createServiceAccount(response http.ResponseWriter, request *http.Request) {
	requestID := workloadRequest(response, request)
	if !workloadNoQuery(response, request, requestID) {
		return
	}
	idempotencyKey, ok := requireIdempotencyKey(response, request, requestID)
	if !ok {
		return
	}
	var body managementapi.ServiceAccountCreateRequest
	if !decodeIdentityBody(response, request, requestID, &body) {
		return
	}
	owner := managementidentity.ServiceAccountOwnerScope(body.OwnerScope)
	namespaceID := body.NamespaceID
	if (owner == managementidentity.ServiceAccountOwnerCluster && namespaceID != "") ||
		(owner == managementidentity.ServiceAccountOwnerNamespace && !canonicalUUID(namespaceID)) {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Service account owner scope is invalid.", requestID)
		return
	}
	session, ok := routes.authenticate(response, request, requestID, namespaceID)
	if !ok || !routes.authorize(response, request, requestID, session, namespaceID,
		routes.operation(managementapi.MethodPOST, serviceAccountsPath), workloadCreateTarget(namespaceID), false) {
		return
	}
	result, err := routes.service.CreateServiceAccount(request.Context(), managementidentity.CreateServiceAccountRequest{
		DisplayName: body.DisplayName, OwnerScope: owner, NamespaceID: namespaceID,
		CredentialExpiresAt: body.CredentialExpiresAt, CredentialClass: managementidentity.WorkloadClass(body.CredentialClass),
		IdempotencyKey: string(idempotencyKey), Actor: workloadActor(request, session, requestID, body.Reason),
	})
	if err != nil {
		writeWorkloadIdentityError(response, err, requestID)
		return
	}
	response.Header().Set("Location", serviceAccountsPath+"/"+result.ServiceAccount.ID)
	writeServiceCredentialIssue(response, http.StatusCreated, result, requestID)
}

func (routes *WorkloadIdentityRoutes) getServiceAccount(response http.ResponseWriter, request *http.Request) {
	requestID := workloadRequest(response, request)
	account, _, ok := routes.authorizedServiceAccount(response, request, requestID,
		routes.operation(managementapi.MethodGET, serviceAccountsPath+"/{serviceAccountId}"))
	if !ok {
		return
	}
	response.Header().Set(managementapi.HeaderETag, serviceAccountETag(account.Revision))
	writeProviderJSON(response, http.StatusOK, map[string]any{"data": serviceAccountDTO(account)}, requestID)
}

func (routes *WorkloadIdentityRoutes) patchServiceAccount(response http.ResponseWriter, request *http.Request) {
	requestID := workloadRequest(response, request)
	account, session, ok := routes.authorizedServiceAccount(response, request, requestID,
		routes.operation(managementapi.MethodPATCH, serviceAccountsPath+"/{serviceAccountId}"))
	if !ok {
		return
	}
	revision, ok := workloadRevision(response, request, requestID, "sa")
	if !ok {
		return
	}
	var body managementapi.ServiceAccountPatchRequest
	if !decodeIdentityBody(response, request, requestID, &body) {
		return
	}
	var status *managementidentity.ServiceAccountStatus
	if body.Status != nil {
		value := managementidentity.ServiceAccountStatus(*body.Status)
		status = &value
	}
	result, err := routes.service.PatchServiceAccount(request.Context(), managementidentity.PatchServiceAccountRequest{
		ID: account.ID, ExpectedRevision: revision, DisplayName: body.DisplayName, Status: status,
		Actor: workloadActor(request, session, requestID, body.Reason),
	})
	if err != nil {
		writeWorkloadIdentityError(response, err, requestID)
		return
	}
	writeWorkloadMutation(response, result, serviceAccountETag, serviceAccountsPath, requestID)
}

func (routes *WorkloadIdentityRoutes) deleteServiceAccount(response http.ResponseWriter, request *http.Request) {
	requestID := workloadRequest(response, request)
	if !noRequestBody(response, request, requestID) {
		return
	}
	account, session, ok := routes.authorizedServiceAccount(response, request, requestID,
		routes.operation(managementapi.MethodDELETE, serviceAccountsPath+"/{serviceAccountId}"))
	if !ok {
		return
	}
	revision, ok := workloadRevision(response, request, requestID, "sa")
	if !ok {
		return
	}
	result, err := routes.service.DeleteServiceAccount(request.Context(), managementidentity.DeleteServiceAccountRequest{
		ID: account.ID, ExpectedRevision: revision,
		Actor: workloadActor(request, session, requestID, "Delete service account"),
	})
	if err != nil {
		writeWorkloadIdentityError(response, err, requestID)
		return
	}
	response.Header().Set(managementapi.HeaderETag, serviceAccountETag(result.Revision))
	response.WriteHeader(http.StatusNoContent)
}

func (routes *WorkloadIdentityRoutes) listServiceCredentials(response http.ResponseWriter, request *http.Request) {
	requestID := workloadRequest(response, request)
	query, ok := workloadListQuery(response, request, requestID)
	if !ok {
		return
	}
	account, _, ok := routes.authorizedServiceAccount(response, request, requestID,
		routes.operation(managementapi.MethodGET, serviceAccountsPath+"/{serviceAccountId}/credentials"))
	if !ok {
		return
	}
	page, err := routes.service.ListServiceCredentials(request.Context(), managementidentity.ServiceCredentialListRequest{
		ServiceAccountID: account.ID, Cursor: query.cursor, PageSize: query.pageSize,
	})
	if err != nil {
		writeWorkloadIdentityError(response, err, requestID)
		return
	}
	data := make([]managementapi.ServiceCredential, len(page.Items))
	for index := range page.Items {
		data[index] = serviceCredentialDTO(page.Items[index])
	}
	writeProviderJSON(response, http.StatusOK, managementapi.Page[managementapi.ServiceCredential]{
		Data: data, Page: managementapi.PageInfo{NextCursor: page.NextCursor, HasMore: page.HasMore, PageSize: page.PageSize},
	}, requestID)
}

func (routes *WorkloadIdentityRoutes) rotateServiceCredential(response http.ResponseWriter, request *http.Request) {
	requestID := workloadRequest(response, request)
	if !workloadNoQuery(response, request, requestID) {
		return
	}
	account, session, ok := routes.authorizedServiceAccount(response, request, requestID,
		routes.operation(managementapi.MethodPOST, serviceAccountsPath+"/{serviceAccountId}/credentials:rotate"))
	if !ok {
		return
	}
	revision, ok := workloadRevision(response, request, requestID, "sa")
	if !ok {
		return
	}
	idempotencyKey, ok := requireIdempotencyKey(response, request, requestID)
	if !ok {
		return
	}
	var body managementapi.ServiceCredentialRotateRequest
	if !decodeIdentityBody(response, request, requestID, &body) {
		return
	}
	if body.OverlapSeconds < 0 {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Credential overlap is invalid.", requestID)
		return
	}
	result, err := routes.service.RotateServiceCredential(request.Context(), managementidentity.RotateServiceCredentialRequest{
		ServiceAccountID: account.ID, ExpectedRevision: revision, ExpiresAt: body.ExpiresAt,
		WorkloadClass: managementidentity.WorkloadClass(body.WorkloadClass), Overlap: time.Duration(body.OverlapSeconds) * time.Second,
		IdempotencyKey: string(idempotencyKey), Actor: workloadActor(request, session, requestID, body.Reason),
	})
	if err != nil {
		writeWorkloadIdentityError(response, err, requestID)
		return
	}
	writeServiceCredentialIssue(response, http.StatusOK, result, requestID)
}

func (routes *WorkloadIdentityRoutes) revokeServiceCredential(response http.ResponseWriter, request *http.Request) {
	requestID := workloadRequest(response, request)
	if !noRequestBody(response, request, requestID) {
		return
	}
	account, session, ok := routes.authorizedServiceAccount(response, request, requestID,
		routes.operation(managementapi.MethodDELETE, serviceAccountsPath+"/{serviceAccountId}/credentials/{credentialId}"))
	if !ok {
		return
	}
	credentialID := request.PathValue("credentialId")
	if !canonicalUUID(credentialID) {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	revision, ok := workloadRevision(response, request, requestID, "sa")
	if !ok {
		return
	}
	result, err := routes.service.RevokeServiceCredential(request.Context(), managementidentity.RevokeServiceCredentialRequest{
		ServiceAccountID: account.ID, CredentialID: credentialID, ExpectedRevision: revision,
		Actor: workloadActor(request, session, requestID, "Revoke service credential"),
	})
	if err != nil {
		writeWorkloadIdentityError(response, err, requestID)
		return
	}
	response.Header().Set(managementapi.HeaderETag, serviceAccountETag(result.Revision))
	response.WriteHeader(http.StatusNoContent)
}

func (routes *WorkloadIdentityRoutes) listMTLSMappings(response http.ResponseWriter, request *http.Request) {
	requestID := workloadRequest(response, request)
	query, ok := workloadListQuery(response, request, requestID, "status")
	if !ok {
		return
	}
	session, ok := routes.authenticate(response, request, requestID, "")
	if !ok || !routes.authorize(response, request, requestID, session, "",
		routes.operation(managementapi.MethodGET, mtlsMappingsPath), nil, false) {
		return
	}
	page, err := routes.service.ListMTLSMappings(request.Context(), managementidentity.MTLSMappingListRequest{
		Status: managementauth.ResourceStatus(request.URL.Query().Get("status")), Cursor: query.cursor, PageSize: query.pageSize,
	})
	if err != nil {
		writeWorkloadIdentityError(response, err, requestID)
		return
	}
	data := make([]managementapi.MTLSIdentityMapping, len(page.Items))
	for index := range page.Items {
		data[index] = mtlsMappingDTO(page.Items[index])
	}
	writeProviderJSON(response, http.StatusOK, managementapi.Page[managementapi.MTLSIdentityMapping]{
		Data: data, Page: managementapi.PageInfo{NextCursor: page.NextCursor, HasMore: page.HasMore, PageSize: page.PageSize},
	}, requestID)
}

func (routes *WorkloadIdentityRoutes) createMTLSMapping(response http.ResponseWriter, request *http.Request) {
	requestID := workloadRequest(response, request)
	if !workloadNoQuery(response, request, requestID) {
		return
	}
	session, ok := routes.authenticate(response, request, requestID, "")
	if !ok || !routes.authorize(response, request, requestID, session, "",
		routes.operation(managementapi.MethodPOST, mtlsMappingsPath), nil, false) {
		return
	}
	idempotencyKey, ok := requireIdempotencyKey(response, request, requestID)
	if !ok {
		return
	}
	var body managementapi.MTLSIdentityMappingCreateRequest
	if !decodeIdentityBody(response, request, requestID, &body) {
		return
	}
	result, err := routes.service.CreateMTLSMapping(request.Context(), managementidentity.CreateMTLSMappingRequest{
		MatcherKind: managementidentity.MTLSMatcherKind(body.MatcherKind), MatcherValue: body.MatcherValue,
		PrincipalID: body.PrincipalID, WorkloadClass: managementidentity.WorkloadClass(body.WorkloadClass),
		IdempotencyKey: string(idempotencyKey), Actor: workloadActor(request, session, requestID, body.Reason),
	})
	if err != nil {
		writeWorkloadIdentityError(response, err, requestID)
		return
	}
	writeWorkloadMutation(response, result, mtlsMappingETag, mtlsMappingsPath, requestID)
}

func (routes *WorkloadIdentityRoutes) getMTLSMapping(response http.ResponseWriter, request *http.Request) {
	requestID := workloadRequest(response, request)
	mapping, _, ok := routes.authorizedMTLSMapping(response, request, requestID,
		routes.operation(managementapi.MethodGET, mtlsMappingsPath+"/{mappingId}"))
	if !ok {
		return
	}
	response.Header().Set(managementapi.HeaderETag, mtlsMappingETag(mapping.Revision))
	writeProviderJSON(response, http.StatusOK, map[string]any{"data": mtlsMappingDTO(mapping)}, requestID)
}

func (routes *WorkloadIdentityRoutes) patchMTLSMapping(response http.ResponseWriter, request *http.Request) {
	requestID := workloadRequest(response, request)
	mapping, session, ok := routes.authorizedMTLSMapping(response, request, requestID,
		routes.operation(managementapi.MethodPATCH, mtlsMappingsPath+"/{mappingId}"))
	if !ok {
		return
	}
	revision, ok := workloadRevision(response, request, requestID, "mtls")
	if !ok {
		return
	}
	var body managementapi.MTLSIdentityMappingPatchRequest
	if !decodeIdentityBody(response, request, requestID, &body) {
		return
	}
	var status *managementauth.ResourceStatus
	if body.Status != nil {
		value := managementauth.ResourceStatus(*body.Status)
		status = &value
	}
	var class *managementidentity.WorkloadClass
	if body.WorkloadClass != nil {
		value := managementidentity.WorkloadClass(*body.WorkloadClass)
		class = &value
	}
	result, err := routes.service.PatchMTLSMapping(request.Context(), managementidentity.PatchMTLSMappingRequest{
		ID: mapping.ID, ExpectedRevision: revision, Status: status, WorkloadClass: class,
		Actor: workloadActor(request, session, requestID, body.Reason),
	})
	if err != nil {
		writeWorkloadIdentityError(response, err, requestID)
		return
	}
	writeWorkloadMutation(response, result, mtlsMappingETag, mtlsMappingsPath, requestID)
}

func (routes *WorkloadIdentityRoutes) deleteMTLSMapping(response http.ResponseWriter, request *http.Request) {
	requestID := workloadRequest(response, request)
	if !noRequestBody(response, request, requestID) {
		return
	}
	mapping, session, ok := routes.authorizedMTLSMapping(response, request, requestID,
		routes.operation(managementapi.MethodDELETE, mtlsMappingsPath+"/{mappingId}"))
	if !ok {
		return
	}
	revision, ok := workloadRevision(response, request, requestID, "mtls")
	if !ok {
		return
	}
	result, err := routes.service.DeleteMTLSMapping(request.Context(), managementidentity.DeleteMTLSMappingRequest{
		ID: mapping.ID, ExpectedRevision: revision,
		Actor: workloadActor(request, session, requestID, "Delete mTLS identity mapping"),
	})
	if err != nil {
		writeWorkloadIdentityError(response, err, requestID)
		return
	}
	response.Header().Set(managementapi.HeaderETag, mtlsMappingETag(result.Revision))
	response.WriteHeader(http.StatusNoContent)
}

func (routes *WorkloadIdentityRoutes) authorizedServiceAccount(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
	operation managementapi.OperationContract,
) (managementidentity.ServiceAccount, managementauth.AuthenticatedSession, bool) {
	allowQuery := operation.Method == managementapi.MethodGET &&
		operation.Path == serviceAccountsPath+"/{serviceAccountId}/credentials"
	if request.URL.RawQuery != "" && !allowQuery {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Query parameters are not accepted.", requestID)
		return managementidentity.ServiceAccount{}, managementauth.AuthenticatedSession{}, false
	}
	id := request.PathValue("serviceAccountId")
	if !canonicalUUID(id) {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return managementidentity.ServiceAccount{}, managementauth.AuthenticatedSession{}, false
	}
	account, err := routes.service.GetServiceAccount(request.Context(), id)
	if err != nil {
		writeWorkloadIdentityError(response, err, requestID)
		return managementidentity.ServiceAccount{}, managementauth.AuthenticatedSession{}, false
	}
	namespaceID := account.NamespaceID
	session, ok := routes.authenticate(response, request, requestID, namespaceID)
	if !ok || !routes.authorize(response, request, requestID, session, namespaceID,
		operation, serviceAccountTarget(account), true) {
		return managementidentity.ServiceAccount{}, managementauth.AuthenticatedSession{}, false
	}
	return account, session, true
}

func (routes *WorkloadIdentityRoutes) authorizedMTLSMapping(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
	operation managementapi.OperationContract,
) (managementidentity.MTLSIdentityMapping, managementauth.AuthenticatedSession, bool) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Query parameters are not accepted.", requestID)
		return managementidentity.MTLSIdentityMapping{}, managementauth.AuthenticatedSession{}, false
	}
	id := request.PathValue("mappingId")
	if !canonicalUUID(id) {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return managementidentity.MTLSIdentityMapping{}, managementauth.AuthenticatedSession{}, false
	}
	mapping, err := routes.service.GetMTLSMapping(request.Context(), id)
	if err != nil {
		writeWorkloadIdentityError(response, err, requestID)
		return managementidentity.MTLSIdentityMapping{}, managementauth.AuthenticatedSession{}, false
	}
	session, ok := routes.authenticate(response, request, requestID, "")
	if !ok || !routes.authorize(response, request, requestID, session, "", operation, nil, true) {
		return managementidentity.MTLSIdentityMapping{}, managementauth.AuthenticatedSession{}, false
	}
	return mapping, session, true
}

func (routes *WorkloadIdentityRoutes) authenticate(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
	namespaceID string,
) (managementauth.AuthenticatedSession, bool) {
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
	if session.NamespaceID != namespaceID {
		writeProviderError(response, http.StatusServiceUnavailable, "authentication_unavailable", "Authentication state is unavailable.", requestID)
		return managementauth.AuthenticatedSession{}, false
	}
	return session, true
}

func (routes *WorkloadIdentityRoutes) authorize(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
	session managementauth.AuthenticatedSession,
	namespaceID string,
	operation managementapi.OperationContract,
	targets map[string][]accesscontrol.ScopedTarget,
	nondisclosing bool,
) bool {
	_, err := routes.authorization.Authorize(request.Context(), AuthorizationRequest{
		Operation: operation, Session: session, NamespaceID: namespaceID, Targets: targets,
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

func (routes *WorkloadIdentityRoutes) operation(method managementapi.HTTPMethod, path string) managementapi.OperationContract {
	return routes.operations[string(method)+" "+path]
}
