package managementserver

import (
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policymanagement"
)

func (routes *PolicyRoutes) listAccessPolicies(response http.ResponseWriter, request *http.Request, requestID string) {
	cursor, pageSize, status, search, ok := policyListQuery(response, request, requestID)
	if !ok {
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	operation := routes.operation(managementapi.MethodGET, accessPoliciesPath)
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
	page, err := routes.service.ListAccessPolicies(request.Context(), policymanagement.ListPoliciesRequest{
		NamespaceID: namespaceID, Status: accesscontrol.PolicyStatus(status), Search: search,
		Cursor: cursor, PageSize: pageSize,
		Scope: scope,
	})
	if err != nil {
		writePolicyError(response, err, requestID)
		return
	}
	writeProviderJSON(response, http.StatusOK, newAccessPolicyPage(page), requestID)
}

func (routes *PolicyRoutes) createAccessPolicy(response http.ResponseWriter, request *http.Request, requestID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Access policy create does not accept query parameters.", requestID)
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
	var body managementapi.AccessPolicyCreateRequest
	if !decodePolicyBody(response, request, requestID, &body) {
		return
	}
	grants := policyGrants(body.Grants)
	if !validPolicyGrantTargets(grants) {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Access policy request is invalid.", requestID)
		return
	}
	if !routes.authorize(response, request, requestID, session, namespaceID,
		routes.operation(managementapi.MethodPOST, accessPoliciesPath),
		accessPolicyTargets(namespaceID, "", grants), accessPolicyConditions(grants), false) {
		return
	}
	result, err := routes.service.CreateAccessPolicy(request.Context(), policymanagement.CreateAccessPolicyRequest{
		NamespaceID: namespaceID, Name: body.Name, Description: body.Description,
		Status: accesscontrol.PolicyStatus(body.Status),
		Grants: grants, IdempotencyKey: string(idempotencyKey), Actor: routes.actor(request, session, requestID),
	})
	if err != nil {
		writePolicyError(response, err, requestID)
		return
	}
	response.Header().Set("Location", accessPoliciesPath+"/"+result.ID)
	writePolicyMutation(response, result, requestID, policyETagAccess, true)
}

func (routes *PolicyRoutes) accessPolicyResource(response http.ResponseWriter, request *http.Request, requestID, policyID string) {
	switch request.Method {
	case http.MethodGet:
		routes.getAccessPolicy(response, request, requestID, policyID)
	case http.MethodPatch:
		routes.patchAccessPolicy(response, request, requestID, policyID)
	case http.MethodDelete:
		routes.deleteAccessPolicy(response, request, requestID, policyID)
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
}

func (routes *PolicyRoutes) getAccessPolicy(response http.ResponseWriter, request *http.Request, requestID, policyID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Access policy detail does not accept query parameters.", requestID)
		return
	}
	policy, _, ok := routes.accessPolicyForAuthorizedRequest(response, request, requestID, policyID,
		routes.operation(managementapi.MethodGET, accessPoliciesPath+"/{policyId}"), true)
	if !ok {
		return
	}
	response.Header().Set(managementapi.HeaderETag, policyResourceETag(policyETagAccess, policy.Revision))
	writeProviderJSON(response, http.StatusOK, managementapi.AccessPolicyDetail{Data: newAccessPolicy(policy)}, requestID)
}

func (routes *PolicyRoutes) patchAccessPolicy(response http.ResponseWriter, request *http.Request, requestID, policyID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Access policy update does not accept query parameters.", requestID)
		return
	}
	operation := routes.operation(managementapi.MethodPATCH, accessPoliciesPath+"/{policyId}")
	policy, session, ok := routes.accessPolicyForAuthorizedRequest(response, request, requestID, policyID,
		operation, true)
	if !ok {
		return
	}
	revision, ok := requirePolicyRevision(response, request, requestID, policyETagAccess)
	if !ok {
		return
	}
	var body managementapi.AccessPolicyPatchRequest
	if !decodePolicyBody(response, request, requestID, &body) {
		return
	}
	grants := policy.Grants
	if body.Grants != nil {
		grants = policyGrants(*body.Grants)
	}
	if !validPolicyGrantTargets(grants) {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Access policy request is invalid.", requestID)
		return
	}
	if !routes.authorize(response, request, requestID, session, policy.NamespaceID, operation,
		accessPolicyTargets(policy.NamespaceID, policy.ID, grants), accessPolicyConditions(grants), false) {
		return
	}
	var updatedGrants *[]policymanagement.AccessGrant
	if body.Grants != nil {
		updatedGrants = &grants
	}
	result, err := routes.service.UpdateAccessPolicy(request.Context(), policymanagement.UpdateAccessPolicyRequest{
		NamespaceID: policy.NamespaceID, PolicyID: policy.ID, ExpectedRevision: revision,
		Name: body.Name, Description: body.Description, Status: policyStatusPointer(body.Status), Grants: updatedGrants,
		Actor: routes.actor(request, session, requestID),
	})
	if err != nil {
		writePolicyError(response, err, requestID)
		return
	}
	writePolicyMutation(response, result, requestID, policyETagAccess, false)
}

func (routes *PolicyRoutes) deleteAccessPolicy(response http.ResponseWriter, request *http.Request, requestID, policyID string) {
	if request.URL.RawQuery != "" || !noRequestBody(response, request, requestID) {
		return
	}
	policy, session, ok := routes.accessPolicyForAuthorizedRequest(response, request, requestID, policyID,
		routes.operation(managementapi.MethodDELETE, accessPoliciesPath+"/{policyId}"), true)
	if !ok {
		return
	}
	revision, ok := requirePolicyRevision(response, request, requestID, policyETagAccess)
	if !ok {
		return
	}
	result, err := routes.service.DeleteAccessPolicy(request.Context(), policymanagement.DeletePolicyRequest{
		NamespaceID: policy.NamespaceID, PolicyID: policy.ID, ExpectedRevision: revision,
		Actor: routes.actor(request, session, requestID),
	})
	if err != nil {
		writePolicyError(response, err, requestID)
		return
	}
	writePolicyMutation(response, result, requestID, policyETagAccess, false)
}

func (routes *PolicyRoutes) accessPolicyForAuthorizedRequest(response http.ResponseWriter, request *http.Request,
	requestID, policyID string, operation managementapi.OperationContract, nondisclosing bool,
) (policymanagement.AccessPolicy, managementauth.AuthenticatedSession, bool) {
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return policymanagement.AccessPolicy{}, managementauth.AuthenticatedSession{}, false
	}
	policy, err := routes.service.GetAccessPolicy(request.Context(), namespaceID, policyID)
	if err != nil {
		writePolicyError(response, err, requestID)
		return policymanagement.AccessPolicy{}, managementauth.AuthenticatedSession{}, false
	}
	if !routes.authorize(response, request, requestID, session, namespaceID, operation,
		accessPolicyTargets(namespaceID, policy.ID, nil), accessPolicyConditions(nil), nondisclosing) {
		return policymanagement.AccessPolicy{}, managementauth.AuthenticatedSession{}, false
	}
	return policy, session, true
}
