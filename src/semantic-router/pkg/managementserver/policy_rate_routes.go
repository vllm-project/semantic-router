package managementserver

import (
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policymanagement"
)

func (routes *PolicyRoutes) listRatePolicies(response http.ResponseWriter, request *http.Request, requestID string) {
	cursor, pageSize, status, search, ok := policyListQuery(response, request, requestID)
	if !ok {
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	operation := routes.operation(managementapi.MethodGET, ratePoliciesPath)
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
	page, err := routes.service.ListRateLimitPolicies(request.Context(), policymanagement.ListPoliciesRequest{
		NamespaceID: namespaceID, Status: accesscontrol.PolicyStatus(status), Search: search,
		Cursor: cursor, PageSize: pageSize,
		Scope: scope,
	})
	if err != nil {
		writePolicyError(response, err, requestID)
		return
	}
	writeProviderJSON(response, http.StatusOK, newRateLimitPolicyPage(page), requestID)
}

func (routes *PolicyRoutes) createRatePolicy(response http.ResponseWriter, request *http.Request, requestID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Rate limit policy create does not accept query parameters.", requestID)
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
	var body managementapi.RateLimitPolicyCreateRequest
	if !decodePolicyBody(response, request, requestID, &body) {
		return
	}
	rules, err := policyRules(body.Rules)
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Rate limit policy request is invalid.", requestID)
		return
	}
	if !routes.authorize(response, request, requestID, session, namespaceID,
		routes.operation(managementapi.MethodPOST, ratePoliciesPath),
		map[string][]accesscontrol.ScopedTarget{"policy": {policyTarget(namespaceID, accesscontrol.ScopeResourceRateLimitPolicy, "")}}, nil, false) {
		return
	}
	result, err := routes.service.CreateRateLimitPolicy(request.Context(), policymanagement.CreateRateLimitPolicyRequest{
		NamespaceID: namespaceID, Name: body.Name, Description: body.Description,
		Status: accesscontrol.PolicyStatus(body.Status), Rules: rules, IdempotencyKey: string(idempotencyKey),
		Actor: routes.actor(request, session, requestID),
	})
	if err != nil {
		writePolicyError(response, err, requestID)
		return
	}
	response.Header().Set("Location", ratePoliciesPath+"/"+result.ID)
	writePolicyMutation(response, result, requestID, policyETagRate, true)
}

func (routes *PolicyRoutes) ratePolicyResource(response http.ResponseWriter, request *http.Request, requestID, policyID string) {
	switch request.Method {
	case http.MethodGet:
		routes.getRatePolicy(response, request, requestID, policyID)
	case http.MethodPatch:
		routes.patchRatePolicy(response, request, requestID, policyID)
	case http.MethodDelete:
		routes.deleteRatePolicy(response, request, requestID, policyID)
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
}

func (routes *PolicyRoutes) getRatePolicy(response http.ResponseWriter, request *http.Request, requestID, policyID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Rate limit policy detail does not accept query parameters.", requestID)
		return
	}
	policy, _, ok := routes.ratePolicyForAuthorizedRequest(response, request, requestID, policyID,
		routes.operation(managementapi.MethodGET, ratePoliciesPath+"/{policyId}"), true)
	if !ok {
		return
	}
	response.Header().Set(managementapi.HeaderETag, policyResourceETag(policyETagRate, policy.Revision))
	writeProviderJSON(response, http.StatusOK, managementapi.RateLimitPolicyDetail{Data: newRateLimitPolicy(policy)}, requestID)
}

func (routes *PolicyRoutes) patchRatePolicy(response http.ResponseWriter, request *http.Request, requestID, policyID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Rate limit policy update does not accept query parameters.", requestID)
		return
	}
	policy, session, ok := routes.ratePolicyForAuthorizedRequest(response, request, requestID, policyID,
		routes.operation(managementapi.MethodPATCH, ratePoliciesPath+"/{policyId}"), true)
	if !ok {
		return
	}
	revision, ok := requirePolicyRevision(response, request, requestID, policyETagRate)
	if !ok {
		return
	}
	var body managementapi.RateLimitPolicyPatchRequest
	if !decodePolicyBody(response, request, requestID, &body) {
		return
	}
	var rules *[]policymanagement.RateLimitRule
	if body.Rules != nil {
		value, err := policyRules(*body.Rules)
		if err != nil {
			writeProviderError(response, http.StatusBadRequest, "invalid_request", "Rate limit policy request is invalid.", requestID)
			return
		}
		rules = &value
	}
	result, err := routes.service.UpdateRateLimitPolicy(request.Context(), policymanagement.UpdateRateLimitPolicyRequest{
		NamespaceID: policy.NamespaceID, PolicyID: policy.ID, ExpectedRevision: revision,
		Name: body.Name, Description: body.Description, Status: policyStatusPointer(body.Status), Rules: rules,
		Actor: routes.actor(request, session, requestID),
	})
	if err != nil {
		writePolicyError(response, err, requestID)
		return
	}
	writePolicyMutation(response, result, requestID, policyETagRate, false)
}

func (routes *PolicyRoutes) deleteRatePolicy(response http.ResponseWriter, request *http.Request, requestID, policyID string) {
	if request.URL.RawQuery != "" || !noRequestBody(response, request, requestID) {
		return
	}
	policy, session, ok := routes.ratePolicyForAuthorizedRequest(response, request, requestID, policyID,
		routes.operation(managementapi.MethodDELETE, ratePoliciesPath+"/{policyId}"), true)
	if !ok {
		return
	}
	revision, ok := requirePolicyRevision(response, request, requestID, policyETagRate)
	if !ok {
		return
	}
	result, err := routes.service.DeleteRateLimitPolicy(request.Context(), policymanagement.DeletePolicyRequest{
		NamespaceID: policy.NamespaceID, PolicyID: policy.ID, ExpectedRevision: revision,
		Actor: routes.actor(request, session, requestID),
	})
	if err != nil {
		writePolicyError(response, err, requestID)
		return
	}
	writePolicyMutation(response, result, requestID, policyETagRate, false)
}

func (routes *PolicyRoutes) ratePolicyForAuthorizedRequest(response http.ResponseWriter, request *http.Request,
	requestID, policyID string, operation managementapi.OperationContract, nondisclosing bool,
) (policymanagement.RateLimitPolicy, managementauth.AuthenticatedSession, bool) {
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return policymanagement.RateLimitPolicy{}, managementauth.AuthenticatedSession{}, false
	}
	policy, err := routes.service.GetRateLimitPolicy(request.Context(), namespaceID, policyID)
	if err != nil {
		writePolicyError(response, err, requestID)
		return policymanagement.RateLimitPolicy{}, managementauth.AuthenticatedSession{}, false
	}
	if !routes.authorize(response, request, requestID, session, namespaceID, operation,
		map[string][]accesscontrol.ScopedTarget{"policy": {policyTarget(namespaceID, accesscontrol.ScopeResourceRateLimitPolicy, policy.ID)}},
		nil, nondisclosing) {
		return policymanagement.RateLimitPolicy{}, managementauth.AuthenticatedSession{}, false
	}
	return policy, session, true
}
