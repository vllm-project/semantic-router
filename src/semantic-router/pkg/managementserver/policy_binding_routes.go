package managementserver

import (
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policybulk"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policymanagement"
)

const singleBindingValidationItemID = "00000000-0000-4000-8000-000000000002"

func (routes *PolicyRoutes) listAccessBindings(response http.ResponseWriter, request *http.Request, requestID string) {
	input, ok := policyBindingListQuery(response, request, requestID, false)
	if !ok {
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	input.NamespaceID = namespaceID
	operation := routes.operation(managementapi.MethodGET, accessBindingsPath)
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
	input.Scope = scope
	page, err := routes.service.ListAccessBindings(request.Context(), input)
	if err != nil {
		writePolicyError(response, err, requestID)
		return
	}
	writeProviderJSON(response, http.StatusOK, newAccessPolicyBindingPage(page), requestID)
}

func (routes *PolicyRoutes) createAccessBinding(response http.ResponseWriter, request *http.Request, requestID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Access policy binding create does not accept query parameters.", requestID)
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
	var body managementapi.AccessPolicyBindingCreateRequest
	if !decodePolicyBody(response, request, requestID, &body) {
		return
	}
	subject := policymanagement.Subject{Type: accesscontrol.SubjectKind(body.Subject.Type), ID: body.Subject.ID}
	if policybulk.ValidateAccessItems([]policybulk.AccessBindingItem{{
		ItemID: singleBindingValidationItemID, PolicyID: body.PolicyID, Subject: subject,
	}}) != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Access policy binding request is invalid.", requestID)
		return
	}
	targets, conditions, valid := policyBindingTargets(namespaceID, body.PolicyID, subject, false)
	if !valid || !routes.authorize(response, request, requestID, session, namespaceID,
		routes.operation(managementapi.MethodPOST, accessBindingsPath), targets, conditions, false) {
		return
	}
	result, err := routes.service.CreateAccessBinding(request.Context(), policymanagement.CreateAccessBindingRequest{
		NamespaceID: namespaceID, PolicyID: body.PolicyID, Subject: subject,
		IdempotencyKey: string(idempotencyKey), Actor: routes.actor(request, session, requestID),
	})
	if err != nil {
		writePolicyError(response, err, requestID)
		return
	}
	response.Header().Set("Location", accessBindingsPath+"/"+result.ID)
	writePolicyMutation(response, result, requestID, policyETagAccessBinding, true)
}

func (routes *PolicyRoutes) accessBindingResource(response http.ResponseWriter, request *http.Request, requestID, bindingID string) {
	switch request.Method {
	case http.MethodGet:
		routes.getAccessBinding(response, request, requestID, bindingID)
	case http.MethodPatch:
		routes.patchAccessBinding(response, request, requestID, bindingID)
	case http.MethodDelete:
		routes.deleteAccessBinding(response, request, requestID, bindingID)
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
}

func (routes *PolicyRoutes) getAccessBinding(response http.ResponseWriter, request *http.Request, requestID, bindingID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Access policy binding detail does not accept query parameters.", requestID)
		return
	}
	binding, _, ok := routes.accessBindingForAuthorizedRequest(response, request, requestID, bindingID,
		routes.operation(managementapi.MethodGET, accessBindingsPath+"/{bindingId}"), true)
	if !ok {
		return
	}
	response.Header().Set(managementapi.HeaderETag, policyResourceETag(policyETagAccessBinding, binding.Revision))
	writeProviderJSON(response, http.StatusOK, managementapi.AccessPolicyBindingDetail{Data: newAccessPolicyBinding(binding)}, requestID)
}

func (routes *PolicyRoutes) patchAccessBinding(response http.ResponseWriter, request *http.Request, requestID, bindingID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Access policy binding update does not accept query parameters.", requestID)
		return
	}
	binding, session, ok := routes.accessBindingForAuthorizedRequest(response, request, requestID, bindingID,
		routes.operation(managementapi.MethodPATCH, accessBindingsPath+"/{bindingId}"), true)
	if !ok {
		return
	}
	revision, ok := requirePolicyRevision(response, request, requestID, policyETagAccessBinding)
	if !ok {
		return
	}
	var body managementapi.AccessPolicyBindingPatchRequest
	if !decodePolicyBody(response, request, requestID, &body) {
		return
	}
	result, err := routes.service.UpdateAccessBinding(request.Context(), policymanagement.UpdateBindingRequest{
		NamespaceID: binding.NamespaceID, BindingID: binding.ID, ExpectedRevision: revision,
		Status: accesscontrol.BindingStatus(body.Status), Actor: routes.actor(request, session, requestID),
	})
	if err != nil {
		writePolicyError(response, err, requestID)
		return
	}
	writePolicyMutation(response, result, requestID, policyETagAccessBinding, false)
}

func (routes *PolicyRoutes) deleteAccessBinding(response http.ResponseWriter, request *http.Request, requestID, bindingID string) {
	if request.URL.RawQuery != "" || !noRequestBody(response, request, requestID) {
		return
	}
	binding, session, ok := routes.accessBindingForAuthorizedRequest(response, request, requestID, bindingID,
		routes.operation(managementapi.MethodDELETE, accessBindingsPath+"/{bindingId}"), true)
	if !ok {
		return
	}
	revision, ok := requirePolicyRevision(response, request, requestID, policyETagAccessBinding)
	if !ok {
		return
	}
	result, err := routes.service.DeleteAccessBinding(request.Context(), policymanagement.DeleteBindingRequest{
		NamespaceID: binding.NamespaceID, BindingID: binding.ID, ExpectedRevision: revision,
		Actor: routes.actor(request, session, requestID),
	})
	if err != nil {
		writePolicyError(response, err, requestID)
		return
	}
	writePolicyMutation(response, result, requestID, policyETagAccessBinding, false)
}

func (routes *PolicyRoutes) accessBindingForAuthorizedRequest(response http.ResponseWriter, request *http.Request,
	requestID, bindingID string, operation managementapi.OperationContract, nondisclosing bool,
) (policymanagement.AccessPolicyBinding, managementauth.AuthenticatedSession, bool) {
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return policymanagement.AccessPolicyBinding{}, managementauth.AuthenticatedSession{}, false
	}
	binding, err := routes.service.GetAccessBinding(request.Context(), namespaceID, bindingID)
	if err != nil {
		writePolicyError(response, err, requestID)
		return policymanagement.AccessPolicyBinding{}, managementauth.AuthenticatedSession{}, false
	}
	targets, conditions, valid := policyBindingTargets(namespaceID, binding.PolicyID, binding.Subject, false)
	if !valid || !routes.authorize(response, request, requestID, session, namespaceID, operation, targets, conditions, nondisclosing) {
		return policymanagement.AccessPolicyBinding{}, managementauth.AuthenticatedSession{}, false
	}
	return binding, session, true
}

func (routes *PolicyRoutes) listRateBindings(response http.ResponseWriter, request *http.Request, requestID string) {
	input, ok := policyBindingListQuery(response, request, requestID, true)
	if !ok {
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	input.NamespaceID = namespaceID
	operation := routes.operation(managementapi.MethodGET, rateBindingsPath)
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
	input.Scope = scope
	page, err := routes.service.ListRateBindings(request.Context(), input)
	if err != nil {
		writePolicyError(response, err, requestID)
		return
	}
	writeProviderJSON(response, http.StatusOK, newRateLimitBindingPage(page), requestID)
}

func (routes *PolicyRoutes) createRateBinding(response http.ResponseWriter, request *http.Request, requestID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Rate limit binding create does not accept query parameters.", requestID)
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
	var body managementapi.RateLimitBindingCreateRequest
	if !decodePolicyBody(response, request, requestID, &body) {
		return
	}
	item := policybulk.RateBindingItem{
		ItemID: singleBindingValidationItemID, PolicyID: body.PolicyID,
		Subject: policymanagement.Subject{Type: accesscontrol.SubjectKind(body.Subject.Type), ID: body.Subject.ID},
		Mode:    accesscontrol.RateBindingMode(body.Mode),
	}
	if body.InlinePolicy != nil {
		rules, err := policyRules(body.InlinePolicy.Rules)
		if err != nil {
			writeProviderError(response, http.StatusBadRequest, "invalid_request", "Rate limit binding request is invalid.", requestID)
			return
		}
		item.InlinePolicy = &policybulk.InlineRateLimitPolicy{
			Name:        body.InlinePolicy.Name,
			Description: body.InlinePolicy.Description, Rules: rules,
		}
	}
	if policybulk.ValidateRateItems([]policybulk.RateBindingItem{item}) != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Rate limit binding request is invalid.", requestID)
		return
	}
	targets, conditions, valid := policyBindingTargets(namespaceID, item.PolicyID, item.Subject, true)
	if !valid || !routes.authorize(response, request, requestID, session, namespaceID,
		routes.operation(managementapi.MethodPOST, rateBindingsPath), targets, conditions, false) {
		return
	}
	actor := routes.actor(request, session, requestID)
	var result policymanagement.MutationResult
	createdPolicy := false
	policyID := item.PolicyID
	var err error
	if item.InlinePolicy == nil {
		result, err = routes.service.CreateRateBinding(request.Context(), policymanagement.CreateRateBindingRequest{
			NamespaceID: namespaceID, PolicyID: item.PolicyID, Subject: item.Subject, Mode: item.Mode,
			IdempotencyKey: string(idempotencyKey), Actor: actor,
		})
	} else {
		inline, inlineErr := routes.service.CreateInlineRateBinding(request.Context(), policymanagement.CreateInlineRateBindingRequest{
			NamespaceID: namespaceID, Name: item.InlinePolicy.Name, Description: item.InlinePolicy.Description,
			Rules: item.InlinePolicy.Rules, Subject: item.Subject, Mode: item.Mode,
			IdempotencyKey: string(idempotencyKey), Actor: actor,
		})
		if inlineErr == nil {
			result, policyID, createdPolicy = inline.MutationResult, inline.Policy.ID, inline.Created
		}
		err = inlineErr
	}
	if err != nil {
		writePolicyError(response, err, requestID)
		return
	}
	response.Header().Set("Location", rateBindingsPath+"/"+result.ID)
	response.Header().Set(managementapi.HeaderETag, policyResourceETag(policyETagRateBinding, result.Revision))
	setIdempotencyReplayHeader(response, result.Replayed)
	replayed := result.Replayed
	writeProviderJSON(response, result.HTTPStatus, managementapi.RateLimitBindingCreateReceipt{
		BindingID: result.ID, PolicyID: policyID, Revision: result.Revision,
		CreatedPolicy: createdPolicy, Idempotency: &managementapi.IdempotencyMetadata{Replayed: replayed},
	}, requestID)
}

func (routes *PolicyRoutes) rateBindingResource(response http.ResponseWriter, request *http.Request, requestID, bindingID string) {
	switch request.Method {
	case http.MethodGet:
		routes.getRateBinding(response, request, requestID, bindingID)
	case http.MethodPatch:
		routes.patchRateBinding(response, request, requestID, bindingID)
	case http.MethodDelete:
		routes.deleteRateBinding(response, request, requestID, bindingID)
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
}

func (routes *PolicyRoutes) getRateBinding(response http.ResponseWriter, request *http.Request, requestID, bindingID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Rate limit binding detail does not accept query parameters.", requestID)
		return
	}
	binding, _, ok := routes.rateBindingForAuthorizedRequest(response, request, requestID, bindingID,
		routes.operation(managementapi.MethodGET, rateBindingsPath+"/{bindingId}"), true)
	if !ok {
		return
	}
	response.Header().Set(managementapi.HeaderETag, policyResourceETag(policyETagRateBinding, binding.Revision))
	writeProviderJSON(response, http.StatusOK, managementapi.RateLimitBindingDetail{Data: newRateLimitBinding(binding)}, requestID)
}

func (routes *PolicyRoutes) patchRateBinding(response http.ResponseWriter, request *http.Request, requestID, bindingID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Rate limit binding update does not accept query parameters.", requestID)
		return
	}
	binding, session, ok := routes.rateBindingForAuthorizedRequest(response, request, requestID, bindingID,
		routes.operation(managementapi.MethodPATCH, rateBindingsPath+"/{bindingId}"), true)
	if !ok {
		return
	}
	revision, ok := requirePolicyRevision(response, request, requestID, policyETagRateBinding)
	if !ok {
		return
	}
	var body managementapi.RateLimitBindingPatchRequest
	if !decodePolicyBody(response, request, requestID, &body) {
		return
	}
	result, err := routes.service.UpdateRateBinding(request.Context(), policymanagement.UpdateBindingRequest{
		NamespaceID: binding.NamespaceID, BindingID: binding.ID, ExpectedRevision: revision,
		Status: accesscontrol.BindingStatus(body.Status), Actor: routes.actor(request, session, requestID),
	})
	if err != nil {
		writePolicyError(response, err, requestID)
		return
	}
	writePolicyMutation(response, result, requestID, policyETagRateBinding, false)
}

func (routes *PolicyRoutes) deleteRateBinding(response http.ResponseWriter, request *http.Request, requestID, bindingID string) {
	if request.URL.RawQuery != "" || !noRequestBody(response, request, requestID) {
		return
	}
	binding, session, ok := routes.rateBindingForAuthorizedRequest(response, request, requestID, bindingID,
		routes.operation(managementapi.MethodDELETE, rateBindingsPath+"/{bindingId}"), true)
	if !ok {
		return
	}
	revision, ok := requirePolicyRevision(response, request, requestID, policyETagRateBinding)
	if !ok {
		return
	}
	result, err := routes.service.DeleteRateBinding(request.Context(), policymanagement.DeleteBindingRequest{
		NamespaceID: binding.NamespaceID, BindingID: binding.ID, ExpectedRevision: revision,
		Actor: routes.actor(request, session, requestID),
	})
	if err != nil {
		writePolicyError(response, err, requestID)
		return
	}
	writePolicyMutation(response, result, requestID, policyETagRateBinding, false)
}

func (routes *PolicyRoutes) rateBindingForAuthorizedRequest(response http.ResponseWriter, request *http.Request,
	requestID, bindingID string, operation managementapi.OperationContract, nondisclosing bool,
) (policymanagement.RateLimitBinding, managementauth.AuthenticatedSession, bool) {
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return policymanagement.RateLimitBinding{}, managementauth.AuthenticatedSession{}, false
	}
	binding, err := routes.service.GetRateBinding(request.Context(), namespaceID, bindingID)
	if err != nil {
		writePolicyError(response, err, requestID)
		return policymanagement.RateLimitBinding{}, managementauth.AuthenticatedSession{}, false
	}
	targets, conditions, valid := policyBindingTargets(namespaceID, binding.PolicyID, binding.Subject, true)
	if !valid || !routes.authorize(response, request, requestID, session, namespaceID, operation, targets, conditions, nondisclosing) {
		return policymanagement.RateLimitBinding{}, managementauth.AuthenticatedSession{}, false
	}
	return binding, session, true
}
