package managementserver

import (
	"errors"
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policybulk"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policymanagement"
)

func (routes *PolicyRoutes) bulkAccessBindings(response http.ResponseWriter, request *http.Request, requestID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Bulk apply does not accept query parameters.", requestID)
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
	var body managementapi.AccessPolicyBindingBulkApplyRequest
	if !decodePolicyBody(response, request, requestID, &body) {
		return
	}
	items := make([]policybulk.AccessBindingItem, len(body.Items))
	for index, item := range body.Items {
		items[index] = policybulk.AccessBindingItem{
			ItemID: item.ItemID, PolicyID: item.PolicyID,
			Subject: policymanagement.Subject{Type: accesscontrol.SubjectKind(item.Subject.Type), ID: item.Subject.ID},
		}
	}
	if policybulk.ValidateAccessItems(items) != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Bulk binding request is invalid.", requestID)
		return
	}
	operation := routes.operation(managementapi.MethodPOST, accessBindingBulkPath)
	for _, item := range items {
		targets, conditions, valid := policyBindingTargets(namespaceID, item.PolicyID, item.Subject, false)
		if !valid || !routes.authorize(response, request, requestID, session, namespaceID,
			operation, targets, conditions, false) {
			return
		}
	}
	result, err := routes.bulk.EnqueueAccessBindings(request.Context(), policybulk.EnqueueAccessRequest{
		NamespaceID: namespaceID, Items: items, IdempotencyKey: string(idempotencyKey),
		Actor: routes.actor(request, session, requestID),
	})
	if err != nil {
		writePolicyBulkError(response, err, requestID)
		return
	}
	response.Header().Set("Location", managementapi.BasePath+"/operations/"+result.Operation.ID)
	setIdempotencyReplayHeader(response, result.Replayed)
	writeProviderJSON(response, http.StatusAccepted, newPolicyBulkOperation(result.Operation), requestID)
}

func (routes *PolicyRoutes) bulkRateBindings(response http.ResponseWriter, request *http.Request, requestID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Bulk apply does not accept query parameters.", requestID)
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
	var body managementapi.RateLimitBindingBulkApplyRequest
	if !decodePolicyBody(response, request, requestID, &body) {
		return
	}
	items := make([]policybulk.RateBindingItem, len(body.Items))
	for index, item := range body.Items {
		items[index] = policybulk.RateBindingItem{
			ItemID: item.ItemID, PolicyID: item.PolicyID,
			Subject: policymanagement.Subject{Type: accesscontrol.SubjectKind(item.Subject.Type), ID: item.Subject.ID},
			Mode:    accesscontrol.RateBindingMode(item.Mode),
		}
		if item.InlinePolicy != nil {
			rules, err := policyRules(item.InlinePolicy.Rules)
			if err != nil {
				writeProviderError(response, http.StatusBadRequest, "invalid_request", "Bulk binding request is invalid.", requestID)
				return
			}
			items[index].InlinePolicy = &policybulk.InlineRateLimitPolicy{
				Name:        item.InlinePolicy.Name,
				Description: item.InlinePolicy.Description, Rules: rules,
			}
		}
	}
	if policybulk.ValidateRateItems(items) != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Bulk binding request is invalid.", requestID)
		return
	}
	operation := routes.operation(managementapi.MethodPOST, rateBindingBulkPath)
	for _, item := range items {
		policyID := item.PolicyID
		if item.InlinePolicy != nil {
			policyID = ""
		}
		targets, conditions, valid := policyBindingTargets(namespaceID, policyID, item.Subject, true)
		if !valid || !routes.authorize(response, request, requestID, session, namespaceID,
			operation, targets, conditions, false) {
			return
		}
	}
	result, err := routes.bulk.EnqueueRateBindings(request.Context(), policybulk.EnqueueRateRequest{
		NamespaceID: namespaceID, Items: items, IdempotencyKey: string(idempotencyKey),
		Actor: routes.actor(request, session, requestID),
	})
	if err != nil {
		writePolicyBulkError(response, err, requestID)
		return
	}
	response.Header().Set("Location", managementapi.BasePath+"/operations/"+result.Operation.ID)
	setIdempotencyReplayHeader(response, result.Replayed)
	writeProviderJSON(response, http.StatusAccepted, newPolicyBulkOperation(result.Operation), requestID)
}

func writePolicyBulkError(response http.ResponseWriter, err error, requestID string) {
	switch {
	case errors.Is(err, policybulk.ErrInvalidRequest):
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Bulk binding request is invalid.", requestID)
	case errors.Is(err, policybulk.ErrNotFound):
		writeProviderError(response, http.StatusNotFound, "not_found", "Operation not found.", requestID)
	case errors.Is(err, policybulk.ErrConflict):
		writeProviderError(response, http.StatusConflict, "operation_conflict", "Operation state changed. Refresh and retry.", requestID)
	case errors.Is(err, accesscontrol.ErrInvalid):
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Bulk binding request is invalid.", requestID)
	default:
		writePolicyError(response, err, requestID)
	}
}
