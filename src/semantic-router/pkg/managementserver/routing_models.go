package managementserver

import (
	"net/http"
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
)

func (routes *RoutingRoutes) listModels(response http.ResponseWriter, request *http.Request, requestID string) {
	page, pageSize, ok := routes.loadModelPage(response, request, requestID)
	if !ok {
		return
	}
	writeProviderJSON(response, http.StatusOK, routingModelPageDTO(page, pageSize), requestID)
}

func (routes *RoutingRoutes) listModelCards(response http.ResponseWriter, request *http.Request, requestID string) {
	page, pageSize, ok := routes.loadModelPage(response, request, requestID)
	if !ok {
		return
	}
	writeProviderJSON(response, http.StatusOK, routingModelCardPageDTO(page, pageSize), requestID)
}

func (routes *RoutingRoutes) loadModelPage(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
) (routingmanagement.Page[routingmanagement.Model], int, bool) {
	pageRequest, pageSize, err := parseRoutingListQuery(request.URL.RawQuery)
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Routing query is invalid.", requestID)
		return routingmanagement.Page[routingmanagement.Model]{}, 0, false
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return routingmanagement.Page[routingmanagement.Model]{}, 0, false
	}
	scope, err := resolveListResultScope(
		request.Context(), routes.scopes, session, namespaceID, accesscontrol.PermissionRoutingRead,
	)
	if err != nil {
		writeResultScopeError(response, err, requestID)
		return routingmanagement.Page[routingmanagement.Model]{}, 0, false
	}
	pageRequest.Scope = scope
	page, err := routes.service.ListModels(request.Context(), namespaceID, pageRequest)
	if err != nil {
		writeRoutingDomainError(response, err, requestID, false)
		return routingmanagement.Page[routingmanagement.Model]{}, 0, false
	}
	return page, pageSize, true
}

func (routes *RoutingRoutes) createModel(response http.ResponseWriter, request *http.Request, requestID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Model create does not accept query parameters.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	var body managementapi.RoutingModelWrite
	if !decodeRoutingBody(response, request, requestID, &body) {
		return
	}
	input, err := routingModelInput(body)
	if err != nil || !validRoutingCredentialIDs(input) {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Model request is invalid.", requestID)
		return
	}
	credentials := routingModelCredentialIDs(input)
	targetID := input.ID
	if targetID != "" && routingmanagement.ValidateResourceID(targetID) != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Model request is invalid.", requestID)
		return
	}
	_, err = routes.authorize(request.Context(), session, namespaceID,
		routes.operation(managementapi.MethodPOST, routingModelsPath),
		routingModelTargets(namespaceID, targetID, credentials),
		map[string]bool{"provider_credential_referenced": len(credentials) != 0})
	if err != nil {
		writeRoutingAuthorizationError(response, err, requestID, false)
		return
	}
	command, ok := routes.bindCommand(response, request, requestID, namespaceID, session, routingModelsPath, body)
	if !ok {
		return
	}
	if routes.replayModelResource(response, request, requestID, command, "") {
		return
	}
	model, receipt, err := routes.service.CreateModel(request.Context(), namespaceID, input,
		routingMutationContext(session, requestID, "create Model", &command))
	if err != nil {
		if routes.replayModelResource(response, request, requestID, command, "") {
			return
		}
		writeRoutingDomainError(response, err, requestID, false)
		return
	}
	response.Header().Set("Location", routingModelsPath+"/"+model.ID)
	response.Header().Set(managementapi.HeaderETag, routingETag("mdl", receipt.ResourceRevision))
	writeRoutingResourceReceipt(response, http.StatusCreated, "routing_model", model.ID, receipt, true, requestID)
}

func (routes *RoutingRoutes) bulkImportModels(response http.ResponseWriter, request *http.Request, requestID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Model import does not accept query parameters.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	var body managementapi.RoutingBulkImportRequest
	if !decodeRoutingBody(response, request, requestID, &body) {
		return
	}
	if body.CredentialID != "" && !canonicalUUID(body.CredentialID) {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Model import request is invalid.", requestID)
		return
	}
	targets, err := routingBulkModelTargets(namespaceID, body)
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Model import request is invalid.", requestID)
		return
	}
	decision, err := routes.authorize(request.Context(), session, namespaceID,
		routes.operation(managementapi.MethodPOST, routingModelsPath+":bulk-import"), targets,
		map[string]bool{"provider_credential_referenced": body.CredentialID != ""})
	if err != nil {
		writeRoutingAuthorizationError(response, err, requestID, false)
		return
	}
	if !validAuthorityDigest(decision.AuthorityDigest) {
		writeProviderError(response, http.StatusServiceUnavailable, "authorization_unavailable", "Authorization state is unavailable.", requestID)
		return
	}
	domain, err := routingBulkImportInput(body, namespaceID, decision.AuthorityDigest)
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Model import request is invalid.", requestID)
		return
	}
	command, ok := routes.bindCommand(
		response, request, requestID, namespaceID, session, routingModelsPath+":bulk-import", body,
	)
	if !ok {
		return
	}
	if routes.replayModelOperation(response, request, requestID, command) {
		return
	}
	_, receipt, err := routes.service.BulkImport(request.Context(), domain,
		routingMutationContext(session, requestID, "bulk import Models", &command))
	if err != nil {
		if routes.replayModelOperation(response, request, requestID, command) {
			return
		}
		writeRoutingDomainError(response, err, requestID, false)
		return
	}
	response.Header().Set("Location", managementapi.BasePath+"/operations/"+receipt.OperationID)
	writeRoutingOperationReceipt(response, receipt, false, requestID)
}

func (routes *RoutingRoutes) modelResource(response http.ResponseWriter, request *http.Request, requestID string) {
	modelID, action, ok := routingResourcePathValue(routingModelsPath, request.URL.Path)
	if !ok {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	switch {
	case request.Method == http.MethodGet && action == "":
		routes.getModel(response, request, requestID, modelID)
	case request.Method == http.MethodPatch && action == "":
		routes.updateModel(response, request, requestID, modelID)
	case request.Method == http.MethodDelete && action == "":
		routes.deleteModel(response, request, requestID, modelID)
	case request.Method == http.MethodPost && action == "probe":
		routes.probeModel(response, request, requestID, modelID)
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
}

func (routes *RoutingRoutes) getModel(
	response http.ResponseWriter, request *http.Request, requestID, modelID string,
) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Model detail does not accept query parameters.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	model, err := routes.service.GetModel(request.Context(), namespaceID, modelID)
	if err != nil {
		writeRoutingDomainError(response, err, requestID, false)
		return
	}
	_, err = routes.authorize(request.Context(), session, namespaceID,
		routes.operation(managementapi.MethodGET, routingModelsPath+"/{modelId}"),
		routingModelTargets(namespaceID, modelID, nil), nil)
	if err != nil {
		writeRoutingAuthorizationError(response, err, requestID, true)
		return
	}
	response.Header().Set(managementapi.HeaderETag, routingETag("mdl", model.Revision))
	writeProviderJSON(response, http.StatusOK, managementapi.RoutingModelDetail{
		Data: routingModelViewDTO(model),
	}, requestID)
}

func (routes *RoutingRoutes) updateModel(
	response http.ResponseWriter, request *http.Request, requestID, modelID string,
) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Model update does not accept query parameters.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	revision, ok := requireRoutingRevision(response, request, requestID, "mdl")
	if !ok {
		return
	}
	var body managementapi.RoutingModelPatch
	if !decodeRoutingBody(response, request, requestID, &body) {
		return
	}
	patch, err := routingModelPatch(body)
	if err != nil || patch.Empty() || !validRoutingModelPatchCredentialIDs(patch) {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Model request is invalid.", requestID)
		return
	}
	credentials := routingModelPatchCredentialIDs(patch)
	_, err = routes.authorize(request.Context(), session, namespaceID,
		routes.operation(managementapi.MethodPATCH, routingModelsPath+"/{modelId}"),
		routingModelTargets(namespaceID, modelID, credentials),
		map[string]bool{"provider_credential_referenced": len(credentials) != 0})
	if err != nil {
		writeRoutingAuthorizationError(response, err, requestID, false)
		return
	}
	_, receipt, err := routes.service.PatchModel(request.Context(), namespaceID, modelID, revision, patch,
		routingMutationContext(session, requestID, "update Model", nil))
	if err != nil {
		writeRoutingDomainError(response, err, requestID, true)
		return
	}
	response.Header().Set(managementapi.HeaderETag, routingETag("mdl", receipt.ResourceRevision))
	writeRoutingResourceReceipt(response, http.StatusOK, "routing_model", modelID, receipt, false, requestID)
}

func (routes *RoutingRoutes) deleteModel(
	response http.ResponseWriter, request *http.Request, requestID, modelID string,
) {
	if request.URL.RawQuery != "" || !rejectRoutingBody(response, request, requestID) {
		if request.URL.RawQuery != "" {
			writeProviderError(response, http.StatusBadRequest, "invalid_request", "Model delete does not accept query parameters.", requestID)
		}
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	_, err := routes.authorize(request.Context(), session, namespaceID,
		routes.operation(managementapi.MethodDELETE, routingModelsPath+"/{modelId}"),
		routingModelTargets(namespaceID, modelID, nil), nil)
	if err != nil {
		writeRoutingAuthorizationError(response, err, requestID, false)
		return
	}
	revision, ok := requireRoutingRevision(response, request, requestID, "mdl")
	if !ok {
		return
	}
	receipt, err := routes.service.DeleteModel(request.Context(), namespaceID, modelID, revision,
		routingMutationContext(session, requestID, "delete Model", nil))
	if err != nil {
		writeRoutingDomainError(response, err, requestID, true)
		return
	}
	setProviderResponseHeaders(response, requestID)
	response.Header().Set(managementapi.HeaderETag, routingETag("mdl", receipt.ResourceRevision))
	response.WriteHeader(http.StatusNoContent)
}

func (routes *RoutingRoutes) probeModel(
	response http.ResponseWriter, request *http.Request, requestID, modelID string,
) {
	if request.URL.RawQuery != "" || !rejectRoutingBody(response, request, requestID) {
		if request.URL.RawQuery != "" {
			writeProviderError(response, http.StatusBadRequest, "invalid_request", "Model probe does not accept query parameters.", requestID)
		}
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	model, err := routes.service.GetModel(request.Context(), namespaceID, modelID)
	if err != nil {
		writeRoutingDomainError(response, err, requestID, false)
		return
	}
	credentials := routingModelCredentialIDsFromRecord(model)
	_, err = routes.authorize(request.Context(), session, namespaceID,
		routes.operation(managementapi.MethodPOST, routingModelsPath+"/{modelId}:probe"),
		routingModelTargets(namespaceID, modelID, credentials),
		map[string]bool{"provider_credential_referenced": len(credentials) != 0})
	if err != nil {
		writeRoutingAuthorizationError(response, err, requestID, true)
		return
	}
	result, err := routes.service.ProbeModel(request.Context(), namespaceID, modelID, 0)
	if err != nil {
		writeRoutingDomainError(response, err, requestID, false)
		return
	}
	writeProviderJSON(response, http.StatusOK, routingProbeResponseDTO(result), requestID)
}

func routingModelTargets(namespaceID, modelID string, credentialIDs []string) map[string][]accesscontrol.ScopedTarget {
	target := routingNamespaceTarget(namespaceID)
	if modelID != "" {
		target = routingTarget(namespaceID, accesscontrol.ScopeResourceModel, modelID)
	}
	result := map[string][]accesscontrol.ScopedTarget{"target": {target}}
	for _, credentialID := range credentialIDs {
		result["credential"] = append(result["credential"], routingTarget(
			namespaceID, accesscontrol.ScopeResourceProviderCredential, credentialID,
		))
	}
	return result
}

func routingBulkModelTargets(
	namespaceID string, body managementapi.RoutingBulkImportRequest,
) (map[string][]accesscontrol.ScopedTarget, error) {
	targets := make([]accesscontrol.ScopedTarget, 0, len(body.Selections))
	seen := make(map[string]struct{}, len(body.Selections))
	needsNamespace := false
	for _, selection := range body.Selections {
		if selection.ID == "" {
			needsNamespace = true
			continue
		}
		if routingmanagement.ValidateResourceID(selection.ID) != nil {
			return nil, routingmanagement.ErrInvalid
		}
		if _, duplicate := seen[selection.ID]; duplicate {
			return nil, routingmanagement.ErrInvalid
		}
		seen[selection.ID] = struct{}{}
		targets = append(targets, routingTarget(namespaceID, accesscontrol.ScopeResourceModel, selection.ID))
	}
	if needsNamespace || len(targets) == 0 {
		targets = append(targets, routingNamespaceTarget(namespaceID))
	}
	result := map[string][]accesscontrol.ScopedTarget{"target": targets}
	if body.CredentialID != "" {
		result["credential"] = []accesscontrol.ScopedTarget{routingTarget(
			namespaceID, accesscontrol.ScopeResourceProviderCredential, body.CredentialID,
		)}
	}
	return result, nil
}

func validRoutingCredentialIDs(input routingmanagement.ModelInput) bool {
	return validRoutingBackendCredentialIDs(input.Backends)
}

func validRoutingModelPatchCredentialIDs(patch routingmanagement.ModelPatch) bool {
	return patch.Backends == nil || validRoutingBackendCredentialIDs(*patch.Backends)
}

func validRoutingBackendCredentialIDs(backends []routingmanagement.ModelBackendInput) bool {
	for _, backend := range backends {
		if backend.CredentialID != "" && !canonicalUUID(backend.CredentialID) {
			return false
		}
	}
	return true
}

func routingModelCredentialIDs(input routingmanagement.ModelInput) []string {
	return routingBackendCredentialIDs(input.Backends)
}

func routingModelPatchCredentialIDs(patch routingmanagement.ModelPatch) []string {
	if patch.Backends == nil {
		return nil
	}
	return routingBackendCredentialIDs(*patch.Backends)
}

func routingBackendCredentialIDs(backends []routingmanagement.ModelBackendInput) []string {
	values := make([]string, 0, len(backends))
	for _, backend := range backends {
		if backend.CredentialID != "" {
			values = append(values, backend.CredentialID)
		}
	}
	return uniqueStrings(values)
}

func routingModelCredentialIDsFromRecord(model routingmanagement.Model) []string {
	values := make([]string, 0, len(model.Current.Backends))
	for _, backend := range model.Current.Backends {
		if backend.ProviderCredentialID != "" {
			values = append(values, backend.ProviderCredentialID)
		}
	}
	return uniqueStrings(values)
}

func uniqueStrings(values []string) []string {
	slices.Sort(values)
	return slices.Compact(values)
}

func (routes *RoutingRoutes) replayModelResource(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
	command managementcommand.Command,
	resourceID string,
) bool {
	stored, found, err := routes.lookupCommand(request.Context(), command)
	if err != nil {
		writeRoutingDomainError(response, err, requestID, false)
		return true
	}
	if !found {
		return false
	}
	receipt, err := routingResourceReplay(stored, "routing_model", resourceID)
	if err != nil {
		writeRoutingDomainError(response, err, requestID, false)
		return true
	}
	id := stored.Resource.ResourceID
	response.Header().Set("Location", routingModelsPath+"/"+id)
	response.Header().Set(managementapi.HeaderETag, routingETag("mdl", receipt.ResourceRevision))
	writeRoutingResourceReceipt(response, stored.Resource.ResponseStatus, "routing_model", id, receipt, true, requestID)
	return true
}

func (routes *RoutingRoutes) replayModelOperation(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
	command managementcommand.Command,
) bool {
	stored, found, err := routes.lookupCommand(request.Context(), command)
	if err != nil {
		writeRoutingDomainError(response, err, requestID, false)
		return true
	}
	if !found {
		return false
	}
	receipt, err := routingOperationReplay(stored, false)
	if err != nil {
		writeRoutingDomainError(response, err, requestID, false)
		return true
	}
	response.Header().Set("Location", managementapi.BasePath+"/operations/"+receipt.OperationID)
	writeRoutingOperationReceipt(response, receipt, false, requestID)
	return true
}
