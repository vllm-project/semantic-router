package managementserver

import (
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
)

func (routes *RoutingRoutes) listRecipes(response http.ResponseWriter, request *http.Request, requestID string) {
	pageRequest, pageSize, err := parseRoutingListQuery(request.URL.RawQuery)
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Routing query is invalid.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	scope, err := resolveListResultScope(
		request.Context(), routes.scopes, session, namespaceID, accesscontrol.PermissionRoutingRead,
	)
	if err != nil {
		writeResultScopeError(response, err, requestID)
		return
	}
	pageRequest.Scope = scope
	page, err := routes.service.ListRecipes(request.Context(), namespaceID, pageRequest)
	if err != nil {
		writeRoutingDomainError(response, err, requestID, false)
		return
	}
	writeProviderJSON(response, http.StatusOK, routingRecipePageDTO(page, pageSize), requestID)
}

func (routes *RoutingRoutes) createRecipe(response http.ResponseWriter, request *http.Request, requestID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Recipe create does not accept query parameters.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	var body managementapi.RoutingRecipeWrite
	if !decodeRoutingBody(response, request, requestID, &body) {
		return
	}
	input := routingRecipeInput(body)
	if input.ID != "" && routingmanagement.ValidateResourceID(input.ID) != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Recipe request is invalid.", requestID)
		return
	}
	_, err := routes.authorize(request.Context(), session, namespaceID,
		routes.operation(managementapi.MethodPOST, routingRecipesPath), routingRecipeTargets(namespaceID, input.ID), nil)
	if err != nil {
		writeRoutingAuthorizationError(response, err, requestID, false)
		return
	}
	command, ok := routes.bindCommand(response, request, requestID, namespaceID, session, routingRecipesPath, body)
	if !ok {
		return
	}
	if routes.replayRecipeResource(response, request, requestID, command, "") {
		return
	}
	recipe, receipt, err := routes.service.CreateRecipe(request.Context(), namespaceID, input,
		routingMutationContext(session, requestID, "create Recipe", &command))
	if err != nil {
		if routes.replayRecipeResource(response, request, requestID, command, "") {
			return
		}
		writeRoutingDomainError(response, err, requestID, false)
		return
	}
	response.Header().Set("Location", routingRecipesPath+"/"+recipe.ID)
	response.Header().Set(managementapi.HeaderETag, routingETag("rcp", receipt.ResourceRevision))
	writeRoutingResourceReceipt(response, http.StatusCreated, "routing_recipe", recipe.ID, receipt, true, requestID)
}

func (routes *RoutingRoutes) recipeResource(response http.ResponseWriter, request *http.Request, requestID string) {
	recipeID, action, ok := routingResourcePathValue(routingRecipesPath, request.URL.Path)
	if !ok || action != "" {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	switch request.Method {
	case http.MethodGet:
		routes.getRecipe(response, request, requestID, recipeID)
	case http.MethodPatch:
		routes.updateRecipe(response, request, requestID, recipeID)
	case http.MethodDelete:
		routes.deleteRecipe(response, request, requestID, recipeID)
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
}

func (routes *RoutingRoutes) getRecipe(
	response http.ResponseWriter, request *http.Request, requestID, recipeID string,
) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Recipe detail does not accept query parameters.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	recipe, err := routes.service.GetRecipe(request.Context(), namespaceID, recipeID)
	if err != nil {
		writeRoutingDomainError(response, err, requestID, false)
		return
	}
	_, err = routes.authorize(request.Context(), session, namespaceID,
		routes.operation(managementapi.MethodGET, routingRecipesPath+"/{recipeId}"),
		routingRecipeTargets(namespaceID, recipeID), nil)
	if err != nil {
		writeRoutingAuthorizationError(response, err, requestID, true)
		return
	}
	response.Header().Set(managementapi.HeaderETag, routingETag("rcp", recipe.Revision))
	writeProviderJSON(response, http.StatusOK, managementapi.RoutingRecipeDetail{
		Data: routingRecipeViewDTO(recipe),
	}, requestID)
}

func (routes *RoutingRoutes) updateRecipe(
	response http.ResponseWriter, request *http.Request, requestID, recipeID string,
) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Recipe update does not accept query parameters.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	revision, ok := requireRoutingRevision(response, request, requestID, "rcp")
	if !ok {
		return
	}
	var body managementapi.RoutingRecipeWrite
	if !decodeRoutingBody(response, request, requestID, &body) {
		return
	}
	_, err := routes.authorize(request.Context(), session, namespaceID,
		routes.operation(managementapi.MethodPATCH, routingRecipesPath+"/{recipeId}"),
		routingRecipeTargets(namespaceID, recipeID), nil)
	if err != nil {
		writeRoutingAuthorizationError(response, err, requestID, false)
		return
	}
	_, receipt, err := routes.service.UpdateRecipe(request.Context(), namespaceID, recipeID, revision, routingRecipeInput(body),
		routingMutationContext(session, requestID, "update Recipe", nil))
	if err != nil {
		writeRoutingDomainError(response, err, requestID, true)
		return
	}
	response.Header().Set(managementapi.HeaderETag, routingETag("rcp", receipt.ResourceRevision))
	writeRoutingResourceReceipt(response, http.StatusOK, "routing_recipe", recipeID, receipt, false, requestID)
}

func (routes *RoutingRoutes) deleteRecipe(
	response http.ResponseWriter, request *http.Request, requestID, recipeID string,
) {
	if request.URL.RawQuery != "" || !rejectRoutingBody(response, request, requestID) {
		if request.URL.RawQuery != "" {
			writeProviderError(response, http.StatusBadRequest, "invalid_request", "Recipe delete does not accept query parameters.", requestID)
		}
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	_, err := routes.authorize(request.Context(), session, namespaceID,
		routes.operation(managementapi.MethodDELETE, routingRecipesPath+"/{recipeId}"),
		routingRecipeTargets(namespaceID, recipeID), nil)
	if err != nil {
		writeRoutingAuthorizationError(response, err, requestID, false)
		return
	}
	revision, ok := requireRoutingRevision(response, request, requestID, "rcp")
	if !ok {
		return
	}
	receipt, err := routes.service.DeleteRecipe(request.Context(), namespaceID, recipeID, revision,
		routingMutationContext(session, requestID, "delete Recipe", nil))
	if err != nil {
		writeRoutingDomainError(response, err, requestID, true)
		return
	}
	setProviderResponseHeaders(response, requestID)
	response.Header().Set(managementapi.HeaderETag, routingETag("rcp", receipt.ResourceRevision))
	response.WriteHeader(http.StatusNoContent)
}

func routingRecipeTargets(namespaceID, recipeID string) map[string][]accesscontrol.ScopedTarget {
	target := routingNamespaceTarget(namespaceID)
	if recipeID != "" {
		target = routingTarget(namespaceID, accesscontrol.ScopeResourceRecipe, recipeID)
	}
	return map[string][]accesscontrol.ScopedTarget{"target": {target}}
}

func (routes *RoutingRoutes) replayRecipeResource(
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
	receipt, err := routingResourceReplay(stored, "routing_recipe", resourceID)
	if err != nil {
		writeRoutingDomainError(response, err, requestID, false)
		return true
	}
	id := stored.Resource.ResourceID
	response.Header().Set("Location", routingRecipesPath+"/"+id)
	response.Header().Set(managementapi.HeaderETag, routingETag("rcp", receipt.ResourceRevision))
	writeRoutingResourceReceipt(response, stored.Resource.ResponseStatus, "routing_recipe", id, receipt, true, requestID)
	return true
}
