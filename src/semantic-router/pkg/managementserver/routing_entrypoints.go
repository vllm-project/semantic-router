package managementserver

import (
	"context"
	"fmt"
	"net/http"
	"sort"
	"strconv"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

func (routes *RoutingRoutes) listEntrypoints(response http.ResponseWriter, request *http.Request, requestID string) {
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
	page, err := routes.service.ListEntrypoints(request.Context(), namespaceID, pageRequest)
	if err != nil {
		writeRoutingDomainError(response, err, requestID, false)
		return
	}
	writeProviderJSON(response, http.StatusOK, routingEntrypointPageDTO(page, pageSize), requestID)
}

func (routes *RoutingRoutes) createEntrypoint(response http.ResponseWriter, request *http.Request, requestID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Entrypoint create does not accept query parameters.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	var body managementapi.RoutingEntrypointWrite
	if !decodeRoutingBody(response, request, requestID, &body) {
		return
	}
	input := routingEntrypointInput(body)
	if input.ID != "" && routingmanagement.ValidateResourceID(input.ID) != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Entrypoint request is invalid.", requestID)
		return
	}
	dependencies, err := routes.loadEntrypointInputDependencies(request.Context(), namespaceID, input)
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Entrypoint dependencies are invalid.", requestID)
		return
	}
	_, err = routes.authorize(request.Context(), session, namespaceID,
		routes.operation(managementapi.MethodPOST, routingEntrypointsPath),
		routingEntrypointTargets(namespaceID, input.ID, dependencies), nil)
	if err != nil {
		writeRoutingAuthorizationError(response, err, requestID, false)
		return
	}
	command, ok := routes.bindCommand(response, request, requestID, namespaceID, session, routingEntrypointsPath, body)
	if !ok {
		return
	}
	if routes.replayEntrypointResource(response, request, requestID, command, "") {
		return
	}
	entrypoint, receipt, err := routes.service.CreateEntrypoint(request.Context(), namespaceID, input,
		routingMutationContext(session, requestID, "create Entrypoint", &command))
	if err != nil {
		if routes.replayEntrypointResource(response, request, requestID, command, "") {
			return
		}
		writeRoutingDomainError(response, err, requestID, false)
		return
	}
	response.Header().Set("Location", routingEntrypointsPath+"/"+entrypoint.ID)
	response.Header().Set(managementapi.HeaderETag, routingETag("ep", receipt.ResourceRevision))
	writeRoutingResourceReceipt(response, http.StatusCreated, "routing_entrypoint", entrypoint.ID, receipt, true, requestID)
}

func (routes *RoutingRoutes) entrypointResource(response http.ResponseWriter, request *http.Request, requestID string) {
	entrypointID, action, ok := routingResourcePathValue(routingEntrypointsPath, request.URL.Path)
	if !ok {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	switch {
	case request.Method == http.MethodGet && action == "":
		routes.getEntrypoint(response, request, requestID, entrypointID)
	case request.Method == http.MethodPatch && action == "":
		routes.updateEntrypoint(response, request, requestID, entrypointID)
	case request.Method == http.MethodDelete && action == "":
		routes.deleteEntrypoint(response, request, requestID, entrypointID)
	case request.Method == http.MethodPost && action == "publish":
		routes.publishEntrypoint(response, request, requestID, entrypointID, true)
	case request.Method == http.MethodPost && action == "unpublish":
		routes.publishEntrypoint(response, request, requestID, entrypointID, false)
	case request.Method == http.MethodPost && action == "resolve":
		routes.resolveEntrypoint(response, request, requestID, entrypointID)
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
}

func (routes *RoutingRoutes) getEntrypoint(
	response http.ResponseWriter, request *http.Request, requestID, entrypointID string,
) {
	query, err := strictRoutingQuery(request.URL.RawQuery, map[string]bool{"includeTopology": true})
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Entrypoint query is invalid.", requestID)
		return
	}
	includeTopology := false
	if value := query.Get("includeTopology"); value != "" {
		includeTopology, err = strconv.ParseBool(value)
		if err != nil {
			writeProviderError(response, http.StatusBadRequest, "invalid_request", "includeTopology must be true or false.", requestID)
			return
		}
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	entrypoint, err := routes.service.GetEntrypoint(request.Context(), namespaceID, entrypointID)
	if err != nil {
		writeRoutingDomainError(response, err, requestID, false)
		return
	}
	var dependencies []accesscontrol.ScopedTarget
	if includeTopology {
		dependencies, err = routingEntrypointRecordDependencies(namespaceID, entrypoint)
		if err != nil {
			writeProviderError(response, http.StatusServiceUnavailable, "routing_unavailable", "Routing Management is unavailable.", requestID)
			return
		}
	}
	_, err = routes.authorize(request.Context(), session, namespaceID,
		routes.operation(managementapi.MethodGET, routingEntrypointsPath+"/{entrypointId}"),
		routingEntrypointTargets(namespaceID, entrypointID, dependencies),
		map[string]bool{"entrypoint_topology_requested": includeTopology})
	if err != nil {
		writeRoutingAuthorizationError(response, err, requestID, true)
		return
	}
	response.Header().Set(managementapi.HeaderETag, routingETag("ep", entrypoint.Revision))
	writeProviderJSON(response, http.StatusOK, managementapi.RoutingEntrypointDetail{
		Data: routingEntrypointViewDTO(entrypoint, includeTopology),
	}, requestID)
}

func (routes *RoutingRoutes) updateEntrypoint(
	response http.ResponseWriter, request *http.Request, requestID, entrypointID string,
) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Entrypoint update does not accept query parameters.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	revision, ok := requireRoutingRevision(response, request, requestID, "ep")
	if !ok {
		return
	}
	var body managementapi.RoutingEntrypointWrite
	if !decodeRoutingBody(response, request, requestID, &body) {
		return
	}
	input := routingEntrypointInput(body)
	dependencies, err := routes.loadEntrypointInputDependencies(request.Context(), namespaceID, input)
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Entrypoint dependencies are invalid.", requestID)
		return
	}
	_, err = routes.authorize(request.Context(), session, namespaceID,
		routes.operation(managementapi.MethodPATCH, routingEntrypointsPath+"/{entrypointId}"),
		routingEntrypointTargets(namespaceID, entrypointID, dependencies), nil)
	if err != nil {
		writeRoutingAuthorizationError(response, err, requestID, false)
		return
	}
	_, receipt, err := routes.service.UpdateEntrypoint(request.Context(), namespaceID, entrypointID, revision, input,
		routingMutationContext(session, requestID, "update Entrypoint", nil))
	if err != nil {
		writeRoutingDomainError(response, err, requestID, true)
		return
	}
	response.Header().Set(managementapi.HeaderETag, routingETag("ep", receipt.ResourceRevision))
	writeRoutingResourceReceipt(response, http.StatusOK, "routing_entrypoint", entrypointID, receipt, false, requestID)
}

func (routes *RoutingRoutes) deleteEntrypoint(
	response http.ResponseWriter, request *http.Request, requestID, entrypointID string,
) {
	if request.URL.RawQuery != "" || !rejectRoutingBody(response, request, requestID) {
		if request.URL.RawQuery != "" {
			writeProviderError(response, http.StatusBadRequest, "invalid_request", "Entrypoint delete does not accept query parameters.", requestID)
		}
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	_, err := routes.authorize(request.Context(), session, namespaceID,
		routes.operation(managementapi.MethodDELETE, routingEntrypointsPath+"/{entrypointId}"),
		routingEntrypointTargets(namespaceID, entrypointID, nil), nil)
	if err != nil {
		writeRoutingAuthorizationError(response, err, requestID, false)
		return
	}
	revision, ok := requireRoutingRevision(response, request, requestID, "ep")
	if !ok {
		return
	}
	receipt, err := routes.service.DeleteEntrypoint(request.Context(), namespaceID, entrypointID, revision,
		routingMutationContext(session, requestID, "delete Entrypoint", nil))
	if err != nil {
		writeRoutingDomainError(response, err, requestID, true)
		return
	}
	setProviderResponseHeaders(response, requestID)
	response.Header().Set(managementapi.HeaderETag, routingETag("ep", receipt.ResourceRevision))
	response.WriteHeader(http.StatusNoContent)
}

func (routes *RoutingRoutes) publishEntrypoint(
	response http.ResponseWriter,
	request *http.Request,
	requestID, entrypointID string,
	publish bool,
) {
	if request.URL.RawQuery != "" || !rejectRoutingBody(response, request, requestID) {
		if request.URL.RawQuery != "" {
			writeProviderError(response, http.StatusBadRequest, "invalid_request", "Entrypoint publication does not accept query parameters.", requestID)
		}
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	revision, ok := requireRoutingRevision(response, request, requestID, "ep")
	if !ok {
		return
	}
	action, operationPath, reason := "unpublish", routingEntrypointsPath+"/{entrypointId}:unpublish", "unpublish Entrypoint"
	var dependencies []accesscontrol.ScopedTarget
	if publish {
		action, operationPath, reason = "publish", routingEntrypointsPath+"/{entrypointId}:publish", "publish Entrypoint"
		entrypoint, err := routes.service.GetEntrypoint(request.Context(), namespaceID, entrypointID)
		if err != nil {
			writeRoutingDomainError(response, err, requestID, false)
			return
		}
		dependencies, err = routingEntrypointRecordDependencies(namespaceID, entrypoint)
		if err != nil {
			writeProviderError(response, http.StatusServiceUnavailable, "routing_unavailable", "Routing Management is unavailable.", requestID)
			return
		}
	}
	_, err := routes.authorize(request.Context(), session, namespaceID,
		routes.operation(managementapi.MethodPOST, operationPath),
		routingEntrypointTargets(namespaceID, entrypointID, dependencies), nil)
	if err != nil {
		writeRoutingAuthorizationError(response, err, requestID, false)
		return
	}
	payload := struct {
		EntrypointID     string `json:"entrypointId"`
		Action           string `json:"action"`
		ExpectedRevision int64  `json:"expectedRevision"`
	}{entrypointID, action, revision}
	endpoint := routingEntrypointsPath + "/" + entrypointID + ":" + action
	command, ok := routes.bindCommand(response, request, requestID, namespaceID, session, endpoint, payload)
	if !ok {
		return
	}
	if routes.replayEntrypointPublication(response, request, requestID, command, revision) {
		return
	}
	var receipt routingmanagement.RevisionReceipt
	if publish {
		_, receipt, err = routes.service.PublishEntrypoint(request.Context(), namespaceID, entrypointID, revision,
			routingMutationContext(session, requestID, reason, &command))
	} else {
		_, receipt, err = routes.service.UnpublishEntrypoint(request.Context(), namespaceID, entrypointID, revision,
			routingMutationContext(session, requestID, reason, &command))
	}
	if err != nil {
		if routes.replayEntrypointPublication(response, request, requestID, command, revision) {
			return
		}
		writeRoutingDomainError(response, err, requestID, true)
		return
	}
	response.Header().Set(managementapi.HeaderETag, routingETag("ep", receipt.ResourceRevision))
	response.Header().Set("Location", managementapi.BasePath+"/operations/"+receipt.OperationID)
	writeRoutingOperationReceipt(response, receipt, true, requestID)
}

func (routes *RoutingRoutes) resolveEntrypoint(
	response http.ResponseWriter, request *http.Request, requestID, entrypointID string,
) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Entrypoint resolve does not accept query parameters.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	var body managementapi.RoutingResolveRequest
	if !decodeRoutingBody(response, request, requestID, &body) {
		return
	}
	operation := routes.operation(managementapi.MethodPOST, routingEntrypointsPath+"/{entrypointId}:resolve")
	baseTargets := routingEntrypointTargets(namespaceID, entrypointID, nil)
	baseConditions := map[string]bool{
		"entrypoint_resolution_matched": false, "routing_subject_supplied": false,
		"routing_context_override_requested": false,
	}
	if _, err := routes.authorize(request.Context(), session, namespaceID, operation, baseTargets, baseConditions); err != nil {
		writeRoutingAuthorizationError(response, err, requestID, true)
		return
	}
	resolution, err := routes.service.ResolveEntrypoint(
		request.Context(), namespaceID, entrypointID, body.Path, routingClaimValues(body.Claims),
	)
	if err != nil {
		writeRoutingDomainError(response, err, requestID, false)
		return
	}
	if resolution.Outcome == routingsnapshot.ResolveMatched {
		dependencies, dependencyErr := routingResolutionDependencies(namespaceID, resolution)
		if dependencyErr != nil {
			writeProviderError(response, http.StatusServiceUnavailable, "routing_unavailable", "Routing Management is unavailable.", requestID)
			return
		}
		targets := routingEntrypointTargets(namespaceID, entrypointID, dependencies)
		conditions := map[string]bool{
			"entrypoint_resolution_matched": true, "routing_subject_supplied": false,
			"routing_context_override_requested": false,
		}
		if _, err := routes.authorize(request.Context(), session, namespaceID, operation, targets, conditions); err != nil {
			writeRoutingAuthorizationError(response, err, requestID, true)
			return
		}
	}
	writeProviderJSON(response, http.StatusOK, routingResolveResponseDTO(resolution), requestID)
}

func (routes *RoutingRoutes) loadEntrypointInputDependencies(
	ctx context.Context, namespaceID string, input routingmanagement.EntrypointInput,
) ([]accesscontrol.ScopedTarget, error) {
	dependencies := make(map[string]accesscontrol.ScopedTarget)
	for _, rule := range input.Rules {
		if routingmanagement.ValidateResourceID(rule.RecipeID) != nil {
			return nil, routingmanagement.ErrInvalid
		}
		recipe, err := routes.service.GetRecipe(ctx, namespaceID, rule.RecipeID)
		if err != nil || recipe.ID != rule.RecipeID {
			return nil, fmt.Errorf("%w: Recipe dependency is unavailable", routingmanagement.ErrInvalid)
		}
		key := "recipe\x00" + recipe.ID
		dependencies[key] = routingTarget(namespaceID, accesscontrol.ScopeResourceRecipe, recipe.ID)
		for _, assignmentSet := range rule.Assignments {
			for _, assignment := range assignmentSet.Models {
				if routingmanagement.ValidateResourceID(assignment.ModelID) != nil {
					return nil, routingmanagement.ErrInvalid
				}
				model, err := routes.service.GetModel(ctx, namespaceID, assignment.ModelID)
				if err != nil || model.ID != assignment.ModelID {
					return nil, fmt.Errorf("%w: Model dependency is unavailable", routingmanagement.ErrInvalid)
				}
				key := "model\x00" + model.ID
				dependencies[key] = routingTarget(namespaceID, accesscontrol.ScopeResourceModel, model.ID)
			}
		}
	}
	if len(dependencies) == 0 {
		return nil, routingmanagement.ErrInvalid
	}
	return sortedRoutingDependencies(dependencies), nil
}

func routingEntrypointRecordDependencies(
	namespaceID string, entrypoint routingmanagement.Entrypoint,
) ([]accesscontrol.ScopedTarget, error) {
	dependencies := make(map[string]accesscontrol.ScopedTarget)
	for _, rule := range entrypoint.Current.Rules {
		if routingmanagement.ValidateResourceID(rule.RecipeID) != nil || rule.RecipeRevision <= 0 {
			return nil, routingmanagement.ErrPublication
		}
		dependencies["recipe\x00"+rule.RecipeID] = routingTarget(
			namespaceID, accesscontrol.ScopeResourceRecipe, rule.RecipeID,
		)
		for _, assignmentSet := range rule.Assignments {
			for _, assignment := range assignmentSet.Models {
				if routingmanagement.ValidateResourceID(assignment.ModelID) != nil || assignment.ModelRevision <= 0 {
					return nil, routingmanagement.ErrPublication
				}
				dependencies["model\x00"+assignment.ModelID] = routingTarget(
					namespaceID, accesscontrol.ScopeResourceModel, assignment.ModelID,
				)
			}
		}
	}
	if len(dependencies) == 0 {
		return nil, routingmanagement.ErrPublication
	}
	return sortedRoutingDependencies(dependencies), nil
}

func routingResolutionDependencies(
	namespaceID string, resolution routingsnapshot.Resolution,
) ([]accesscontrol.ScopedTarget, error) {
	if resolution.Rule == nil || resolution.Recipe == nil ||
		resolution.Rule.RecipeID != resolution.Recipe.ID ||
		resolution.Rule.RecipeRevision != resolution.Recipe.Revision {
		return nil, routingmanagement.ErrPublication
	}
	dependencies := map[string]accesscontrol.ScopedTarget{
		"recipe\x00" + resolution.Recipe.ID: routingTarget(
			namespaceID, accesscontrol.ScopeResourceRecipe, resolution.Recipe.ID,
		),
	}
	for _, assignmentSet := range resolution.Rule.Assignments {
		for _, assignment := range assignmentSet.Models {
			if routingmanagement.ValidateResourceID(assignment.ModelID) != nil || assignment.ModelRevision <= 0 {
				return nil, routingmanagement.ErrPublication
			}
			dependencies["model\x00"+assignment.ModelID] = routingTarget(
				namespaceID, accesscontrol.ScopeResourceModel, assignment.ModelID,
			)
		}
	}
	return sortedRoutingDependencies(dependencies), nil
}

func sortedRoutingDependencies(source map[string]accesscontrol.ScopedTarget) []accesscontrol.ScopedTarget {
	keys := make([]string, 0, len(source))
	for key := range source {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	result := make([]accesscontrol.ScopedTarget, len(keys))
	for index, key := range keys {
		result[index] = source[key]
	}
	return result
}

func routingEntrypointTargets(
	namespaceID, entrypointID string, dependencies []accesscontrol.ScopedTarget,
) map[string][]accesscontrol.ScopedTarget {
	target := routingNamespaceTarget(namespaceID)
	if entrypointID != "" {
		target = routingTarget(namespaceID, accesscontrol.ScopeResourceEntrypoint, entrypointID)
	}
	result := map[string][]accesscontrol.ScopedTarget{"target": {target}}
	if len(dependencies) != 0 {
		result["all_dependencies"] = append([]accesscontrol.ScopedTarget(nil), dependencies...)
	}
	return result
}

func (routes *RoutingRoutes) replayEntrypointResource(
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
	receipt, err := routingResourceReplay(stored, "routing_entrypoint", resourceID)
	if err != nil {
		writeRoutingDomainError(response, err, requestID, false)
		return true
	}
	id := stored.Resource.ResourceID
	response.Header().Set("Location", routingEntrypointsPath+"/"+id)
	response.Header().Set(managementapi.HeaderETag, routingETag("ep", receipt.ResourceRevision))
	writeRoutingResourceReceipt(response, stored.Resource.ResponseStatus, "routing_entrypoint", id, receipt, true, requestID)
	return true
}

func (routes *RoutingRoutes) replayEntrypointPublication(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
	command managementcommand.Command,
	expectedRevision int64,
) bool {
	stored, found, err := routes.lookupCommand(request.Context(), command)
	if err != nil {
		writeRoutingDomainError(response, err, requestID, false)
		return true
	}
	if !found {
		return false
	}
	receipt, err := routingOperationReplay(stored, true)
	if err != nil {
		writeRoutingDomainError(response, err, requestID, false)
		return true
	}
	response.Header().Set(managementapi.HeaderETag, routingETag("ep", expectedRevision+1))
	response.Header().Set("Location", managementapi.BasePath+"/operations/"+receipt.OperationID)
	writeRoutingOperationReceipt(response, receipt, true, requestID)
	return true
}
