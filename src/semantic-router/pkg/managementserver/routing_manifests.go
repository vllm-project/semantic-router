package managementserver

import (
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
)

func (routes *RoutingRoutes) importManifest(response http.ResponseWriter, request *http.Request, requestID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Routing import does not accept query parameters.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	var body managementapi.RoutingManifestImportRequest
	if !decodeRoutingBody(response, request, requestID, &body) {
		return
	}
	if body.Manifest == "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Routing manifest is required.", requestID)
		return
	}
	prepared, err := routes.service.PrepareManifest(request.Context(), namespaceID, []byte(body.Manifest))
	if err != nil {
		writeRoutingDomainError(response, err, requestID, false)
		return
	}
	credentials := prepared.CredentialIDs
	targets := map[string][]accesscontrol.ScopedTarget{
		"target": {routingNamespaceTarget(namespaceID)},
	}
	if len(credentials) != 0 {
		targets["credential"] = make([]accesscontrol.ScopedTarget, 0, len(credentials))
		for _, credentialID := range credentials {
			targets["credential"] = append(targets["credential"], routingTarget(
				namespaceID, accesscontrol.ScopeResourceProviderCredential, credentialID,
			))
		}
	}
	_, err = routes.authorize(request.Context(), session, namespaceID,
		routes.operation(managementapi.MethodPOST, routingImportsPath),
		targets, map[string]bool{"provider_credential_referenced": len(credentials) != 0})
	if err != nil {
		writeRoutingAuthorizationError(response, err, requestID, false)
		return
	}
	revision, ok := requireRoutingRevision(response, request, requestID, "routing")
	if !ok {
		return
	}
	command, ok := routes.bindCommand(response, request, requestID, namespaceID, session, routingImportsPath, body)
	if !ok {
		return
	}
	if stored, found, lookupErr := routes.lookupCommand(request.Context(), command); lookupErr != nil {
		writeRoutingDomainError(response, lookupErr, requestID, false)
		return
	} else if found {
		receipt, replayErr := routingOperationReplay(stored, true)
		if replayErr != nil {
			writeRoutingDomainError(response, replayErr, requestID, false)
			return
		}
		diff, diffErr := routes.commandResults.LookupManifestDiff(request.Context(), namespaceID, receipt.OperationID)
		if diffErr != nil {
			writeRoutingDomainError(response, diffErr, requestID, false)
			return
		}
		writeRoutingManifestResult(response, routingmanagement.ManifestImportResult{Diff: diff, Receipt: receipt}, requestID, false)
		return
	}
	var boundCommand *managementcommand.Command
	if !body.DryRun {
		boundCommand = &command
	}
	result, err := routes.service.ImportManifest(request.Context(), namespaceID, routingmanagement.ManifestImportRequest{
		Prepared: prepared, DryRun: body.DryRun, ExpectedRevision: revision,
	}, routingMutationContext(session, requestID, "import routing manifest", boundCommand))
	if err != nil {
		writeRoutingDomainError(response, err, requestID, true)
		return
	}
	writeRoutingManifestResult(response, result, requestID, body.DryRun)
}

func (routes *RoutingRoutes) exportCurrentManifest(response http.ResponseWriter, request *http.Request, requestID string) {
	if request.URL.RawQuery != "" || !rejectRoutingBody(response, request, requestID) {
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	_, err := routes.authorize(request.Context(), session, namespaceID,
		routes.operation(managementapi.MethodGET, routingCurrentExportPath),
		map[string][]accesscontrol.ScopedTarget{
			"target": {routingNamespaceTarget(namespaceID)},
		}, nil)
	if err != nil {
		writeRoutingAuthorizationError(response, err, requestID, false)
		return
	}
	document, revision, err := routes.service.ExportCurrentManifest(request.Context(), namespaceID)
	if err != nil {
		writeRoutingDomainError(response, err, requestID, false)
		return
	}
	setProviderResponseHeaders(response, requestID)
	response.Header().Set("Content-Type", managementapi.YAMLMediaType+"; charset=utf-8")
	response.Header().Set(managementapi.HeaderETag, routingETag("routing", revision))
	response.Header().Set("Cache-Control", "no-store")
	response.WriteHeader(http.StatusOK)
	_, _ = response.Write(document)
}

func writeRoutingManifestResult(response http.ResponseWriter, result routingmanagement.ManifestImportResult, requestID string, dryRun bool) {
	dto := managementapi.RoutingManifestImportResult{Diff: routingManifestDiffDTO(result.Diff), Replayed: result.Receipt.Replayed}
	status := http.StatusOK
	if !dryRun {
		status = http.StatusAccepted
		dto.OperationID = result.Receipt.OperationID
		desired := publicRevision(result.Receipt.DesiredRevision)
		dto.DesiredRevision = &desired
		response.Header().Set("Location", managementapi.BasePath+"/operations/"+result.Receipt.OperationID)
		response.Header().Set(managementapi.HeaderETag, routingETag("routing", result.Receipt.DesiredRevision))
		setIdempotencyReplayHeader(response, result.Receipt.Replayed)
	}
	writeProviderJSON(response, status, dto, requestID)
}

func routingManifestDiffDTO(value routingmanagement.ManifestDiff) managementapi.RoutingManifestDiff {
	convert := func(source routingmanagement.ManifestResourceDiff) managementapi.RoutingManifestResourceDiff {
		return managementapi.RoutingManifestResourceDiff{
			Create: append([]string{}, source.Create...), Update: append([]string{}, source.Update...),
			Disable: append([]string{}, source.Disable...),
		}
	}
	return managementapi.RoutingManifestDiff{Models: convert(value.Models), Recipes: convert(value.Recipes), Entrypoints: convert(value.Entrypoints)}
}
