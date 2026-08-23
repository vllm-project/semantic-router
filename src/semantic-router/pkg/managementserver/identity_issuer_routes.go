package managementserver

import (
	"encoding/json"
	"net/http"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth/issuerverifier"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity"
)

func (routes *IdentityLifecycleRoutes) trustedIssuers(response http.ResponseWriter, request *http.Request) {
	requestID := managementRequestID(request)
	if !validLifecycleRequest(response, request, requestID, request.Method == http.MethodGet) {
		return
	}
	switch request.Method {
	case http.MethodGet:
		if !noRequestBody(response, request, requestID) {
			return
		}
		pageRequest, ok := identityPageRequest(response, request, requestID)
		if !ok {
			return
		}
		_, ok = routes.authenticatedAndAuthorized(response, request, requestID, managementapi.MethodGET, trustedIssuerPath)
		if !ok {
			return
		}
		page, err := routes.service.ListTrustedIdentityIssuers(request.Context(), pageRequest)
		if err != nil {
			writeIdentityError(response, err, requestID)
			return
		}
		data := make([]managementapi.TrustedIdentityIssuer, len(page.Items))
		for index := range page.Items {
			data[index] = trustedIssuerDTO(page.Items[index])
		}
		writeProviderJSON(response, http.StatusOK, managementapi.TrustedIdentityIssuerPage{
			Data: data, Page: identityPageInfo(page.NextCursor, pageRequest.Limit),
		}, requestID)
	case http.MethodPost:
		routes.createTrustedIssuer(response, request, requestID)
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
}

func (routes *IdentityLifecycleRoutes) createTrustedIssuer(response http.ResponseWriter, request *http.Request, requestID string) {
	session, ok := routes.authenticatedAndAuthorized(response, request, requestID, managementapi.MethodPOST, trustedIssuerPath)
	if !ok {
		return
	}
	key, ok := requireIdempotencyKey(response, request, requestID)
	if !ok {
		return
	}
	var body managementapi.TrustedIdentityIssuerCreateRequest
	if !decodeIdentityBody(response, request, requestID, &body) {
		return
	}
	canonical, err := json.Marshal(body)
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Request is invalid.", requestID)
		return
	}
	now := routes.now().UTC()
	command, err := routes.commands.Bind(
		managementcommand.ClusterCommandScope(), session.Session.PrincipalID, trustedIssuerPath,
		string(key), canonical, now, now.Add(identityCommandTTL),
	)
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Request is invalid.", requestID)
		return
	}
	result, err := routes.service.CreateTrustedIdentityIssuer(request.Context(), managementidentity.CreateTrustedIdentityIssuer{
		Issuer: managementidentity.TrustedIdentityIssuer{
			ID: uuid.NewString(), Issuer: body.Issuer, Kind: issuerverifier.IssuerKind(body.Kind),
			DiscoveryURL: body.DiscoveryURL, JWKSURL: body.JWKSURL,
			Audiences:        append([]string(nil), body.Audiences...),
			ClaimMapping:     cloneDTOStringMap(body.ClaimMapping),
			AssuranceMapping: cloneDTOStringMap(body.AssuranceMapping),
			Status:           managementauth.ResourceActive, Revision: 1, CreatedAt: now, UpdatedAt: now,
		},
		Command: command,
		Actor:   identityActor(request, session, requestID, "Create trusted identity issuer"),
	})
	if err != nil {
		writeIdentityError(response, err, requestID)
		return
	}
	writeIdentityMutation(response, result.Result, trustedIssuerETag, trustedIssuerPath, requestID)
}

func (routes *IdentityLifecycleRoutes) trustedIssuer(response http.ResponseWriter, request *http.Request) {
	requestID := managementRequestID(request)
	if !validLifecycleRequest(response, request, requestID, false) {
		return
	}
	issuerID := request.PathValue("issuerId")
	if !canonicalUUID(issuerID) {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	method := managementapi.HTTPMethod(request.Method)
	session, ok := routes.authenticatedAndAuthorized(response, request, requestID, method, trustedIssuerPath+"/{issuerId}")
	if !ok {
		return
	}
	switch request.Method {
	case http.MethodGet:
		if !noRequestBody(response, request, requestID) {
			return
		}
		issuer, err := routes.service.GetTrustedIdentityIssuer(request.Context(), issuerID)
		if err != nil {
			writeIdentityError(response, err, requestID)
			return
		}
		response.Header().Set(managementapi.HeaderETag, trustedIssuerETag(issuer.Revision))
		writeProviderJSON(response, http.StatusOK, map[string]any{"data": trustedIssuerDTO(issuer)}, requestID)
	case http.MethodPatch:
		revision, ok := identityRevision(response, request, requestID, "tii")
		if !ok {
			return
		}
		var body managementapi.TrustedIdentityIssuerPatchRequest
		if !decodeIdentityBody(response, request, requestID, &body) {
			return
		}
		var status *managementauth.ResourceStatus
		if body.Status != nil {
			value := managementauth.ResourceStatus(*body.Status)
			status = &value
		}
		result, err := routes.service.UpdateTrustedIdentityIssuer(request.Context(), managementidentity.UpdateTrustedIdentityIssuer{
			ID: issuerID, ExpectedRevision: revision,
			DiscoveryURL: body.DiscoveryURL, JWKSURL: body.JWKSURL,
			Audiences: body.Audiences, ClaimMapping: body.ClaimMapping,
			AssuranceMapping: body.AssuranceMapping, Status: status,
			Actor: identityActor(request, session, requestID, body.Reason),
		})
		if err != nil {
			writeIdentityError(response, err, requestID)
			return
		}
		writeIdentityMutation(response, result.Result, trustedIssuerETag, trustedIssuerPath, requestID)
	case http.MethodDelete:
		if !noRequestBody(response, request, requestID) {
			return
		}
		revision, ok := identityRevision(response, request, requestID, "tii")
		if !ok {
			return
		}
		result, err := routes.service.DeleteTrustedIdentityIssuer(
			request.Context(), issuerID, revision,
			identityActor(request, session, requestID, "Delete trusted identity issuer"),
		)
		if err != nil {
			writeIdentityError(response, err, requestID)
			return
		}
		response.Header().Set(managementapi.HeaderETag, trustedIssuerETag(result.Result.Revision))
		response.WriteHeader(http.StatusNoContent)
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
}

func (routes *IdentityLifecycleRoutes) refreshTrustedIssuerKeys(response http.ResponseWriter, request *http.Request) {
	requestID := managementRequestID(request)
	if !validLifecycleRequest(response, request, requestID, false) {
		return
	}
	issuerID, pathOK := lifecycleActionID(request.URL.Path, trustedIssuerPath, ":refresh-keys")
	if !pathOK {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	session, ok := routes.authenticatedAndAuthorized(response, request, requestID, managementapi.MethodPOST, trustedIssuerPath+"/{issuerId}:refresh-keys")
	if !ok {
		return
	}
	key, ok := requireIdempotencyKey(response, request, requestID)
	if !ok {
		return
	}
	var body managementapi.TrustedIdentityIssuerRefreshRequest
	if !decodeIdentityBody(response, request, requestID, &body) {
		return
	}
	canonical, err := json.Marshal(body)
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Request is invalid.", requestID)
		return
	}
	now := routes.now().UTC()
	command, err := routes.commands.Bind(
		managementcommand.ClusterCommandScope(), session.Session.PrincipalID, request.URL.Path,
		string(key), canonical, now, now.Add(identityCommandTTL),
	)
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Request is invalid.", requestID)
		return
	}
	result, err := routes.service.RefreshTrustedIdentityIssuer(request.Context(), managementidentity.RefreshTrustedIdentityIssuer{
		ID: issuerID, Command: command,
		Actor: identityActor(request, session, requestID, body.Reason),
	})
	if err != nil {
		writeIdentityError(response, err, requestID)
		return
	}
	writeIdentityMutation(response, result.Result, trustedIssuerETag, trustedIssuerPath, requestID)
}
