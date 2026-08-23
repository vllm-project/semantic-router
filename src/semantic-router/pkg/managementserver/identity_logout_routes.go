package managementserver

import (
	"errors"
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity"
)

func (routes *IdentityLifecycleRoutes) backchannelLogout(response http.ResponseWriter, request *http.Request) {
	requestID := managementRequestID(request)
	if !validLifecycleRequest(response, request, requestID, false) {
		return
	}
	if request.TLS == nil && !routes.allowPlaintext {
		writeProviderError(response, http.StatusBadRequest, "tls_required", "TLS is required.", requestID)
		return
	}
	if len(request.Header.Values("Authorization")) != 0 {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Authorization credentials are not accepted.", requestID)
		return
	}
	var body managementapi.BackchannelLogoutRequest
	if !decodeIdentityBody(response, request, requestID, &body) {
		return
	}
	token := body.LogoutToken
	body.LogoutToken = ""
	defer zeroString(&token)
	if !canonicalUUID(body.IssuerID) {
		writeProviderError(response, http.StatusUnauthorized, "unauthenticated", "Back-channel logout authentication failed.", requestID)
		return
	}
	result, err := routes.service.BackchannelLogout(
		request.Context(), body.IssuerID, token, requestID, routes.now().UTC(),
	)
	if err != nil {
		writeBackchannelLogoutError(response, err, requestID)
		return
	}
	writeProviderJSON(response, http.StatusOK, managementapi.BackchannelLogoutResponse{
		Applied: !result.Replayed, Replayed: result.Replayed,
	}, requestID)
}

func writeBackchannelLogoutError(response http.ResponseWriter, err error, requestID string) {
	switch {
	case errors.Is(err, managementauth.ErrAuthenticationDenied):
		writeProviderError(response, http.StatusUnauthorized, "unauthenticated", "Back-channel logout authentication failed.", requestID)
	case errors.Is(err, managementidentity.ErrBackchannelReplay):
		writeProviderError(response, http.StatusConflict, "logout_token_reuse", "Logout token identifier conflicts with an earlier request.", requestID)
	default:
		writeProviderError(response, http.StatusServiceUnavailable, "logout_unavailable", "Back-channel logout is unavailable.", requestID)
	}
}
