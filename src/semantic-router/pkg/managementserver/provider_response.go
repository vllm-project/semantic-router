package managementserver

import (
	"encoding/json"
	"errors"
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providerdiscovery"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
)

func setProviderResponseHeaders(response http.ResponseWriter, requestID string) {
	response.Header().Set(managementapi.HeaderRequestID, requestID)
	response.Header().Set("Cache-Control", "no-store")
	response.Header().Set("Vary", "Accept, Authorization")
	response.Header().Set("X-Content-Type-Options", "nosniff")
}

func writeProviderJSON(response http.ResponseWriter, status int, value any, requestID string) {
	setProviderResponseHeaders(response, requestID)
	response.Header().Set("Content-Type", managementapi.JSONMediaType)
	payload, err := json.Marshal(value)
	if err != nil {
		writeProviderError(response, http.StatusInternalServerError, "internal_error", "Request could not be completed.", requestID)
		return
	}
	response.WriteHeader(status)
	_, _ = response.Write(append(payload, '\n'))
}

func writeProviderError(response http.ResponseWriter, status int, code, message, requestID string) {
	writeProviderJSON(response, status, managementapi.ErrorResponse{Error: managementapi.APIError{
		Code: code, Message: message, RequestID: requestID,
	}}, requestID)
}

func writeCatalogDomainError(response http.ResponseWriter, err error, requestID string) {
	switch {
	case errors.Is(err, providercatalog.ErrNotFound):
		writeProviderError(response, http.StatusNotFound, "not_found", "Provider not found.", requestID)
	case errors.Is(err, providercatalog.ErrStaleCursor):
		writeProviderError(response, http.StatusConflict, "stale_catalog", "The Provider catalog changed. Start a new page request.", requestID)
	case errors.Is(err, providercatalog.ErrInvalidRequest), errors.Is(err, providercatalog.ErrInvalidCursor):
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Provider request is invalid.", requestID)
	case errors.Is(err, providercatalog.ErrDiscoveryUnsupported):
		writeProviderError(response, http.StatusConflict, "discovery_unsupported", "This Provider does not support discovery.", requestID)
	default:
		writeProviderError(response, http.StatusServiceUnavailable, "catalog_unavailable", "Provider catalog is unavailable.", requestID)
	}
}

func writeDiscoveryDomainError(response http.ResponseWriter, err error, requestID string) {
	switch {
	case errors.Is(err, providerdiscovery.ErrInvalidRequest):
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Discovery request is invalid.", requestID)
	case errors.Is(err, providerdiscovery.ErrCredentialMismatch):
		writeProviderError(response, http.StatusForbidden, "credential_unavailable", "The selected credential cannot be used.", requestID)
	case errors.Is(err, providerdiscovery.ErrUpstream), errors.Is(err, providerdiscovery.ErrInvalidResponse):
		writeProviderError(response, http.StatusBadGateway, "provider_unavailable", "Provider discovery is temporarily unavailable.", requestID)
	default:
		writeProviderError(response, http.StatusServiceUnavailable, "discovery_unavailable", "Provider discovery is unavailable.", requestID)
	}
}
