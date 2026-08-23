package managementserver

import (
	"errors"
	"net/http"

	accesspostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
	credentialmanagement "github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential/management"
)

func writeProviderCredentialDomainError(response http.ResponseWriter, err error, requestID string) {
	switch {
	case errors.Is(err, credentialmanagement.ErrInvalidRequest), errors.Is(err, providercatalog.ErrInvalidRequest):
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Provider credential request is invalid.", requestID)
	case errors.Is(err, credentialmanagement.ErrUnsafeOrigin):
		writeProviderError(response, http.StatusBadRequest, "origin_denied", "The selected base URL is not allowed.", requestID)
	case errors.Is(err, accesspostgres.ErrNotFound), errors.Is(err, providercatalog.ErrNotFound):
		writeProviderError(response, http.StatusNotFound, "not_found", "Provider credential not found.", requestID)
	case errors.Is(err, managementcommand.ErrConflict):
		writeProviderError(response, http.StatusConflict, "idempotency_conflict", "Idempotency-Key was already used for a different request.", requestID)
	case errors.Is(err, accesspostgres.ErrAlreadyExists):
		writeProviderError(response, http.StatusConflict, "already_exists", "A Provider credential with this name already exists.", requestID)
	case errors.Is(err, accesspostgres.ErrRevisionConflict):
		writeProviderError(response, http.StatusPreconditionFailed, "revision_conflict", "The Provider credential changed. Refresh and retry.", requestID)
	case errors.Is(err, credentialmanagement.ErrProviderMismatch), errors.Is(err, providercredential.ErrUnavailable):
		writeProviderError(response, http.StatusConflict, "credential_unavailable", "The Provider credential cannot perform this action.", requestID)
	default:
		writeProviderError(response, http.StatusServiceUnavailable, "credential_service_unavailable", "Provider credential service is unavailable.", requestID)
	}
}
