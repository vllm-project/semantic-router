package managementserver

import (
	"encoding/json"
	"errors"
	"io"
	"mime"
	"net/http"
	"strconv"
	"strings"

	accesspostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apikeymanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
)

func decodeAPIKeyBody(response http.ResponseWriter, request *http.Request, requestID string, target any) bool {
	if request.ContentLength > maximumAPIKeyBodyBytes {
		writeProviderError(response, http.StatusRequestEntityTooLarge, "invalid_request", "Request body is too large.", requestID)
		return false
	}
	mediaType, parameters, err := mime.ParseMediaType(request.Header.Get("Content-Type"))
	if err != nil || mediaType != managementapi.JSONMediaType ||
		(len(parameters) != 0 && (len(parameters) != 1 || !strings.EqualFold(parameters["charset"], "utf-8"))) {
		writeProviderError(response, http.StatusUnsupportedMediaType, "unsupported_media_type", "Use the Management API media type.", requestID)
		return false
	}
	request.Body = http.MaxBytesReader(response, request.Body, maximumAPIKeyBodyBytes)
	decoder := json.NewDecoder(request.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		var maximum *http.MaxBytesError
		if errors.As(err, &maximum) {
			writeProviderError(response, http.StatusRequestEntityTooLarge, "invalid_request", "Request body is too large.", requestID)
		} else {
			writeProviderError(response, http.StatusBadRequest, "invalid_request", "Request body is invalid.", requestID)
		}
		return false
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Request body is invalid.", requestID)
		return false
	}
	return true
}

func requireAPIKeyRevision(response http.ResponseWriter, request *http.Request, requestID string) (uint64, bool) {
	values := request.Header.Values(managementapi.HeaderIfMatch)
	if len(values) != 1 {
		writeProviderError(response, http.StatusPreconditionRequired, "precondition_required", "If-Match is required.", requestID)
		return 0, false
	}
	match := apiKeyETagPattern.FindStringSubmatch(values[0])
	if len(match) != 2 {
		writeProviderError(response, http.StatusBadRequest, "invalid_precondition", "If-Match is invalid.", requestID)
		return 0, false
	}
	revision, err := strconv.ParseUint(match[1], 10, 64)
	if err != nil || revision == 0 {
		writeProviderError(response, http.StatusBadRequest, "invalid_precondition", "If-Match is invalid.", requestID)
		return 0, false
	}
	return revision, true
}

func apiKeyETag(revision uint64) string {
	return `"key:` + strconv.FormatUint(revision, 10) + `"`
}

func noRequestBody(response http.ResponseWriter, request *http.Request, requestID string) bool {
	if request.ContentLength > 0 || len(request.TransferEncoding) != 0 {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "This operation does not accept a request body.", requestID)
		return false
	}
	return true
}

func writeAPIKeySecret(response http.ResponseWriter, status int, result apikeymanagement.SecretMutationResult, requestID string) {
	setProviderResponseHeaders(response, requestID)
	response.Header().Set("Content-Type", managementapi.JSONMediaType)
	response.Header().Set(managementapi.HeaderETag, apiKeyETag(result.ResponseRevision))
	setIdempotencyReplayHeader(response, result.Replayed)
	response.WriteHeader(status)
	_, _ = response.Write(append(append([]byte(nil), result.CanonicalJSON...), '\n'))
}

func writeAPIKeyError(response http.ResponseWriter, err error, requestID string) {
	switch {
	case errors.Is(err, apikeymanagement.ErrInvalidRequest):
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "API key request is invalid.", requestID)
	case errors.Is(err, apikeymanagement.ErrNotFound):
		writeProviderError(response, http.StatusNotFound, "not_found", "API key not found.", requestID)
	case errors.Is(err, managementcommand.ErrConflict):
		writeProviderError(response, http.StatusConflict, "idempotency_conflict", "Idempotency-Key was already used for a different request.", requestID)
	case errors.Is(err, accesspostgres.ErrAlreadyExists):
		writeProviderError(response, http.StatusConflict, "already_exists", "An API key with this identity already exists.", requestID)
	case errors.Is(err, apikeymanagement.ErrRevisionConflict):
		writeProviderError(response, http.StatusPreconditionFailed, "revision_conflict", "The API key changed. Refresh and retry.", requestID)
	case errors.Is(err, apikeymanagement.ErrLastActiveCredential):
		writeProviderError(response, http.StatusConflict, "last_active_credential", "Disable the API key before deleting its final credential.", requestID)
	case errors.Is(err, apikeymanagement.ErrSecretResultExpired):
		writeProviderError(response, http.StatusGone, "secret_result_expired", "The one-time secret delivery window expired.", requestID)
	case errors.Is(err, apikeymanagement.ErrRevealDisabled):
		writeProviderError(response, http.StatusConflict, "reveal_disabled", "This credential cannot be revealed.", requestID)
	case errors.Is(err, apikeymanagement.ErrCredentialUnavailable):
		writeProviderError(response, http.StatusConflict, "credential_unavailable", "The API-key credential cannot perform this action.", requestID)
	default:
		writeProviderError(response, http.StatusServiceUnavailable, "api_key_service_unavailable", "API key service is unavailable.", requestID)
	}
}
