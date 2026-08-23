package managementserver

import (
	"errors"
	"net/http"
	"regexp"
	"strconv"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policybulk"
)

var operationETagPattern = regexp.MustCompile(`^"operation:([1-9][0-9]*)"$`)

func operationETag(version uint64) string {
	return `"operation:` + strconv.FormatUint(version, 10) + `"`
}

func requireOperationRevision(response http.ResponseWriter, request *http.Request, requestID string) (uint64, bool) {
	values := request.Header.Values(managementapi.HeaderIfMatch)
	if len(values) != 1 {
		writeProviderError(response, http.StatusPreconditionRequired, "precondition_required", "If-Match is required.", requestID)
		return 0, false
	}
	match := operationETagPattern.FindStringSubmatch(values[0])
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

func operationRequestHasBody(request *http.Request) bool {
	return request == nil || request.ContentLength != 0 || len(request.TransferEncoding) != 0 ||
		request.Header.Get("Content-Type") != ""
}

func writeOperationError(response http.ResponseWriter, err error, requestID string) {
	switch {
	case errors.Is(err, policybulk.ErrInvalidRequest):
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Operation request is invalid.", requestID)
	case errors.Is(err, policybulk.ErrNotFound):
		writeProviderError(response, http.StatusNotFound, "not_found", "Operation not found.", requestID)
	case errors.Is(err, policybulk.ErrRevisionConflict):
		writeProviderError(response, http.StatusPreconditionFailed, "revision_conflict", "The operation changed. Refresh and retry.", requestID)
	case errors.Is(err, policybulk.ErrConflict):
		writeProviderError(response, http.StatusConflict, "operation_conflict", "The operation cannot be cancelled in its current state.", requestID)
	default:
		writeProviderError(response, http.StatusServiceUnavailable, "operation_service_unavailable", "Operation state is unavailable.", requestID)
	}
}

func writeOperationDetailError(response http.ResponseWriter, err error, requestID string) {
	if errors.Is(err, errOperationDetailNotFound) {
		writeProviderError(response, http.StatusNotFound, "not_found", "Operation not found.", requestID)
		return
	}
	writeProviderError(response, http.StatusServiceUnavailable, "operation_service_unavailable", "Operation state is unavailable.", requestID)
}
