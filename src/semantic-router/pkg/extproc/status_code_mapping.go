package extproc

import (
	"net/http"

	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
)

func statusCodeToImmediateResponseCode(statusCode int) typev3.StatusCode {
	// Immediate responses are final HTTP responses. Check the range before
	// narrowing to int32, and use Envoy's enum rather than a partial local list.
	if statusCode >= http.StatusOK && statusCode <= 599 {
		if _, defined := typev3.StatusCode_name[int32(statusCode)]; defined {
			return typev3.StatusCode(statusCode)
		}
	}
	return typev3.StatusCode_InternalServerError
}

// An error body must never be paired with a success or redirect status.
func errorResponseStatusCode(statusCode int) int {
	code := int(statusCodeToImmediateResponseCode(statusCode))
	if code < http.StatusBadRequest {
		return http.StatusInternalServerError
	}
	return code
}
