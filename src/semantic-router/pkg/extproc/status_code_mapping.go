package extproc

import (
	"net/http"

	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
)

func statusCodeToImmediateResponseCode(statusCode int) typev3.StatusCode {
	switch statusCode {
	case http.StatusOK, http.StatusBadRequest, http.StatusNotFound, http.StatusInternalServerError:
		return statusCodeToEnum(statusCode)
	}

	switch statusCode {
	case http.StatusUnauthorized:
		return typev3.StatusCode_Unauthorized
	case http.StatusForbidden:
		return typev3.StatusCode_Forbidden
	case http.StatusMethodNotAllowed:
		return typev3.StatusCode_MethodNotAllowed
	case http.StatusRequestEntityTooLarge:
		return typev3.StatusCode_PayloadTooLarge
	case http.StatusUnprocessableEntity:
		return typev3.StatusCode_UnprocessableEntity
	case http.StatusTooManyRequests:
		return typev3.StatusCode_TooManyRequests
	case http.StatusBadGateway:
		return typev3.StatusCode_BadGateway
	case http.StatusServiceUnavailable:
		return typev3.StatusCode_ServiceUnavailable
	default:
		return typev3.StatusCode_OK
	}
}
